"""The BacLCT tracking widget."""

from __future__ import annotations

import re
import tempfile
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import torch
from magicgui.application import use_app
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    LineEdit,
    ProgressBar,
    PushButton,
    SpinBox,
    create_widget,
)
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QSizePolicy
from superqt import QCollapsible

from baclct.napari._layers import add_result_layers
from baclct.napari._radius import radius_preview_shapes, resolve_radius_px
from baclct.napari._worker import CancelToken, TrackParams, TrackResult, tracking_worker
from baclct.utils.pretrained import MODEL_SPECS, resolve_model_dir

if TYPE_CHECKING:
    import napari.viewer

RADIUS_LAYER_NAME = "BacLCT search radius"
DEFAULT_RADIUS = "2.5x"
DEFAULT_SUFFIX = {"flat": "_tracks", "ctc": "_RES"}
# sentinel for radius text that is neither a number nor an 'Nx' spec
INVALID_RADIUS = object()

# (label, value) pairs. values stay EdgePruneArg members plus 'off'. 'gt' is left out,
# it is not available during inference
PRUNE_CHOICES = (
    ("Default", "default"),
    ("Overlap", "overlap"),
    ("Overlap (expanded)", "dilated_overlap"),
    ("Ellipse", "ellipse"),
    ("Circle", "radius"),
    ("Off", "off"),
)
PRUNE_HELP = {
    "default": "Pruning determined by the pretrained model.",
    "overlap": "Cells are linked only if their masks overlap.",
    "dilated_overlap": "Cells are linked if their thickness-expanded masks overlap.",
    "ellipse": "Cells are linked within an elliptical search region.",
    "radius": "Cells are linked within a circular search region.",
    "off": "No pruning: every cell within the search radius stays a candidate.",
}
# 'overlap' is an area fraction, the others scale a cell axis
PRUNE_PARAM_SPECS = {
    "overlap": {"label": "Minimum overlap", "min": 0.0, "max": 1.0, "step": 0.05},
    "default": {"label": "Factor", "min": 0.1, "max": 100.0, "step": 0.5},
}
PRUNE_PARAM_DEFAULTS = {"overlap": 0.1}

# fallback for 'ellipse' and 'radius'. dilated_overlap's multiplier is fixed as
# DILATED_OVERLAP_RADIUS_MULTIPLIER in features/graph.py
DEFAULT_PRUNE_PARAM = 3.0
DILATED_OVERLAP_COLUMN = "thickness"
PARAMETERLESS_PRUNE_METHODS = ("default", "dilated_overlap", "off")
# swatch colors match radius_preview_shapes
RADIUS_LEGEND = (
    '<span style="color:yellow">●</span> search region&nbsp;&nbsp;&nbsp;'
    '<span style="color:cyan">●</span> candidates (pruned)'
)

# the first tracks both bright field and phase contrast
PRETRAINED_MODELS = ("baclct_track", "baclct_spore_classification_bf", "baclct_toiam_pc")
CACHE_MODES = ("In memory", "Session", "Directory")

# section header lines and tooltips are worded from docs/guide/inference.rst and
# docs/guide/cli.rst
TRACKING_HELP = (
    "Raise or lower the thresholds to link more cells or suppress division detection. "
    "Increase the search radius to link over larger distances, narrow it to save memory."
)
RUNTIME_HELP = (
    "More workers and a larger batch speed up inference, at the cost of memory. For "
    "small sequences not using workers is often faster."
)
# Qt renders a plain-text tooltip on a single line
TOOLTIP_WIDTH = 90
_HELP_TEXTS = {
    "channel": (
        "Channel to track, for an image with a channel axis the segmentation does not "
        "have."
    ),
    "model_source": "A pretrained model, or a directory holding one.",
    "model_path": "Directory of a trained model, as written by a training run.",
    "thr_corr": (
        "Correspondence threshold: lower it to recover missed tracks, raise it to "
        "mitigate false correspondences."
    ),
    "thr_div": (
        "Division threshold: lower it to recover missed divisions, raise it to mitigate "
        "false division detection, or set it to 1.0 to disable division detection."
    ),
    "segmentation_correction": (
        "Correct over- and undersegmentation from the division and correspondence "
        "predictions. Mainly developed for rod-shaped bacteria, so disable it for round "
        "objects or very dense regions."
    ),
    "radius": (
        "Largest displacement of an object between two frames, in pixels or as a "
        "multiple of the expected cell size ('2.5x'). Empty keeps the trained value."
    ),
    "prune_param": (
        "Parameter of the pruning method: an area overlap threshold for 'Overlap', a "
        "scaling factor for 'Ellipse' and 'Circle'."
    ),
    "preview_cell": (
        "Cell to outline, by label. 0 takes the one closest to the frame center."
    ),
    "show_radius": "Outline the search region of one cell in the current frame.",
    "num_workers": (
        "Worker processes for feature extraction. Worth raising on long or very dense "
        "sequences. More than 8 was not worth it in tests."
    ),
    "batch_size": (
        "Subgraphs per prediction batch. Raise it on a machine with memory to spare."
    ),
    "device": "Device to run on. Detected by default.",
    "use_patches": (
        "Tile each frame instead of building one whole-frame graph. Needed when a full "
        "frame exceeds memory."
    ),
    "patch_size": "Edge length of a tile, in pixels.",
    "cache_mode": (
        "Caches graph and features (under 1 GB) to save memory and reuse them across "
        "runs. "
        "'In memory' keeps no files, 'Session' lasts until napari closes, and "
        "'Directory' reuses an existing cache."
    ),
    "cache_dir": "Directory the features are cached in.",
    "use_output": "Write the results to disk as well as adding them as layers.",
    "output_dir": "Directory to write the results to.",
    "export_format": (
        "Layout of the results: 'ctc' for a sequence directory, 'flat' for "
        "'{id}_tracks.csv' plus '{id}_tracks.tif'."
    ),
    "export_suffix": "Appended to the sequence id during export.",
}
HELP = {key: textwrap.fill(text, TOOLTIP_WIDTH) for key, text in _HELP_TEXTS.items()}

# materialising a lazy layer this large will likely exhaust memory
LARGE_INPUT_BYTES = 8 * 1024**3


def _available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name).strip("_") or "sequence"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _shape(layer) -> tuple[int, ...]:
    """Shape of a layer's data without materialising it, which dask arrays would."""
    data = layer.data
    if isinstance(data, list):  # multiscale: the full-resolution level leads
        data = data[0]
    return tuple(data.shape)


def _wrapping_label(value: str = "", *, visible: bool = True) -> Label:
    """A Label that breaks lines, so long help text does not pin the dock pane wide."""
    label = Label(value=value, visible=visible)
    label.native.setWordWrap(True)
    # magicgui's Qt Label is Fixed/Fixed, so it never shrinks below its unwrapped text
    policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    label.native.setSizePolicy(policy)
    label.native.setMinimumWidth(1)
    # without this the label asks for its one-word-per-line height at a width of 1px
    label.native.setMinimumHeight(label.native.fontMetrics().height())
    return label


class BacLCTWidget(Container):
    """Track bacteria in an image + segmentation layer pair and add the results back."""

    # napari matches the parameter name and the annotation string, rename neither
    def __init__(self, napari_viewer: napari.viewer.Viewer):
        """Build the widget and bind it to `napari_viewer`."""
        self._viewer = napari_viewer
        self._worker = None
        self._cancel: CancelToken | None = None
        # last run's outcome, kept across _refresh()
        self._outcome: str = ""
        # segmentation the running job started on, in case the selection changes
        self._run_layer: Labels | None = None
        self._result_layers: list = []
        self._session_cache: tempfile.TemporaryDirectory | None = None
        self._seeded_model: str | None = None

        self._title = Label(value="<b>BacLCT</b>")
        self._image_layer = create_widget(
            annotation="napari.layers.Image", label="Image layer"
        )
        self._labels_layer = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        self._channel = SpinBox(
            label="Channel", min=0, value=0, visible=False, tooltip=HELP["channel"]
        )

        self._model_source = ComboBox(
            label="Model",
            choices=["Pretrained", "From path"],
            value="Pretrained",
            tooltip=HELP["model_source"],
        )
        self._pretrained = ComboBox(
            label="", choices=PRETRAINED_MODELS, value=PRETRAINED_MODELS[0]
        )
        self._model_help = _wrapping_label()
        self._model_path = FileEdit(
            label="Model path", mode="d", visible=False, tooltip=HELP["model_path"]
        )
        self._resolved_model = Label(value="")

        self._thr_corr = FloatSpinBox(
            label="Correspondence",
            value=0.50,
            min=0.0,
            max=1.0,
            step=0.05,
            tooltip=HELP["thr_corr"],
        )
        self._thr_div = FloatSpinBox(
            label="Division",
            value=0.30,
            min=0.0,
            max=1.0,
            step=0.05,
            tooltip=HELP["thr_div"],
        )
        self._seg_correction = CheckBox(
            value=True,
            text="Segmentation correction",
            tooltip=HELP["segmentation_correction"],
        )

        self._radius = LineEdit(
            label="Search radius", value=DEFAULT_RADIUS, tooltip=HELP["radius"]
        )
        # tooltip is set per method, in _refresh()
        self._prune_method = ComboBox(
            label="Edge pruning", choices=PRUNE_CHOICES, value="default"
        )
        self._prune_param = FloatSpinBox(
            value=DEFAULT_PRUNE_PARAM,
            tooltip=HELP["prune_param"],
            **PRUNE_PARAM_SPECS["default"],
        )
        self._preview_cell = SpinBox(
            label="Preview cell",
            value=0,
            min=0,
            max=1_000_000,
            tooltip=HELP["preview_cell"],
        )
        self._show_radius = PushButton(
            text="Show graph radius", tooltip=HELP["show_radius"]
        )
        self._radius_legend = Label(value=RADIUS_LEGEND, visible=False)

        self._num_workers = SpinBox(
            label="Workers", value=0, min=0, max=8, tooltip=HELP["num_workers"]
        )
        self._batch_size = SpinBox(
            label="Batch size", value=1, min=1, max=16, tooltip=HELP["batch_size"]
        )
        self._device = ComboBox(
            label="Device", choices=_available_devices(), tooltip=HELP["device"]
        )

        self._use_patches = CheckBox(
            value=False, text="Tile into patches", tooltip=HELP["use_patches"]
        )
        self._patch_size = SpinBox(
            label="Patch size",
            value=256,
            min=16,
            max=4096,
            visible=False,
            tooltip=HELP["patch_size"],
        )

        self._cache_mode = ComboBox(
            label="Feature cache",
            choices=CACHE_MODES,
            value="Session",
            tooltip=HELP["cache_mode"],
        )
        self._cache_dir = FileEdit(
            label="", mode="d", visible=False, tooltip=HELP["cache_dir"]
        )
        self._use_output = CheckBox(
            value=False, text="Output directory", tooltip=HELP["use_output"]
        )
        self._output_dir = FileEdit(
            label="", mode="d", visible=False, tooltip=HELP["output_dir"]
        )
        self._export_format = ComboBox(
            label="Format",
            choices=["flat", "ctc"],
            value="flat",
            visible=False,
            tooltip=HELP["export_format"],
        )
        self._export_suffix = LineEdit(
            label="Suffix",
            value=DEFAULT_SUFFIX["flat"],
            visible=False,
            tooltip=HELP["export_suffix"],
        )

        self._run = PushButton(text="Start tracking")
        self._cancel_button = PushButton(text="Cancel", enabled=False)
        self._progress = ProgressBar(value=0, min=0, max=100, visible=False)
        self._status = _wrapping_label(visible=False)

        # parameters of the collapsible section, kept out of the flat list
        self._tracking_group = Container(
            widgets=[
                self._thr_corr,
                self._thr_div,
                self._seg_correction,
                self._radius,
                self._prune_method,
                self._prune_param,
                self._preview_cell,
                self._show_radius,
            ]
        )
        self._runtime_group = Container(
            widgets=[self._num_workers, self._batch_size, self._device]
        )
        self._additional_section = self._build_additional_parameters()

        # 'row' goes through magicgui, which puts a label column in front of it. 'text'
        # is inserted into the layout directly and spans the pane
        items = [
            ("text", self._title.native),
            ("row", self._image_layer),
            ("row", self._channel),
            ("row", self._labels_layer),
            ("row", self._model_source),
            ("row", self._pretrained),
            ("row", self._model_path),
            ("text", self._model_help.native),
            ("text", self._resolved_model.native),
            ("text", self._additional_section),
            ("row", self._use_patches),
            ("row", self._patch_size),
            ("row", self._cache_mode),
            ("row", self._cache_dir),
            ("row", self._use_output),
            ("row", self._output_dir),
            ("row", self._export_format),
            ("row", self._export_suffix),
            ("row", self._run),
            ("row", self._cancel_button),
            ("row", self._progress),
            ("text", self._status.native),
        ]
        super().__init__(
            widgets=[widget for kind, widget in items if kind == "row"], scrollable=True
        )
        # inner content widget, for the layout edits below. napari is handed the scroll
        # area instead, via the `native` override
        self._content = self._widget._mgui_get_native_widget()
        self._content._magic_widget = self
        # ascending, so each insertion finds everything before it already in place.
        # magicgui counts only its own widgets, so it must not add or remove any after
        layout = self._content.layout()
        for index, (kind, widget) in enumerate(items):
            if kind == "text":
                layout.insertWidget(index, widget)
        # without a trailing stretch the rows spread over the viewport height
        layout.addStretch(1)
        self._align_labels()

        self._image_layer.changed.connect(self._refresh)
        self._labels_layer.changed.connect(self._refresh)
        self._pretrained.changed.connect(self._refresh)
        self._model_source.changed.connect(self._refresh)
        self._prune_method.changed.connect(self._refresh)
        self._use_patches.changed.connect(self._refresh)
        self._cache_mode.changed.connect(self._refresh)
        self._use_output.changed.connect(self._refresh)
        self._export_format.changed.connect(self._on_export_format_changed)
        self._show_radius.changed.connect(self._on_show_radius)
        self._run.changed.connect(self._on_run)
        self._cancel_button.changed.connect(self._on_cancel)

        self._select_defaults()
        self._refresh()

    @property
    def native(self):
        """The scroll area, so a tall widget scrolls in the dock instead of overflowing.

        napari docks whatever `native` returns; magicgui's default is the inner content
        widget, letting the dock grow past the screen when a section expands.
        """
        return self._widget._mgui_get_root_native_widget()

    def _build_additional_parameters(self) -> QCollapsible:
        """Fold the advanced tracking and runtime parameters into one collapsed section.

        They appear as plain titled groups inside it, not nested collapsibles. Tiling,
        cache, and output stay flat above the run button.
        """
        # magicgui pads containers by 11px with 6px between rows, too loose under a header
        for group in (self._tracking_group, self._runtime_group):
            group.margins = (0, 0, 0, 0)
            group.native.layout().setSpacing(2)

        self._tracking_header = Label(value="<b>Tracking parameters</b>")
        self._tracking_help = _wrapping_label(TRACKING_HELP)
        self._runtime_header = Label(value="<b>Runtime parameters</b>")
        self._runtime_help = _wrapping_label(RUNTIME_HELP)

        section = QCollapsible("Additional parameters")
        section.setCollapsedIcon("▸")
        section.setExpandedIcon("▾")
        # both default to Qt's 9px, which would indent the section against the form
        section.layout().setContentsMargins(0, 0, 0, 6)
        section.content().layout().setContentsMargins(0, 0, 0, 0)
        section.addWidget(self._tracking_header.native)
        section.addWidget(self._tracking_help.native)
        section.addWidget(self._tracking_group.native)
        section.addWidget(self._radius_legend.native)
        section.addWidget(self._runtime_header.native)
        section.addWidget(self._runtime_help.native)
        section.addWidget(self._runtime_group.native)
        section.collapse(animate=False)
        return section

    def _align_labels(self) -> None:
        """Give every row of the form and of both groups the same label column.

        magicgui unifies label widths per container and skips the widgets passed to
        `__init__` altogether, which leaves each row indented by its own label.
        """
        measure = use_app().get_obj("get_text_width")
        rows = [
            widget
            for container in (self, self._tracking_group, self._runtime_group)
            for widget in container
            if widget._labeled_widget() is not None
        ]
        # count both pruning parameter labels, so switching the method cannot move the
        # column
        labels = [widget.label for widget in rows]
        labels += [spec["label"] for spec in PRUNE_PARAM_SPECS.values()]
        width = max(measure(label) for label in labels)
        for widget in rows:
            widget._labeled_widget().label_width = width

    # ------------------------------------------------------------------ state

    def _select_defaults(self) -> None:
        """Preselect the first image layer and a segmentation layer named like a mask."""
        images = [ly for ly in self._viewer.layers if isinstance(ly, Image)]
        if images:
            self._image_layer.value = images[0]

        labels = [ly for ly in self._viewer.layers if isinstance(ly, Labels)]
        masks = [ly for ly in labels if "mask" in ly.name.lower()]
        if masks or labels:
            self._labels_layer.value = (masks or labels)[0]

    def apply_settings(
        self,
        *,
        model: str | Path | None = None,
        channel: int | None = None,
        graph_search_radius: str | None = None,
        prune_edges_by: str | None = None,
        prune_param: float | str | None = None,
        thr_corr: float | None = None,
        thr_div: float | None = None,
        segmentation_correction: bool | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        patch_size: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        output_dir: Path | None = None,
        export_format: str | None = None,
        export_suffix: str | None = None,
    ) -> None:
        """Preset the fields, leaving anything passed as `None` untouched.

        `None` means "leave alone" and never "unset": the CLI defaults `thr_corr`,
        `thr_div`, and `segmentation_correction` to `None` to follow the model, and a
        spinbox cannot show that, so an unpassed flag must not overwrite the widget's
        value.

        Assignment order matters: setting a value triggers `changed`, which runs
        `_refresh()` before the next line. The model reseeds the radius, the pruning
        method resets its parameter, and the export format resets the suffix, so each
        is set before the field it would otherwise overwrite.
        """
        if model is not None:
            if str(model) in PRETRAINED_MODELS:
                self._model_source.value = "Pretrained"
                self._pretrained.value = str(model)
            else:
                self._model_source.value = "From path"
                self._model_path.value = Path(model)
        # after the model, which reseeds the radius
        if graph_search_radius is not None:
            # the CLI spells the trained value "trained", the field leaves it empty
            self._radius.value = (
                "" if graph_search_radius == "trained" else graph_search_radius
            )

        if prune_edges_by is not None:
            self._prune_method.value = prune_edges_by

        takes_param = self._prune_method.value not in PARAMETERLESS_PRUNE_METHODS
        if prune_param is not None and takes_param:
            self._prune_param.value = float(prune_param)

        if export_format is not None:
            self._export_format.value = export_format
        # after the format, which resets the suffix
        if export_suffix is not None:
            self._export_suffix.value = export_suffix

        if channel is not None:
            self._channel.value = channel
        if thr_corr is not None:
            self._thr_corr.value = thr_corr
        if thr_div is not None:
            self._thr_div.value = thr_div
        if segmentation_correction is not None:
            self._seg_correction.value = segmentation_correction
        if device is not None:
            self._device.value = device
        if num_workers is not None:
            self._num_workers.value = num_workers
        if batch_size is not None:
            self._batch_size.value = batch_size

        if cache_dir is not None:
            self._apply_cache_dir(cache_dir)
        if patch_size is not None:
            self._use_patches.value = True
            self._patch_size.value = patch_size
        if output_dir is not None:
            self._use_output.value = True
            self._output_dir.value = Path(output_dir)

        self._refresh()

    def _apply_cache_dir(self, cache_dir: str | Path) -> None:
        """Map the CLI's --cache-dir vocabulary onto the cache mode combo.

        'temp' becomes 'Session', the closest the widget has. The match is loose: the
        CLI's temporary directory lasts one run, the widget's lasts the napari session,
        which is what makes a re-track with changed thresholds cheap.
        """
        if cache_dir == "memory":
            self._cache_mode.value = "In memory"
        elif cache_dir == "temp":
            self._cache_mode.value = "Session"
        else:
            self._cache_mode.value = "Directory"
            self._cache_dir.value = Path(cache_dir)

    @property
    def model_name(self) -> str:
        """The selected pretrained model."""
        return str(self._pretrained.value)

    def is_multichannel(self) -> bool:
        """Whether the image layer carries a channel axis the segmentation does not."""
        image, masks = self._image_layer.value, self._labels_layer.value
        if image is None or masks is None:
            return False
        return len(_shape(image)) == len(_shape(masks)) + 1

    def _validate(self) -> str | None:
        """Why tracking cannot start, or `None` when it can."""
        if self._labels_layer.value is None:
            return "Select a segmentation layer."
        # both shipped models consume image features
        if self._image_layer.value is None:
            return "BacLCT requires an image layer."
        if self._model_source.value == "From path" and not self._model_path.value:
            return "Select a model directory."
        # an untouched FileEdit reads as Path('.'), napari's working directory
        if (
            self._cache_mode.value == "Directory"
            and Path(self._cache_dir.value) == Path()
        ):
            return "Select a feature cache directory."

        if self._search_radius() is INVALID_RADIUS:
            return "Search radius must be a number or a multiple like '2.5x'."

        if self.is_multichannel():
            return None

        image, masks = _shape(self._image_layer.value), _shape(self._labels_layer.value)
        if image != masks:
            return f"Image {image} and segmentation {masks} shapes differ."
        return None

    def _refresh(self) -> None:
        """Sync widget visibility and the run button with the current selection."""
        self._seed_radius_from_model()
        from_path = self._model_source.value == "From path"
        self._model_path.visible = from_path
        self._pretrained.visible = not from_path
        self._model_help.visible = not from_path
        self._model_help.value = (
            "" if from_path else MODEL_SPECS[self.model_name].description
        )
        self._resolved_model.value = (
            "" if from_path else f"Model: <code>{self.model_name}</code>"
        )
        self._resolved_model.visible = not from_path

        self._prune_method.tooltip = textwrap.fill(
            PRUNE_HELP[self._prune_method.value], TOOLTIP_WIDTH
        )
        self._prune_param.visible = (
            self._prune_method.value not in PARAMETERLESS_PRUNE_METHODS
        )
        self._sync_prune_param()
        self._patch_size.visible = self._use_patches.value
        self._cache_dir.visible = self._cache_mode.value == "Directory"
        self._output_dir.visible = self._use_output.value
        self._export_format.visible = self._use_output.value
        self._export_suffix.visible = self._use_output.value

        self._channel.visible = self.is_multichannel()

        reason = self._validate()
        running = self._worker is not None
        self._run.enabled = reason is None and not running
        self._show_radius.enabled = self._labels_layer.value is not None and not running
        if not running:
            self._set_status(reason or self._outcome or "")

    def _set_status(self, text: str) -> None:
        """Show a message, or take the row back when there is nothing to report."""
        self._status.value = text
        self._status.visible = bool(text)

    def _sync_prune_param(self) -> None:
        """Give the pruning parameter the range and label its method actually uses."""
        method = self._prune_method.value
        spec = PRUNE_PARAM_SPECS.get(method, PRUNE_PARAM_SPECS["default"])
        if self._prune_param.label == spec["label"]:
            return

        self._prune_param.label = spec["label"]
        # widen before narrowing, so the value never sits outside both ranges
        self._prune_param.min = min(self._prune_param.min, spec["min"])
        self._prune_param.max = max(self._prune_param.max, spec["max"])
        self._prune_param.value = PRUNE_PARAM_DEFAULTS.get(method, DEFAULT_PRUNE_PARAM)
        self._prune_param.min, self._prune_param.max = spec["min"], spec["max"]
        self._prune_param.step = spec["step"]
        # magicgui re-unifies the group whenever one of its labels changes
        self._align_labels()

    # ------------------------------------------------------------------ inputs

    def _image_data(self) -> da.Array | np.ndarray:
        """Image stack as `(T, ...)`, with the channel axis sliced off if present.

        The layer's own array is passed through without materializing it: `track()`
        takes dask, prefers it for long sequences, and slicing a channel off keeps it
        lazy.
        """
        image = self._image_layer.value.data
        if self.is_multichannel():
            image = image[:, self._channel.value]
        return image

    def _output_name(self) -> str:
        """Layer name, used both for on-disk export naming and as the cache key.

        The feature cache itself now guards against reusing another movie's features
        (its sidecar records a content hash of the images/masks it was built from), so
        the widget no longer needs to bake one into the cache key.
        """
        return _slug(self._labels_layer.value.name)

    def _on_export_format_changed(self) -> None:
        self._export_suffix.value = DEFAULT_SUFFIX[self._export_format.value]

    def _build_params(self) -> TrackParams:
        masks = self._labels_layer.value.data
        images = self._image_data()

        output_name = self._output_name()
        return TrackParams(
            images=images,
            masks=masks,
            model=self._model_arg(),
            sequence_id=output_name,
            export_suffix=self._export_suffix.value or None,
            graph_search_radius=self._search_radius(),
            prune_edges_by=self._prune_edges_by(),
            thr_corr=self._thr_corr.value,
            thr_div=self._thr_div.value,
            segmentation_correction=self._seg_correction.value,
            cache_dir=self._cache_dir_param(),
            output_dir=(Path(self._output_dir.value) if self._use_output.value else None),
            export_format=self._export_format.value,
            device=self._device.value,
            num_workers=self._num_workers.value,
            batch_size=self._batch_size.value,
            patch_size=(self._patch_size.value if self._use_patches.value else None),
        )

    def _model_arg(self) -> str | Path:
        """The selected model, as a registry name or a directory."""
        if self._model_source.value == "From path":
            return Path(self._model_path.value)
        return self.model_name

    def _cache_dir_param(self) -> Path | None:
        """The feature cache implied by the selected mode.

        'Session' is a temporary directory of the widget rather than of one run, so a
        re-track reuses the features that do not depend on the model: node features, the
        deep embeddings, and the edges of an unchanged graph configuration.
        """
        if self._cache_mode.value == "Session":
            if self._session_cache is None:
                self._session_cache = tempfile.TemporaryDirectory(prefix="baclct-napari-")
            return Path(self._session_cache.name)
        if self._cache_mode.value == "Directory":
            return Path(self._cache_dir.value)
        return None

    def _prune_edges_by(self) -> str | tuple[str, float | str] | None:
        """The pruning override, or `None` to keep what the model was trained with."""
        method = self._prune_method.value
        if method == "default":
            return None
        if method == "off":
            return "off"
        if method == "dilated_overlap":
            return (method, DILATED_OVERLAP_COLUMN)
        return (method, self._prune_param.value)

    def _model_config(self):
        """The selected model's training config, or `None` when it is not local.

        Never downloads: the config is read to fill in defaults, which must not cost a
        model download while the user is still picking one.
        """
        from omegaconf import OmegaConf

        try:
            return OmegaConf.load(
                resolve_model_dir(self._model_arg(), download=False)
                / ".hydra"
                / "config.yaml"
            )
        except Exception:
            return None

    def _default_search_radius(self) -> str | None:
        """The search radius the selected model was trained with, as field text.

        Returns `None` for a model that is not available locally or was trained on a
        radius range.
        """
        from omegaconf import OmegaConf

        cfg = self._model_config()
        if cfg is None:
            return None
        radius = OmegaConf.select(cfg, "data.graph_search_radius")
        if isinstance(radius, (int, float)):
            return str(int(radius))
        return radius if isinstance(radius, str) else None

    def _seed_radius_from_model(self) -> None:
        """Preset the radius field from the model, unless the model is unchanged.

        The field is passed to `track()` as an override, so its own default would replace
        the trained radius. `DEFAULT_RADIUS` used when the model's own is unknown.
        """
        model = str(self._model_arg())
        if model == self._seeded_model:
            return
        self._seeded_model = model
        radius = self._default_search_radius()
        self._radius.value = radius if radius is not None else DEFAULT_RADIUS

    def _default_prune_edges_by(self) -> tuple[str, float | str] | None:
        """The pruning the selected model was trained with, so 'Default' can be previewed.

        Returns `None` when the model is not available locally or records no pruning. A
        per-cell column such as 'thickness' is passed on as-is, since the preview
        resolves it per cell.
        """
        from omegaconf import OmegaConf

        cfg = self._model_config()
        if cfg is None:
            return None
        prune = OmegaConf.select(cfg, "features.edges.prune_edges_by")
        if prune is None or isinstance(prune, str) or len(prune) < 2:
            return None
        param = prune[1]
        if isinstance(param, str):
            return (str(prune[0]), param)
        try:
            return (str(prune[0]), float(param))
        except (TypeError, ValueError):
            return None

    def _search_radius(self) -> int | str | None:
        """The radius field as pixels, a relative str, or `None` when blank.

        The field is free text, so a plain number is parsed to pixels rather than passed
        on as a string. A trailing 'x' (e.g. '2.5x') stays a factor. Anything else returns
        `INVALID_RADIUS` for `_validate` to report.
        """
        text = str(self._radius.value).strip()
        if not text:
            return None
        if text.endswith("x"):
            return text if _is_number(text[:-1]) else INVALID_RADIUS
        return int(round(float(text))) if _is_number(text) else INVALID_RADIUS

    # ------------------------------------------------------------------ actions

    def _on_show_radius(self) -> None:
        """Outline the search region of one cell in the current frame."""
        # passed through unmaterialised, the helpers index one frame at a time
        masks = self._labels_layer.value.data
        if masks.ndim != 3:
            show_warning("The radius preview supports 2D + time segmentations only.")
            return

        radius = self._search_radius()
        if radius is None or radius is INVALID_RADIUS:
            show_warning("Enter a search radius, in pixels or as a multiple like '2.5x'.")
            return

        t = int(self._viewer.dims.current_step[0])
        # 'Default' defers to the model, so resolve its pruning for the preview. 'Off'
        # has no region and leaves the search circle as the only shape
        prune = self._prune_edges_by()
        from_default = prune is None
        preview_prune = self._default_prune_edges_by() if from_default else prune
        if isinstance(preview_prune, str):
            preview_prune = None
        label = self._preview_cell.value or None
        try:
            radius_px = resolve_radius_px(masks, radius)
            shapes, shape_types, edge_colors = radius_preview_shapes(
                masks, t, radius_px, preview_prune, label=label
            )
        except ValueError as err:
            show_warning(str(err))
            return

        if not shapes:
            show_warning(f"No cells in frame {t}.")
            return

        # remove all previous previews, however many have piled up
        for layer in [ly for ly in self._viewer.layers if ly.name == RADIUS_LAYER_NAME]:
            self._viewer.layers.remove(layer)
        self._viewer.add_shapes(
            shapes,
            shape_type=shape_types,
            name=RADIUS_LAYER_NAME,
            edge_color=edge_colors,
            face_color="transparent",
            edge_width=1,
            scale=self._labels_layer.value.scale,
        )
        self._radius_legend.visible = True
        detail = ""
        if from_default and preview_prune is not None:
            method, param = preview_prune
            shown = param if isinstance(param, str) else f"{param:g}"
            detail = f" (model default: {method}, {shown})"
        show_info(f"Search radius: {self._radius.value} -> {radius_px} px{detail}")

    def _on_run(self) -> None:
        """Start tracking in the background."""
        reason = self._validate()
        if reason is not None:
            show_warning(reason)
            return

        self._run_layer = self._labels_layer.value
        params = self._build_params()
        # only an eager stack is resident, a lazy layer is read frame by frame
        nbytes = sum(
            stack.nbytes
            for stack in (params.images, params.masks)
            if not isinstance(stack, da.Array)
        )
        if nbytes > LARGE_INPUT_BYTES:
            show_warning(
                f"Loading {nbytes / 1024**3:.1f} GiB into memory. This may be slow."
            )

        self._cancel = CancelToken()
        self._worker = tracking_worker(params, self._cancel)
        self._worker.yielded.connect(self._on_progress)
        self._worker.returned.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._worker.finished.connect(self._on_done)

        self._outcome = ""
        self._progress.value = 0
        self._progress.visible = True
        self._set_status("Starting...")
        self._cancel_button.enabled = True
        self._refresh()
        self._worker.start()

    def _on_cancel(self) -> None:
        """Ask the running worker to stop at its next progress report.

        Only the token is set. `worker.quit()` would suppress the `returned` signal, so
        the run would end without ever reporting that it was cancelled.
        """
        if self._cancel is not None:
            self._cancel.cancel()
        self._set_status("Cancelling...")
        self._cancel_button.enabled = False

    def _on_progress(self, event) -> None:
        if event is None:  # keepalive yield
            return
        self._progress.value = int(event.fraction * 100)
        self._set_status(event.label)

    def _on_finished(self, result: TrackResult | None) -> None:
        # a cancelled run returns None instead of erroring
        if result is None:
            self._outcome = "Cancelled."
            self._set_status(self._outcome)
            return

        labels = self._run_layer
        self._clear_results()
        self._result_layers = add_result_layers(
            self._viewer,
            result.tracks,
            result.masks_tracked,
            labels.name if labels is not None else "BacLCT",
            scale=labels.scale if labels is not None else None,
        )
        # the tracked-mask layer overlaps the input segmentation
        if labels is not None:
            labels.visible = False
        self._progress.value = 100
        n_tracks = result.tracks["label"].n_unique()
        self._outcome = f"Done: {n_tracks} tracks."
        self._set_status(self._outcome)

    def _clear_results(self) -> None:
        """Drop the layers of the previous run, so tracking again replaces them."""
        for layer in self._result_layers:
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        self._result_layers = []

    def _on_error(self, err: Exception) -> None:
        self._outcome = f"Failed: {err}"
        self._set_status(self._outcome)
        show_warning(f"BacLCT tracking failed: {err}")

    def _on_done(self) -> None:
        self._worker = None
        self._cancel = None
        self._cancel_button.enabled = False
        self._progress.visible = False
        self._refresh()
