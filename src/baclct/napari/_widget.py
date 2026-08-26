"""The BacLCT tracking widget."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
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
# returned by _search_radius for free text that is neither a number nor an 'Nx' spec
INVALID_RADIUS = object()

# edge pruning offered as (label, value). 'off' is what `BacLCT.track` reads as no
# pruning at all, the others must stay valid non-'gt' EdgePruneArg members, since 'gt'
# is not available during inference
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
    "dilated_overlap": (
        "Cells are linked if their expanded masks overlap. Expansion is done based on a "
        "cell's thickness."
    ),
    "ellipse": "Cells are linked within an elliptical search region.",
    "radius": "Cells are linked within a circular search region.",
    "off": "No pruning: every cell within the search radius stays a candidate.",
}
# 'overlap' filters on a fraction of the source cell's area, the others scale a cell axis,
# so the same spinbox has to mean two different things
PRUNE_PARAM_SPECS = {
    "overlap": {"label": "Minimum overlap", "min": 0.0, "max": 1.0, "step": 0.05},
    "default": {"label": "Factor", "min": 0.1, "max": 100.0, "step": 0.5},
}
PRUNE_PARAM_DEFAULTS = {"overlap": 0.1}

# dilated_overlap's dilation multiplier is fixed (DILATED_OVERLAP_RADIUS_MULTIPLIER in
# features/graph.py); DEFAULT_PRUNE_PARAM below is only the 'ellipse'/'radius' fallback.
DEFAULT_PRUNE_PARAM = 3.0
DILATED_OVERLAP_COLUMN = "thickness"
# methods with no parameter for the spinbox to set
PARAMETERLESS_PRUNE_METHODS = ("default", "dilated_overlap", "off")
# swatch colors match the preview shapes drawn by radius_preview_shapes
RADIUS_LEGEND = (
    '<span style="color:yellow">●</span> search region&nbsp;&nbsp;&nbsp;'
    '<span style="color:cyan">●</span> candidates (pruned)'
)

# pretrained models, in the order they are offered. the first is the safe default: it
# tracks both bright field and phase contrast.
PRETRAINED_MODELS = ("baclct_track", "baclct_spore_classification_bf", "baclct_toiam_pc")
CACHE_MODES = ("In memory", "Session", "Directory")
CACHE_HELP = (
    "Cache stores the sequence graph and features locally (usually <1GB), which saves "
    "RAM and lets you reuse the same features and graphs across models and parameters. "
    "'Session' keeps cache until napari closes, 'Directory' re-uses an existing cache "
    "(e.g., from a previous training run), and 'In memory' keeps everything in RAM."
)
TRACKING_HELP = (
    "Raise or lower the thresholds to link more cells or suppress division detection. "
    "Increase the search radius to link over larger distances, narrow it to save memory."
)
RUNTIME_HELP = (
    "Parallel jobs speed up feature extraction on long movies. Small (or even adverse) "
    "effect for short movies (<1K objects). Encoder workers crop cell patches while the "
    "encoder runs, where 2 is fastest and more is slower again. Both increase memory "
    "requirements and startup time."
)

# materialising a lazily loaded layer this large will likely exhaust memory
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


def _wrapping_label(value: str = "") -> Label:
    """A Label that breaks lines, so long help text does not pin the dock pane wide."""
    label = Label(value=value)
    label.native.setWordWrap(True)
    return label


class BacLCTWidget(Container):
    """Track bacteria in an image + segmentation layer pair and add the results back."""

    # napari injects the viewer by matching the parameter name or the literal annotation
    # string, so neither may be renamed or shortened.
    def __init__(self, napari_viewer: napari.viewer.Viewer):
        """Build the widget and bind it to `napari_viewer`."""
        self._viewer = napari_viewer
        self._worker = None
        self._cancel: CancelToken | None = None
        # outcome of the last run, kept so _refresh() does not clear it
        self._outcome: str = ""
        # the segmentation the running job was started on, so a layer switch mid-run
        # cannot misname or misplace its results
        self._run_layer: Labels | None = None
        self._result_layers: list = []
        # feature cache shared by every run of this session, released with the widget
        self._session_cache: tempfile.TemporaryDirectory | None = None
        # model the search radius was last seeded from
        self._seeded_model: str | None = None

        self._image_layer = create_widget(
            annotation="napari.layers.Image", label="Image layer"
        )
        self._labels_layer = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        self._channel = SpinBox(label="Channel", min=0, value=0, visible=False)

        self._model_source = ComboBox(
            label="Model", choices=["Pretrained", "From path"], value="Pretrained"
        )
        self._pretrained = ComboBox(
            label="", choices=PRETRAINED_MODELS, value=PRETRAINED_MODELS[0]
        )
        self._model_help = _wrapping_label()
        self._model_path = FileEdit(label="Model path", mode="d", visible=False)
        self._resolved_model = Label(value="")

        self._thr_corr = FloatSpinBox(
            label="Correspondence threshold", value=0.50, min=0.0, max=1.0, step=0.05
        )
        self._thr_div = FloatSpinBox(
            label="Division threshold", value=0.30, min=0.0, max=1.0, step=0.05
        )
        self._seg_correction = CheckBox(value=True, text="Segmentation correction")

        self._radius = LineEdit(label="Search radius", value=DEFAULT_RADIUS)
        self._prune_method = ComboBox(
            label="Edge pruning", choices=PRUNE_CHOICES, value="default"
        )
        self._prune_help = _wrapping_label()
        self._prune_param = FloatSpinBox(
            value=DEFAULT_PRUNE_PARAM, **PRUNE_PARAM_SPECS["default"]
        )
        # 0 previews the centermost cell, any other value previews that label
        self._preview_cell = SpinBox(label="Preview cell", value=0, min=0, max=1_000_000)
        self._show_radius = PushButton(text="Show graph radius")
        self._radius_legend = Label(value=RADIUS_LEGEND, visible=False)

        self._num_jobs = SpinBox(label="Parallel jobs", value=1, min=1, max=16)
        self._num_workers_encode = SpinBox(label="Encoder workers", value=2, min=0, max=8)
        self._batch_size = SpinBox(label="Batch size", value=1, min=1, max=16)
        self._device = ComboBox(label="Device", choices=_available_devices())

        self._use_patches = CheckBox(value=False, text="Tile into patches")
        self._patch_size = SpinBox(
            label="Patch size", value=256, min=16, max=4096, visible=False
        )

        self._cache_mode = ComboBox(
            label="Feature cache", choices=CACHE_MODES, value="Session"
        )
        self._cache_dir = FileEdit(label="", mode="d", visible=False)
        self._cache_help = _wrapping_label(CACHE_HELP)
        self._use_output = CheckBox(value=False, text="Output directory")
        self._output_dir = FileEdit(label="", mode="d", visible=False)
        self._export_format = ComboBox(
            label="Format", choices=["flat", "ctc"], value="flat", visible=False
        )
        self._export_suffix = LineEdit(
            label="Suffix", value=DEFAULT_SUFFIX["flat"], visible=False
        )

        self._run = PushButton(text="Start tracking")
        self._cancel_button = PushButton(text="Cancel", enabled=False)
        self._progress = ProgressBar(value=0, min=0, max=100, visible=False)
        self._status = _wrapping_label()

        top_widgets = [
            Label(value="<b>BacLCT</b>"),
            self._image_layer,
            self._channel,
            self._labels_layer,
            self._model_source,
            self._pretrained,
            self._model_path,
            self._model_help,
            self._resolved_model,
        ]
        # held out of the flat list: _insert_additional_parameters folds these into the
        # collapsible section after construction
        self._tracking_group = Container(
            widgets=[
                _wrapping_label(TRACKING_HELP),
                self._thr_corr,
                self._thr_div,
                self._seg_correction,
                self._radius,
                self._prune_method,
                self._prune_help,
                self._prune_param,
                self._preview_cell,
                self._show_radius,
                self._radius_legend,
            ]
        )
        self._runtime_group = Container(
            widgets=[
                _wrapping_label(RUNTIME_HELP),
                self._num_jobs,
                self._num_workers_encode,
                self._batch_size,
                self._device,
            ]
        )
        io_widgets = [
            self._use_patches,
            self._patch_size,
            self._cache_mode,
            self._cache_dir,
            self._cache_help,
            self._use_output,
            self._output_dir,
            self._export_format,
            self._export_suffix,
        ]
        action_widgets = [
            self._run,
            self._cancel_button,
            self._progress,
            self._status,
        ]
        super().__init__(
            widgets=[*top_widgets, *io_widgets, *action_widgets], scrollable=True
        )
        # the inner content widget, for our own layout edits. napari is handed the scroll
        # area instead, via the `native` override below
        self._content = self._widget._mgui_get_native_widget()
        self._content._magic_widget = self
        self._insert_additional_parameters(after=len(top_widgets))
        # the scroll area stretches the content to the viewport height. without a trailing
        # stretch the rows spread out to fill it, worst when the section is collapsed
        self._content.layout().addStretch(1)

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

    def _insert_additional_parameters(self, after: int) -> None:
        """Fold the advanced tracking and runtime parameters into one collapsed section.

        They appear as plain titled groups inside it, not nested collapsibles. Tiling,
        cache, and output stay flat above the run button. `after` is the layout index the
        section slots into, right below the model block.
        """
        # magicgui pads containers by 11px with 6px between rows, which reads as dead
        # space once a group is a tight list under a header
        for group in (self._tracking_group, self._runtime_group):
            group.margins = (0, 0, 0, 0)
            group.native.layout().setSpacing(2)

        self._tracking_header = Label(value="<b>Tracking parameters</b>")
        self._runtime_header = Label(value="<b>Runtime parameters</b>")

        section = QCollapsible("Additional parameters")
        section.setCollapsedIcon("▸")
        section.setExpandedIcon("▾")
        section.content().layout().setContentsMargins(0, 0, 0, 0)
        section.addWidget(self._tracking_header.native)
        section.addWidget(self._tracking_group.native)
        section.addWidget(self._runtime_header.native)
        section.addWidget(self._runtime_group.native)
        section.collapse(animate=False)
        self._additional_section = section

        self._content.layout().insertWidget(after, section)

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
        # both shipped models consume image features, so images are never optional
        if self._image_layer.value is None:
            return "BacLCT requires an image layer."
        if self._model_source.value == "From path" and not self._model_path.value:
            return "Select a model directory."
        # an untouched FileEdit reads as Path('.'), which would cache into the working
        # directory of whatever launched napari
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

        self._prune_help.value = PRUNE_HELP[self._prune_method.value]
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
            # a validation hint takes precedence, but must not wipe the outcome of the
            # last run: _refresh() runs again as the worker finishes
            self._status.value = reason or self._outcome or ""

    def _sync_prune_param(self) -> None:
        """Give the pruning parameter the range and label its method actually uses."""
        method = self._prune_method.value
        spec = PRUNE_PARAM_SPECS.get(method, PRUNE_PARAM_SPECS["default"])
        if self._prune_param.label == spec["label"]:
            return

        self._prune_param.label = spec["label"]
        # widen before narrowing, so the pending value can never sit outside both ranges
        self._prune_param.min = min(self._prune_param.min, spec["min"])
        self._prune_param.max = max(self._prune_param.max, spec["max"])
        self._prune_param.value = PRUNE_PARAM_DEFAULTS.get(method, DEFAULT_PRUNE_PARAM)
        self._prune_param.min, self._prune_param.max = spec["min"], spec["max"]
        self._prune_param.step = spec["step"]

    # ------------------------------------------------------------------ inputs

    def _image_data(self) -> np.ndarray:
        """Image stack as `(T, ...)`, with the channel axis sliced off if present."""
        image = np.asarray(self._image_layer.value.data)
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
        masks = np.asarray(self._labels_layer.value.data)
        images = self._image_data()

        tracker_overrides: dict = {
            "thr_corr": self._thr_corr.value,
            "thr_div": self._thr_div.value,
        }
        if not self._seg_correction.value:
            tracker_overrides["segmentation_correction"] = None

        output_name = self._output_name()
        return TrackParams(
            images=images,
            masks=masks,
            model=self._model_arg(),
            sequence_id=output_name,
            export_suffix=self._export_suffix.value or None,
            # None lets the model decide, which is what every registered model wants too
            classify_states=None,
            graph_search_radius=self._search_radius(),
            prune_edges_by=self._prune_edges_by(),
            tracker_overrides=tracker_overrides,
            cache_dir=self._cache_dir_param(),
            output_dir=(Path(self._output_dir.value) if self._use_output.value else None),
            export_format=self._export_format.value,
            device=self._device.value,
            num_jobs=self._num_jobs.value,
            num_workers_encode=self._num_workers_encode.value,
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
        masks = np.asarray(self._labels_layer.value.data)
        if masks.ndim != 3:
            show_warning("The radius preview supports 2D + time segmentations only.")
            return

        radius = self._search_radius()
        if radius is None or radius is INVALID_RADIUS:
            show_warning("Enter a search radius, in pixels or as a multiple like '2.5x'.")
            return

        t = int(self._viewer.dims.current_step[0])
        # 'Default' defers to the model, so resolve its own pruning to preview a region.
        # 'Off' has no region, leaving the search circle as the only shape
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

        # remove every previous preview, not just one by name, so nothing can accumulate
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
        nbytes = params.images.nbytes + params.masks.nbytes
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
        self._status.value = "Starting..."
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
        self._status.value = "Cancelling..."
        self._cancel_button.enabled = False

    def _on_progress(self, event) -> None:
        if event is None:  # keepalive yield
            return
        self._progress.value = int(event.fraction * 100)
        self._status.value = event.label

    def _on_finished(self, result: TrackResult | None) -> None:
        # a cancelled run returns None rather than erroring, so no layers are added
        if result is None:
            self._outcome = "Cancelled."
            self._status.value = self._outcome
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
        # the tracked-mask layer overlaps the input segmentation, so hide the latter
        if labels is not None:
            labels.visible = False
        self._progress.value = 100
        n_tracks = result.tracks["label"].n_unique()
        self._outcome = f"Done: {n_tracks} tracks."
        self._status.value = self._outcome

    def _clear_results(self) -> None:
        """Drop the layers of the previous run, so tracking again replaces them."""
        for layer in self._result_layers:
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        self._result_layers = []

    def _on_error(self, err: Exception) -> None:
        self._outcome = f"Failed: {err}"
        self._status.value = self._outcome
        show_warning(f"BacLCT tracking failed: {err}")

    def _on_done(self) -> None:
        self._worker = None
        self._cancel = None
        self._cancel_button.enabled = False
        self._progress.visible = False
        self._refresh()
