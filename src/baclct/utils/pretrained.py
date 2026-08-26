"""Resolution and download of pre-trained BacLCT models.

Three models ship with BacLCT, all of which utilize DINO-pretrained ViT features:

    - `baclct_track`: tracking, bright field and phase contrast. The default.
    - `baclct_spore_classification_bf`: tracking and life cycle state classification,
      bright field.
    - `baclct_toiam_pc`: tracking, phase contrast.

A model is an experiment directory (`checkpoints/`, `.hydra/`, logs), the same layout a
training run produces. Names are resolved to such a directory by `resolve_model_dir`,
which falls back to downloading the model from the GitHub release assets into a user cache
directory. Passing a path to your own experiment directory bypasses all of this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from baclct.utils.logger import get_pylogger
from baclct.utils.progress import ProgressCallback, report

logger = get_pylogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A pre-trained model available by name."""

    name: str
    classifies_states: bool
    description: str
    # release asset to download from, pinned to the tag the asset was uploaded under.
    # an empty `url` means the model must be provided locally (see `resolve_model_dir`).
    url: str = ""
    sha256: str = ""

    @property
    def downloadable(self) -> bool:
        """Whether this model can be fetched from a release asset."""
        return bool(self.url and self.sha256)


# release the model assets are attached to. bump together with the sha256 values
_RELEASE_BASE = "https://github.com/bmcv/baclct/releases/download/v0.4.0"

MODEL_SPECS: dict[str, ModelSpec] = {
    "baclct_track": ModelSpec(
        name="baclct_track",
        classifies_states=False,
        description=(
            "Tracking and division detection in bright field and phase contrast "
            "images. Trained on the spore and TOIAM datasets."
        ),
        url=f"{_RELEASE_BASE}/baclct_track.zip",
        sha256="1d885d77b24c141bea05d93e74c24d0d8a626c7ad9e425405e02197405d6c2f4",
    ),
    "baclct_spore_classification_bf": ModelSpec(
        name="baclct_spore_classification_bf",
        classifies_states=True,
        description=(
            "Tracking, division detection, and life cycle state classification of "
            "germinating and outgrowing B. subtilis spores in bright field images. "
            "Trained on the spore dataset."
        ),
        url=f"{_RELEASE_BASE}/baclct_spore_classification_bf.zip",
        sha256="912a850637b9911bc50bfd3ab9567177a7f959ce7dfa0cb8a3b507ce16514584",
    ),
    "baclct_toiam_pc": ModelSpec(
        name="baclct_toiam_pc",
        classifies_states=False,
        description=(
            "Tracking and division detection in phase contrast images. Trained on "
            "the TOIAM dataset."
        ),
        url=f"{_RELEASE_BASE}/baclct_toiam_pc.zip",
        sha256="a27d74b4b8bbb4f40b06ae02ac0b8e90cf2f69cbbde98bf0ac5f5402c199af59",
    ),
}

# used by `BacLCT.track()` when neither the call nor the constructor names a model
DEFAULT_MODEL = "baclct_track"


def default_cache_dir() -> Path:
    """Directory downloaded models are cached in.

    Honours `$BACLCT_HOME`, else the platform user cache (`~/.cache/baclct` on Linux).
    """
    root = os.environ.get("BACLCT_HOME") or platformdirs.user_cache_dir("baclct")
    return Path(root) / "models"


def _repo_models_dir() -> Path:
    """`trained_models/` next to the source checkout (only exists for dev installs)."""
    return Path(__file__).parent.parent.parent.parent / "trained_models"


def _shipped_models_dir() -> Path:
    """`shipped_models/` next to the source checkout (only exists for dev installs)."""
    return Path(__file__).parent.parent.parent.parent / "shipped_models"


def fetch_model(name: str, progress: ProgressCallback | None = None) -> Path:
    """Download a registered model into the cache directory.

    Args:
        name: Registered model name.
        progress: Optional sink for download progress.

    Returns:
        The downloaded experiment directory.
    """
    import pooch  # imported lazily; only needed on a cache miss

    spec = MODEL_SPECS[name]
    if not spec.downloadable:
        raise FileNotFoundError(
            f"Model '{spec.name}' is not available for download yet. Download it "
            "manually from the BacLCT release assets on GitHub and either place it in "
            f"'{default_cache_dir() / spec.name}', point $BACLCT_MODEL_DIR at its parent "
            "directory, or pass the path to the experiment directory directly."
        )

    cache = default_cache_dir()
    report(progress, "model", 0, 1, f"Downloading model '{spec.name}'")
    logger.info(f"Downloading model '{spec.name}' to {cache}.")
    pooch.retrieve(
        url=spec.url,
        known_hash=f"sha256:{spec.sha256}",
        fname=f"{spec.name}.zip",
        path=cache,
        processor=pooch.Unzip(),
    )
    report(progress, "model", 1, 1, f"Downloaded model '{spec.name}'")

    model_dir = cache / f"{spec.name}.zip.unzip" / spec.name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Downloaded model '{spec.name}' but found no experiment directory at "
            f"{model_dir}. The release asset layout may have changed."
        )
    return model_dir


def resolve_model_dir(
    model: str | Path,
    download: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    """Resolve a model name or path to an experiment directory.

    Paths are returned as-is. Registered names are looked up in order:

        1. `$BACLCT_MODEL_DIR/{name}`
        2. `shipped_models/{name}` next to the source checkout (dev installs)
        3. `trained_models/{name}` next to the source checkout (dev installs)
        4. the user cache directory
        5. downloaded from the release assets (unless `download` is `False`)

    Args:
        model: Path to an experiment directory or YAML config, or a registered model name.
        download: Whether to download the model when it is not available locally.
        progress: Optional sink for download progress.

    Returns:
        Path to the resolved experiment directory.
    """
    path = Path(model)
    if path.exists():
        return path

    name = str(model)
    if name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model}'. Valid names: {list(MODEL_SPECS)}. "
            "Or pass a path to an existing experiment directory."
        )

    candidates = []
    if env_dir := os.environ.get("BACLCT_MODEL_DIR"):
        candidates.append(Path(env_dir) / name)
    candidates.append(_shipped_models_dir() / name)
    candidates.append(_repo_models_dir() / name)
    candidates.append(default_cache_dir() / f"{name}.zip.unzip" / name)
    candidates.append(default_cache_dir() / name)

    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Resolved model '{name}' to {candidate}.")
            return candidate

    if not download:
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Model '{name}' not found locally and downloading is disabled. Searched:\n"
            f"  {searched}"
        )

    return fetch_model(name, progress=progress)
