# BacLCT: Bacteria Life Cycle Tracking

[![PyPI](https://img.shields.io/pypi/v/baclct)](https://pypi.org/project/baclct/)
[![Docs](https://readthedocs.org/projects/baclct/badge/?version=latest)](https://baclct.readthedocs.io/en/latest/)

Code for the paper [*Bacteria Tracking and Life Cycle State Classification using Graph
Neural Networks and Pretrained Vision
Transformers*](https://doi.org/10.1016/j.media.2026.104275) (Medical Image Analysis,
2026).

BacLCT is a unified GNN-based method for simultaneous tracking, division detection, and
life cycle state classification of bacteria in time-lapse microscopy. Segmented cells are
represented as nodes of a graph and their interactions over time as multi-frame edges. A
message-passing GNN classifies the graph edges as correspondence, division, or no
correspondence, and the graph nodes as life cycle states. From these predictions,
trajectories are reconstructed. The division and multi-frame predictions are used for
segmentation error correction, such as for missed detections, early divisions, and
incorrect merges. The node features combine learned features from a DINO-pretrained Vision
Transformer with handcrafted single-object features, so no task-specific encoder has to be
trained.

BacLCT includes pre-trained models for tracking bacteria in bright field and phase
contrast images, and for simultaneous tracking and life cycle state classification of *B.
subtilis* spore germination and outgrowth in bright field images. It is also available as
a napari plugin.

## Installation

It is recommended to install on a machine with a GPU. System requirements depend on the
size of the image data and the number of objects in it. The smaller 2D sequences used in
the paper (190 frames, ~500x500 px, ~10K objects) stayed below 8 GB of GPU and system
RAM, while the larger ones (800 frames, ~1000x1000 px, >100K objects) required 16 GB of
GPU and 32 GB of system RAM. Inference also works without a GPU, but will be much
slower.

For inference and the napari plugin, install from PyPI into an environment (e.g., using
[Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)).
If the environment should use a GPU, install
[PyTorch](https://pytorch.org/get-started/locally/) first.

```bash
pip install baclct              # inference
pip install "baclct[napari]"    # + the napari plugin
```

For training, it is recommended to clone the repository and install locally.

```bash
git clone https://github.com/bmcv/baclct
cd baclct
pip install -e ".[train]"       # training
```

Or let [uv](https://docs.astral.sh/uv/) or [Pixi](https://pixi.sh/) set up and run
everything in one command:

```bash
pixi run baclct-track --help   # check the install
uv run baclct-track --help
```

The Pixi environments are configured for Linux only and the full development environment
is pinned in `pixi.lock`. Setup using uv also works on macOS and Windows. The
[documentation](https://baclct.readthedocs.io) lists the commands for the napari plugin
and for training.

## Usage

```python
import tifffile
from baclct import BacLCT

images = tifffile.imread("images.tif")  # (T, H, W)
masks = tifffile.imread("masks.tif")    # instance segmentation

pipeline = BacLCT()
tracked_masks, tracks = pipeline.track(images, masks, model="baclct_track")
```

`masks` must be an instance segmentation, one label per object; relabel a binary mask
with `skimage.measure.label` first. Both `images` and `masks` may be `numpy` or `dask`
arrays. `tracked_masks` are the input
masks relabelled along their trajectories, and `tracks` has one row per cell and frame
(`label`, `t`, the center coordinate, `parent`, and the single-cell features). If the
model classifies life cycle states, `tracks` also has a state column. Pass `output_dir` to
additionally export in [CTC format](https://celltrackingchallenge.net/datasets/) or as
flat CSV/TIF.

The `baclct-track` CLI mirrors this API. It takes the two paths directly, or a dataset
directory in one of three layouts, and then tracks every sequence in it. See `--help`.

```bash
baclct-track images.tif masks.tif -o outputs/   # one sequence
baclct-track --data-dir data/ -o outputs/       # every sequence in a dataset
```

The same runs interactively in napari. `baclct-napari` opens a sequence with the plugin
docked and the layers preselected, reading the frames on demand so a long movie opens at
once:

```bash
baclct-napari images.tif masks.tif
```

It takes the same tracking flags as `baclct-track`, and the plugin is also reachable the
usual way, under **Plugins → BacLCT**.

### Pre-trained Models

Three pre-trained models for bacteria tracking are available by name, optionally with life
cycle state classification. They were trained on two datasets, each model on the subset
listed in the table below: bright-field sequences of germinating and outgrowing *B.
subtilis* spores with annotated trajectories and life cycle states
(<https://doi.org/10.5281/zenodo.21805068>) and phase-contrast sequences of growing *C.
glutamicum* microcolonies with annotated trajectories (TOIAM, [Seiffarth et al.
2025](https://doi.org/10.5281/zenodo.7260136)).

| Model | Use case | Trained on |
|-------|----------|------------|
| `baclct_track` | Bacteria tracking and division detection. Bright-field and phase-contrast. Default. | Spores + TOIAM |
| `baclct_spore_classification_bf` | Bacteria tracking and division detection. Life cycle state classification for *B. subtilis* spore germination and outgrowth. Bright-field. Used in paper. | Spores |
| `baclct_toiam_pc` | Bacteria tracking and division detection. Phase-contrast. Used in paper. | TOIAM |

The models are downloaded automatically from the GitHub release on first use. A model is an
experiment directory containing the config it was trained with and a checkpoint.

## Training

```bash
baclct-train dataset=spores task=tracking_with_states fold=0
baclct-train dataset=toiam task=tracking fold=0
```

A run requires a configured `dataset` and `task` (see the directories in
`src/baclct/config/`). The `dataset` defines the data and its graph parameters, and the
`task` selects whether life cycle states and divisions are predicted. `fold` selects the
cross-validation split and defaults to 0.

Datasets are read from `paths.data_dir` in [CTC
format](https://celltrackingchallenge.net/datasets/). Next to the sequences, a
`splits.yaml` maps each fold to train, val, and test sequence IDs, and an optional
`states.txt` holds per-cell life cycle states. The splits used in the paper are in
`examples/splits/`. Caching is mandatory for training: node features, DINO embeddings, and
candidate edges are always written under `paths.feature_dir` and stay below 1 GB for a
typical sequence, growing to a few GB for long or very dense ones.

To train on your own data, copy the annotated
[`examples/configs/dataset/example.yaml`](examples/configs/dataset/example.yaml), adjust the
graph parameters to your images, and pass it with `--config-dir`:

```bash
baclct-train --config-dir examples/configs dataset=example task=tracking fold=0
```

BacLCT is configured with [Hydra](https://hydra.cc), so any key can be overridden on the
command line or swept with `--multirun`. See the
[documentation](https://baclct.readthedocs.io) for the config groups and for reproducing
the paper's folds and ablations.

## Citation

> Kunzmann, M., Elizondo-Cantú, M. C., Bischofs, I. B., Rohr, K. Bacteria tracking and
> life cycle state classification using graph neural networks and pretrained vision
> transformers. *Medical Image Analysis*, 104275 (2026).
> [doi:10.1016/j.media.2026.104275](https://doi.org/10.1016/j.media.2026.104275)

When using the models with DINO-pretrained ViT features, please also cite [Caron et al.,
ICCV 2021](https://arxiv.org/abs/2104.14294).

## Declaration of generative AI use

Parts of the codebase were developed with AI assistance (Claude Code). This was used
primarily for refactoring, organizing code and tests (e.g., converting existing notebooks
into integration tests), packaging and parts of the documentation (e.g., Sphinx),
debugging, as well as runtime and memory optimization (e.g., replacing existing code with
faster libraries). Some components, notably several tests and the napari plugin, started
as generated drafts that were subsequently corrected, partly reimplemented, or heavily
refactored manually. Core functionality was ported from the author's previous
implementation of this work. Where new functionality was generated, it was validated
against the previous implementation, existing benchmarks, or hand-written tests. All
generated code was reviewed and validated by the author.
