# BacLCT: Bacteria Life Cycle Tracking

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

It is strongly recommended to install on a machine with a CUDA-capable GPU. System
requirements depend on the size of the image data and the number of objects in it. The
smaller 2D sequences used in the paper (190 frames, ~500x500 px, ~10K objects) stayed
below 8 GB of GPU and system RAM, while the larger ones (800 frames, ~1000x1000 px, >100K
objects) required 16 GB of GPU and 32 GB of system RAM.

### Local installation

Clone the repository, then either install it into an environment (e.g., using
[Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)).
If the environment should use a GPU, install
[PyTorch](https://pytorch.org/get-started/locally/) first.

```bash
git clone https://github.com/bmcv/baclct
cd baclct
pip install -e .                # inference
pip install -e ".[napari]"      # + the napari plugin
```

Or let [uv](https://docs.astral.sh/uv/) or [Pixi](https://pixi.sh/) set up and run
everything in one command:

```bash
pixi run track --help                         # check the install
pixi run -e napari napari                     # launch the napari plugin
pixi run train dataset=spores task=tracking   # train a model (needs the dataset)

# alternatively
uv run baclct-track --help
uv run --extra napari napari
uv run --extra train baclct-train dataset=spores task=tracking
```

The Pixi environments are configured for Linux only and the full development environment
is pinned in `pixi.lock`. Setup using uv also works on macOS and Windows.

## Usage

### Python

```python
import tifffile
from baclct import BacLCT

images = tifffile.imread("images.tif")  # (T, H, W)
masks = tifffile.imread("masks.tif")    # instance segmentation

pipeline = BacLCT()
tracked_masks, tracks = pipeline.track(images, masks, model="baclct_track")
```

Both `images` and `masks` may be `numpy` or `dask` arrays. `tracked_masks` are the input
masks relabelled along their trajectories, and `tracks` has one row per cell and frame
(`label`, `t`, the center coordinate, `parent`, and the single-cell features). If the
model classifies life cycle states, `tracks` also has a state column. Pass `output_dir` to
additionally export in [CTC format](https://celltrackingchallenge.net/datasets/) or as
flat CSV/TIF.

The `baclct-track` CLI mirrors this API and additionally tracks whole directories of
sequences in one call. See `--help`.

```bash
baclct-track --model baclct_track --data-dir data/ --output-dir outputs/
```

### napari plugin

Open **Plugins → BacLCT**, pick an image layer, a labels layer, and a model, then press
**Start tracking**. The tracking parameters are exposed in the widget, together with an
optional feature cache and export directory, and results come back as a relabelled masks
layer and a tracks layer.

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
experiment directory containing the config (see Configuration) it was trained with and a
checkpoint.

## Advanced Usage

### Configuration

BacLCT is configured with [Hydra](https://hydra.cc), with all config groups under
`src/baclct/config/`. During training the config determines the model architecture, loss,
and the rest of the run (see `default.yaml`). During inference the config of the trained
model is loaded and only runtime parameters can be overridden, such as the batch size or
the number of parallel jobs; their defaults come from `inference.yaml`. Overrides go to
`BacLCT(config_overrides=...)` as a dict, a dotlist of 'key=value' strings, a
`DictConfig`, or a path to a YAML file.

### Training

```bash
pixi run train dataset=spores task=tracking_with_states fold=0
pixi run train dataset=toiam  task=tracking fold=0
```

A run requires a configured `dataset` and `task` (see the directories in
`src/baclct/config/`). The `dataset` defines the data and its graph parameters, and the
`task` selects whether life cycle states and divisions are predicted. `fold` selects the
cross-validation split and defaults to 0.

Datasets are read from `paths.data_dir` in [CTC
format](https://celltrackingchallenge.net/datasets/). Next to the sequences, a
`splits.yaml` maps each fold to train, val, and test sequence IDs, and an optional
`states.txt` holds per-cell life cycle states. The splits used in the paper are in
`examples/splits/`.

Caching is mandatory for training: node features, DINO embeddings, and candidate edges are
always written under `paths.feature_dir`, keyed by the parameters that produced them, so a
changed radius, stride, pruning, or encoder invalidates only what it affects. These caches
range from a few hundred MB to around 10 GB per sequence.

#### Extending to other datasets

To train on your own data, copy an existing dataset config such as
`src/baclct/config/dataset/toiam.yaml` to `mydata.yaml`, set `dataset_name: mydata`, and
adjust the graph parameters to your images: `graph_search_radius` (the largest
displacement between two consecutive frames, in pixels or relative to cell size such as
`2x`), the batch sizes, `use_patches` with `patch_size` and `patch_overlap` for large
fields of view, `num_node_classes` if your cells carry states, and `prune_edges_by` for
rod-shaped cells. Point `paths.data_dir` at the data (default `./data/{dataset_name}`) and
run `pixi run train dataset=mydata task=tracking fold=0`. The model can also be trained
using multiple datasets (see `combined_spores_toiam.yaml`).

#### Paper

The paper trains the spores model over all five folds, which Hydra runs as a sweep with
`fold=0,1,2,3,4 --multirun`. The toiam model is trained on a single fold (`fold=0`). The
ablations are provided as named `experiment` configs, run the same way, e.g. `pixi run
train experiment=no_dino fold=0`.

## License

MIT, see [LICENSE](LICENSE).

## Citation

> Kunzmann, M., Elizondo-Cantú, M. C., Bischofs, I. B., Rohr, K. Bacteria tracking and
> life cycle state classification using graph neural networks and pretrained vision
> transformers. *Medical Image Analysis*, 104275 (2026).
> [doi:10.1016/j.media.2026.104275](https://doi.org/10.1016/j.media.2026.104275)

```bibtex
@article{KUNZMANN2026104275,
  title = {Bacteria tracking and life cycle state classification using graph neural networks and pretrained vision transformers},
  journal = {Medical Image Analysis},
  pages = {104275},
  year = {2026},
  issn = {1361-8415},
  doi = {10.1016/j.media.2026.104275},
  url = {https://www.sciencedirect.com/science/article/pii/S1361841526003440},
  author = {Moritz Kunzmann and M. Carolina Elizondo-Cant\'u and Ilka B. Bischofs and Karl Rohr},
  keywords = {Live-cell microscopy, Bacteria tracking, Division detection, State classification, Graph neural network, Foundation model},
}
```

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
