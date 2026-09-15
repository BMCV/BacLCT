Inference
=========

Python
------

.. code-block:: python

   import tifffile
   from baclct import BacLCT

   images = tifffile.imread("images.tif")  # (T, H, W)
   masks = tifffile.imread("masks.tif")    # instance segmentation

   pipeline = BacLCT()
   tracked_masks, tracks = pipeline.track(images, masks, model="baclct_track")

``masks`` must be an instance segmentation, one label per object; relabel a binary mask
with ``skimage.measure.label`` first. Both ``images`` and ``masks`` may be ``numpy`` or
``dask`` arrays. ``tracked_masks`` are
the input masks relabelled along their trajectories, and ``tracks`` has one row per cell
and frame (``label``, ``t``, the center coordinate, ``parent``, and the single-cell
features). If the model classifies life cycle states, ``tracks`` also has a state column.
Pass ``output_dir`` to additionally export in `CTC format
<https://celltrackingchallenge.net/datasets/>`__ or as flat CSV/TIF.

Command line
------------

The ``baclct-track`` CLI mirrors this API and additionally tracks whole directories of
sequences in one call. Sequences are lazy-loaded unless ``--no-lazy`` is given. See
:doc:`cli`.

.. code-block:: bash

   baclct-track images.tif masks.tif -o outputs/   # one sequence
   baclct-track --data-dir data/ -o outputs/       # every sequence in a dataset

The same run opens interactively in napari. ``baclct-napari`` docks the :doc:`plugin
<napari>` with the layers preselected and takes the same tracking flags:

.. code-block:: bash

   baclct-napari images.tif masks.tif

.. _available-models:

Available models
----------------

Three pre-trained models for bacteria tracking are available by name, optionally with life
cycle state classification. They were trained on two datasets, each model on the subset
listed in the table below: bright-field sequences of germinating and outgrowing *B.
subtilis* spores with annotated trajectories and life cycle states
(https://doi.org/10.5281/zenodo.21805068) and phase-contrast sequences of growing *C.
glutamicum* microcolonies with annotated trajectories (TOIAM, `Seiffarth et al. 2025
<https://doi.org/10.5281/zenodo.7260136>`__).

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Model
     - Use case
     - Trained on
   * - ``baclct_track``
     - Bacteria tracking and division detection. Bright-field and phase-contrast. Default.
     - Spores + TOIAM
   * - ``baclct_spore_classification_bf``
     - Bacteria tracking and division detection. Life cycle state classification for *B.
       subtilis* spore germination and outgrowth. Bright-field. Used in paper.
     - Spores
   * - ``baclct_toiam_pc``
     - Bacteria tracking and division detection. Phase-contrast. Used in paper.
     - TOIAM

The models are downloaded automatically from the GitHub release on first use. A model is
an experiment directory containing the config it was trained with and a checkpoint.

Inference parameters
--------------------

All parameters required for inference are shared between
:py:meth:`baclct.api.BacLCT.track` and the CLI. Architecture- and data-specific defaults
are read from the loaded pre-trained model. Additional parameters can be passed as config
overrides via ``BacLCT(config_overrides=...)`` (see :ref:`configuration`), but are
generally not required.

The main inference parameters are:

- ``graph_search_radius`` -- largest displacement between objects in two consecutive
  frames, in pixels or relative to cell size (default: ``'2.5x'``).
- ``prune_edges_by`` -- how candidate edges are pruned. The defaults are selected for
  rod-shaped bacteria. Pass ``'off'`` for round cells. The pruning methods are
  described on :py:class:`~baclct.features.graph.EdgeFinder` and can also be visualized in
  the napari plugin.
- ``thr_div`` -- lower to recover missed divisions, raise to mitigate FP division
  detection for new data, or set to 1.0 to disable division detection.
- ``thr_corr`` -- lower to recover missed tracks, raise to mitigate FP correspondences.
- ``segmentation_correction`` -- mainly developed for rod-shaped bacteria. Disable
  for round objects or very dense regions.
- ``patch_size`` -- tile large frames whose whole-frame graph exceeds memory.
- ``device``, ``output_dir``, ``export_format``.

Life cycle state classification is enabled depending on the pre-trained model. See
:ref:`available-models`.

Throughput
----------

To support low-RAM machines (and Windows), parts of BacLCT run single-threaded by default
and with a batch size of 1. Increasing batch size and the number of workers speeds up
inference at the cost of higher RAM consumption. For large data, BacLCT supports
inference within image tiles and lazy data loading using ``dask`` and ``polars``:

- ``batch_size`` (1) -- subgraphs per prediction batch.
- ``num_workers`` (0) -- worker processes for feature extraction. Used during extraction
  of handcrafted features and during image preparation / cropping for ViT features. Worth
  raising on long or very dense sequences (e.g., TOIAM).
- ``cache_dir`` -- where node, deep, and edge features are cached (below 1 GB for a
  typical sequence). ``'temp'`` uses a temporary directory that is removed afterwards,
  ``None`` (``memory`` on the CLI) caches nothing and keeps everything in RAM, and a path
  keeps the features for repeated runs, which makes re-tracking with changed thresholds
  cheap. The napari plugin offers the same three as *In memory*, *Session* (temporary
  directory that is removed once the viewer closes), and *Directory*.
