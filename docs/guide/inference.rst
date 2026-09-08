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

Both ``images`` and ``masks`` may be ``numpy`` or ``dask`` arrays. ``tracked_masks`` are
the input masks relabelled along their trajectories, and ``tracks`` has one row per cell
and frame (``label``, ``t``, the center coordinate, ``parent``, and the single-cell
features). If the model classifies life cycle states, ``tracks`` also has a state column.
Pass ``output_dir`` to additionally export in `CTC format
<https://celltrackingchallenge.net/datasets/>`__ or as flat CSV/TIF.

Command line
------------

The ``baclct-track`` CLI mirrors this API and additionally tracks whole directories of
sequences in one call. See ``--help``.

.. code-block:: bash

   baclct-track --model baclct_track --data-dir data/ --output-dir outputs/

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

All parameters are documented on :py:meth:`baclct.api.BacLCT.track`. The ones usually
worth setting:

- ``graph_search_radius`` -- largest displacement between two consecutive frames, in
  pixels or relative to cell size (``'2.5x'``, the default).
- ``prune_edges_by`` -- how candidate edges are pruned. Leave at the trained value unless
  cells are not rod-shaped.
- ``patch_size`` -- tile large frames whose whole-frame graph exceeds memory.
- ``classify_states``, ``device``, ``cache_dir``, ``output_dir``, ``export_format``.

Throughput and tracker thresholds are not arguments but config keys, passed as
``BacLCT(config_overrides=...)``: ``batch_size``, ``num_jobs_features``,
``num_workers_encode``, and ``tracker.thr_corr``, ``tracker.thr_div``,
``tracker.segmentation_correction``. Their defaults and the multiprocessing caveats are
documented in ``src/baclct/config/inference.yaml``.
