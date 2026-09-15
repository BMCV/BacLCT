Command line
============

``baclct-track`` runs the pipeline of :doc:`inference` on one sequence or on a whole
dataset. Every flag mirrors an argument of :py:meth:`baclct.api.BacLCT.track`:

.. code-block:: bash

   baclct-track --help

``--help`` documents every flag, grouped into **Data**, **Graph**, **Tracker**,
**Throughput**, and **Export**. ``--model`` takes any name from :ref:`available-models` or
a path to a model directory. See :doc:`inference` for details on multi-processing and
caching.

``baclct-napari`` takes the same inputs and tracking flags, but opens them in the
:doc:`napari plugin <napari>`. It only accepts single sequences.

Input
-----

Either name the images and the masks directly, each a tif stack or a directory of
per-frame tifs:

.. code-block:: bash

   baclct-track images.tif masks.tif -o outputs/
   baclct-track images/ segmentation/ -o outputs/

Results are written next to the masks unless ``-o/--output-dir`` is given (see
:ref:`naming-the-results`).

Or point at a dataset directory to track every sequence in it:

.. code-block:: bash

   baclct-track --data-dir data/ -o outputs/
   baclct-track --data-dir data/ --sequences 01,02

Results then default to ``data/baclct_results``. A sequence whose results are already
there is skipped unless ``--overwrite`` is given, so an interrupted batch run continues
where it stopped.

.. _naming-the-results:

Naming the results
------------------

Every export is named ``{id}{suffix}``: a directory for ``ctc``, a ``.csv`` and a
``.tif`` for ``flat``.

The **id** is the sequence id under ``--data-dir``, or the id given by ``--sequences``.
With two explicit paths it is what the paths agree on: their shared filename prefix
(``01_images.tif`` beside ``01_masks.tif``), directory (``01/BF`` beside
``01/Segmentation``), or branch (``01`` beside ``01_GT/TRA``). If they share nothing, the
mask filename is used.

The **suffix** is ``--export-suffix``, which defaults to ``_RES`` for ``ctc`` (the CTC
convention) and ``_tracks`` for ``flat``:

.. code-block:: bash

   # outputs/colony3_run7.csv and outputs/colony3_run7.tif
   baclct-track images.tif masks.tif --sequences colony3 --export-suffix _run7 -o outputs/

The :doc:`napari plugin <napari>` derives the id the same way from the two layer names,
and takes the suffix from its *Suffix* field.

Dataset layouts
---------------

``--data-format`` specifies how ``--data-dir`` is arranged:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Layout
     - Shape on disk
   * - ``ctc`` (default)
     - A sequence directory beside its segmentation, ``01/`` and ``01_GT/TRA``. The
       `Cell Tracking Challenge <https://celltrackingchallenge.net/datasets/>`__ layout.
   * - ``dirs``
     - One directory per sequence, each with an image and a segmentation directory,
       ``01/BF`` and ``01/Segmentation``.
   * - ``flat``
     - Paired tifs, ``01_images.tif`` and ``01_masks.tif``.

``--segmentation-name`` and ``--img-name`` replace the names in the second column, which
default to ``GT`` (``ctc``), ``Segmentation`` and ``BF`` (``dirs``), and ``masks`` and
``images`` (``flat``). This selects between several segmentations of the same dataset:

.. code-block:: bash

   baclct-track --data-dir data/ --data-format dirs --segmentation-name cellpose

Sequence ids are discovered in ``--data-dir``, or selected with ``--sequences``
(comma-separated or the flag repeated). A sequence counts only when its segmentation
exists under the name in use, which leaves out directories that are not sequences. An id
that is not in the dataset stops the run before any tracking starts.

Dry run
-------

``--dry-run`` resolves the inputs of every selected sequence, opens them lazily, and
reports their shape without tracking:

.. code-block:: bash

   baclct-track --data-dir data/ --data-format dirs --segmentation-name cellpose --dry-run

.. code-block:: text

   Sequence pos01: images (121, 1040, 1392) uint16, masks (121, 1040, 1392) uint16
   Sequence pos02: images (121, 1040, 1392) uint16, masks (122, 1040, 1392) uint16 -- frame counts differ

Examples
--------

.. code-block:: bash

   # one pair of tif stacks, named results
   baclct-track images.tif masks.tif --sequences 01 -o outputs/

   # one pair of per-frame tif directories, from a CTC dataset
   baclct-track data/01 data/01_GT/TRA -o outputs/

   # one sequence of a CTC dataset, on the ground truth and on a predicted segmentation
   baclct-track --data-dir data/ --sequences 01 -o outputs/
   baclct-track --data-dir data/ --sequences 01 --segmentation-name OP -o outputs/

   # every sequence of a flat dataset, tracked on a second segmentation
   baclct-track --data-dir data/ --data-format flat --segmentation-name omnipose -o outputs/

   # every sequence of a per-sequence directory layout, on two workers
   baclct-track --data-dir data/ --data-format dirs --segmentation-name omnipose \
       --num-workers 2 -o outputs/
