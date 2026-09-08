Training
========

Running a training
------------------

.. code-block:: bash

   pixi run train dataset=spores task=tracking_with_states fold=0
   pixi run train dataset=toiam  task=tracking fold=0

A run requires a configured ``dataset`` and ``task`` (see the directories in
``src/baclct/config/``). The ``dataset`` defines the data and its graph parameters, and
the ``task`` selects whether life cycle states and divisions are predicted. ``fold``
selects the cross-validation split and defaults to 0.

Data layout
-----------

Datasets are read from ``paths.data_dir`` (``./data/<dataset_name>`` by default) in `CTC
format <https://celltrackingchallenge.net/datasets/>`__. Next to the sequences, a
``splits.yaml`` maps each fold to train, val, and test sequence IDs, and an optional
``states.txt`` holds per-cell life cycle states. The splits used in the paper are in
``examples/splits/``.

Caching
-------

Caching is mandatory for training: node features, DINO embeddings, and candidate edges are
always written under ``paths.feature_dir``, keyed by the parameters that produced them, so
a changed radius, stride, pruning, or encoder invalidates only what it affects. These
caches range from a few hundred MB to around 10 GB per sequence.

Configuration
-------------

Training is configured with `Hydra <https://hydra.cc>`__, with all config groups under
``src/baclct/config/``; ``default.yaml`` composes a run. Any key can be overridden on the
command line, and sweeping over several values runs them as a multirun:

.. code-block:: bash

   pixi run train dataset=toiam task=tracking hidden_dim=64,128 --multirun

From Python, overrides go to ``BacLCT(config_overrides=...)`` as a dict, a dotlist of
``key=value`` strings, a ``DictConfig``, or a path to a YAML file.

Your own dataset
----------------

Copy ``examples/configs/dataset/mydata.yaml``, adjust the graph parameters to your images,
and point ``--config-dir`` at it:

.. code-block:: bash

   pixi run train --config-dir examples/configs dataset=mydata task=tracking fold=0

The example config is commented and covers the keys that matter: ``graph_search_radius``
(the largest displacement between two consecutive frames, in pixels or relative to cell
size such as ``2x``), the batch sizes, ``use_patches`` with ``patch_size`` and
``patch_overlap`` for large fields of view, ``num_node_classes`` if your cells carry
states, and ``prune_edges_by`` for rod-shaped cells. Point ``paths.data_dir`` at the data
(default ``./data/<dataset_name>``). The model can also be trained using multiple datasets
(see ``combined_spores_toiam.yaml``).

Reproducing the paper
---------------------

The paper trains the spores model over all five folds, which Hydra runs as a sweep with
``fold=0,1,2,3,4 --multirun``. The toiam model is trained on a single fold (``fold=0``).
The ablations are provided as named ``experiment`` configs, run the same way, e.g. ``pixi
run train experiment=no_dino fold=0``.
