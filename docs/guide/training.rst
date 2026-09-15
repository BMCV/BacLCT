Training
========

Training a Model
----------------

.. code-block:: bash

   baclct-train dataset=spores task=tracking_with_states fold=0
   baclct-train dataset=toiam  task=tracking fold=0

A run requires a configured ``dataset`` and ``task`` (see the directories in
``src/baclct/config/``). The ``dataset`` config group defines the data and its graph
parameters, and the ``task`` config group selects whether life cycle states and divisions
are predicted. ``fold`` selects the cross-validation split and defaults to 0.

Data layout
-----------

Datasets are read from ``paths.data_dir`` (``./data/{dataset_name}`` by default) in `CTC
format <https://celltrackingchallenge.net/datasets/>`__. Next to the sequences, a
``splits.yaml`` maps each fold to train, val, and test sequence IDs, and an optional
``states.txt`` holds per-cell life cycle states. The splits used in the paper are in
``examples/splits/``.

Caching
-------

Caching is mandatory for training: node features, DINO embeddings, and candidate edges are
always written under ``paths.feature_dir``. Cache directories are named by dataset,
segmentation, and the parameters that were used to create them (e.g., graph parameters,
DINO ViT backbone). In addition, metadata are stored to invalidate caches upon mismatch.
Caches stay below 1 GB for a typical sequence and grow to a few GB for long or very dense
ones (<3GB for a TOIAM sequence with 14M candidate edges).

.. _configuration:

Configuration
-------------

Training is configured with `Hydra <https://hydra.cc>`__, with all config groups under
``src/baclct/config/``. Config groups are loaded/selected in ``default.yaml``. Any key can
be overridden on the command line. Sweeps can be run using ``--multirun`` and providing
several values:

.. code-block:: bash

   baclct-train dataset=toiam task=tracking hidden_dim=64,128 model.lr=0.001,0.0001 --multirun

From Python, the same ``key=value`` overrides are passed to
``BacLCT(config_overrides=...)`` as a list of strings (like in command line):

.. code-block:: python

   BacLCT(config_overrides=["dataset=toiam", "task=tracking", "model.lr=1e-4"])

Dicts, ``DictConfig`` objects, and paths to YAML files are accepted too, but can't select
config groups, so they only apply on top of an already composed config, passed as
``BacLCT(config=...)``, e.g., from a previous run ``.hydra/config.yaml``.

Custom datasets
---------------

Dataset-specific parameters (e.g., learning rate, model size, graph parameters) are
specified in the ``dataset`` config group. The recommended way to train on custom data is
to clone the repository, copy the commented example config
(``./examples/configs/dataset/example.yaml``) into the packaged ``dataset`` group, and
adapt it there. It is then selected by name like any other dataset:

.. code-block:: bash

   cp examples/configs/dataset/example.yaml src/baclct/config/dataset/example.yaml
   baclct-train dataset=example task=tracking fold=0

See `example.yaml
<https://github.com/bmcv/baclct/blob/main/examples/configs/dataset/example.yaml>`__ for
the available/recommended config keys and details.

Training is also supported without cloning the repository (``pip install
"baclct[train]"``). Settings are then passed either as ``key=value`` overrides on the
command line, or as a config file in a separate directory: keep the adapted
``example.yaml`` in a ``dataset/`` subdirectory and point Hydra at its parent directory
with ``--config-dir``:

.. code-block:: bash

   baclct-train --config-dir ./configs dataset=example task=tracking fold=0

Reproducing the paper
---------------------

The paper trains the spores model over all five folds, which Hydra runs as a sweep with
``fold=0,1,2,3,4 --multirun``. The toiam model is trained on a single fold (``fold=0``).
The ablations are provided as named ``experiment`` configs, e.g., ``baclct-train
experiment=no_dino fold=0,1,2,3,4 --multirun``.
