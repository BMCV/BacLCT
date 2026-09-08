API reference
=============

:py:class:`~baclct.api.BacLCT` in ``baclct.api`` is the top-level entry point for training
and tracking. It reads the config, prepares the data and the model, and then either runs
training based on the loaded config (:py:meth:`~baclct.api.BacLCT.run_training`) or tracks
a single sequence (:py:meth:`~baclct.api.BacLCT.track`). :py:class:`~baclct.api.BacLCT` is
also used by the two command line tools and the napari plugin.

The pipeline is roughly split into the following parts: ``baclct.features`` extracts
single-cell node and relational edge features and embeds them into a graph.
:py:class:`~baclct.data.dataset.GraphDataset` holds the resulting trajectory graph of a
single image sequence and returns temporal windows around a frame as PyG ``Data`` objects.
Training uses the Lightning DataModule around it,
:py:class:`~baclct.data.dataset.TrackingDataset`, which combines datasets of several
sequences, handles feature extraction and caching, and applies the fold splits. The GNN
itself is defined in :py:class:`~baclct.models.model.MPModel`, and
:py:class:`~baclct.models.lightning_model.TrackingModel` contains the training and
prediction logic, losses, and metrics. Predictions are processed in ``baclct.tracking``,
which reconstructs trajectories and applies segmentation error correction.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   baclct.api
   baclct.features
   baclct.data
   baclct.models
   baclct.tracking
   baclct.io
   baclct.track
   baclct.train
   baclct.napari
   baclct.utils
   baclct.viz
   baclct.config
