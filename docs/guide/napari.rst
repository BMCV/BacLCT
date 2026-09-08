napari plugin
=============

Install the plugin with the ``napari`` extra (see :doc:`../getting-started`), then open
**Plugins → BacLCT**, pick an image layer, a labels layer, and a model, then press **Start
tracking**. Results come back as a relabelled masks layer and a tracks layer.

The widget exposes:

- **Model** -- one of the :ref:`pre-trained models <available-models>`, or a path to an
  experiment directory.
- **Tracking** -- correspondence and division threshold, search radius, edge pruning.
- **Runtime** -- device, parallel jobs, encoder workers, batch size, and patch size for
  large images.
- **Feature cache** and **export directory**, both optional. Caching a session's features
  makes re-runs with changed thresholds fast.
