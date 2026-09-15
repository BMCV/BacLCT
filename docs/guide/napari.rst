napari plugin
=============

Install the plugin with the ``napari`` extra (see :doc:`../getting-started`), then start
``napari``, load the images and the masks, and open **Plugins → BacLCT**. Pick an image
layer, a labels layer, and a model, then run **Start tracking**. Results come back as a
relabelled masks layer and a tracks layer.

``baclct-napari`` does the same from the command line: it loads a sequence (lazy by
default), docks the widget, and selects the default parameters.

.. code-block:: bash

   baclct-napari images.tif masks.tif

Without paths it opens an empty viewer with the plugin. Parameters can be set on the
command line as well -- see :doc:`cli` or ``baclct-napari --help`` for the available
flags:

.. code-block:: bash

   baclct-napari images.tif masks.tif --model baclct_toiam_pc --num-workers 4 --thr-div 1.0

If not provided, the plugin takes the search radius and the edge pruning from the
pre-trained model. The thresholds and segmentation correction keep the widget's own
values.

Exports are named after what the two layer names agree on, plus the *Suffix* field,
following the same rule as the command line (see :ref:`naming-the-results`). The result
layers are named after the segmentation layer they were tracked from.

The widget exposes:

- **Image layer** and **Segmentation**, plus the **Channel** to track when the image
  carries a channel axis the segmentation does not have.
- **Model** -- one of the :ref:`pre-trained models <available-models>`, or a path to an
  experiment directory.
- **Additional parameters**:

  - *Tracking parameters* -- correspondence and division threshold, segmentation
    correction, search radius, and edge pruning. **Show graph radius** outlines the
    search region of a single cell in the current frame.
  - *Runtime parameters* -- workers, batch size, and device.

- **Tile into patches**, **Feature cache**, and **Output directory**, all optional.
  Caching the features makes re-runs with changed thresholds fast and reduces memory
  consumption.
