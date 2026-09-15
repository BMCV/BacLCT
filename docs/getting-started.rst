Getting started
===============

Installation
------------

It is recommended to install on a machine with a GPU. System requirements depend on the
size of the image data and the number of objects in it. The smaller 2D sequences used in
the paper (190 frames, ~500x500 px, ~10K objects) stayed below 8 GB of GPU and system
RAM, while the larger ones (800 frames, ~1000x1000 px, >100K objects) required 16 GB of
GPU and 32 GB of system RAM. Inference also works without a GPU, but will be much
slower.

.. tab-set::

   .. tab-item:: pip

      For inference and the napari plugin, install from PyPI into an environment (e.g.,
      using `Conda
      <https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html>`__).
      If the environment should use a GPU, install `PyTorch
      <https://pytorch.org/get-started/locally/>`__ first.

      .. code-block:: bash

         pip install baclct                  # inference
         pip install "baclct[napari]"        # + the napari plugin
         baclct-napari images.tif masks.tif  # open napari plugin

      For training, it is recommended to clone the repository and install locally.

      .. code-block:: bash

         git clone https://github.com/bmcv/baclct
         cd baclct
         pip install -e ".[train]"           # training

   .. tab-item:: uv

      `uv <https://docs.astral.sh/uv/>`__ sets up and runs everything in one command.

      .. code-block:: bash

         uv run baclct-track --help                                        # check the install
         uv run --extra napari baclct-napari images.tif masks.tif          # napari plugin
         uv run --extra train baclct-train dataset=spores task=tracking    # train (needs dataset)

   .. tab-item:: pixi

      `Pixi <https://pixi.sh/>`__ sets up and runs everything in one command.

      .. code-block:: bash

         pixi run baclct-track --help                         # check the install
         pixi run -e napari baclct-napari                     # napari plugin (see the guide)
         pixi run baclct-train dataset=spores task=tracking   # train a model (needs dataset)

The Pixi environments are configured for Linux only and the full development environment
is pinned in ``pixi.lock``. Setup using uv also works on macOS and Windows.

Tracking a sequence
-------------------

.. code-block:: python

   import tifffile
   from baclct import BacLCT

   images = tifffile.imread("images.tif")  # (T, H, W)
   masks = tifffile.imread("masks.tif")    # instance segmentation

   pipeline = BacLCT()
   tracked_masks, tracks = pipeline.track(images, masks, model="baclct_track")

``masks`` must be an instance segmentation, one label per object; relabel a binary mask
with ``skimage.measure.label`` first. ``tracked_masks`` are the input masks relabelled
along their trajectories, and ``tracks`` has one row per cell and frame. The same run from
the command line:

.. code-block:: bash

   baclct-track --model baclct_track -o tracking_results images.tif masks.tif

Continue with :doc:`guide/inference`, :doc:`guide/cli`, :doc:`guide/napari`, or
:doc:`guide/training`.
