Getting started
===============

Installation
------------

It is strongly recommended to install on a machine with a CUDA-capable GPU. System
requirements depend on the size of the image data and the number of objects in it. The
smaller 2D sequences used in the paper (190 frames, ~500x500 px, ~10K objects) stayed
below 8 GB of GPU and system RAM, while the larger ones (800 frames, ~1000x1000 px, >100K
objects) required 16 GB of GPU and 32 GB of system RAM.

.. tab-set::

   .. tab-item:: pip

      Clone the repository and install it into an environment (e.g., using `Conda
      <https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html>`__).
      If the environment should use a GPU, install `PyTorch
      <https://pytorch.org/get-started/locally/>`__ first.

      .. code-block:: bash

         git clone https://github.com/bmcv/baclct
         cd baclct
         pip install -e .                # inference
         pip install -e ".[napari]"      # + the napari plugin

   .. tab-item:: uv

      `uv <https://docs.astral.sh/uv/>`__ sets up and runs everything in one command.

      .. code-block:: bash

         uv run baclct-track --help                                      # check the install
         uv run --extra napari napari                                    # napari plugin
         uv run --extra train baclct-train dataset=spores task=tracking  # train (needs the dataset)

   .. tab-item:: pixi

      `Pixi <https://pixi.sh/>`__ sets up and runs everything in one command.

      .. code-block:: bash

         pixi run track --help                         # check the install
         pixi run -e napari napari                     # launch the napari plugin
         pixi run train dataset=spores task=tracking   # train a model (needs the dataset)

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

``tracked_masks`` are the input masks relabelled along their trajectories, and ``tracks``
has one row per cell and frame. The same run from the command line:

.. code-block:: bash

   baclct-track --model baclct_track --data-dir data/ --output-dir outputs/

Continue with :doc:`guide/inference`, :doc:`guide/napari`, or :doc:`guide/training`.
