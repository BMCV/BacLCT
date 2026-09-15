BacLCT: Bacteria Life Cycle Tracking
====================================

BacLCT is a unified GNN-based method for simultaneous tracking, division detection, and
life cycle state classification of bacteria in time-lapse microscopy. Segmented cells are
represented as nodes of a graph and their interactions over time as multi-frame edges. A
message-passing GNN classifies the graph edges as correspondence, division, or no
correspondence, and the graph nodes as life cycle states. From these predictions,
trajectories are reconstructed. The division and multi-frame predictions are used for
segmentation error correction, such as for missed detections, early divisions, and
incorrect merges. The node features combine learned features from a DINO-pretrained Vision
Transformer with handcrafted single-object features, so no task-specific encoder has to be
trained.

BacLCT includes pre-trained models for tracking bacteria in bright field and phase
contrast images, and for simultaneous tracking and life cycle state classification of *B.
subtilis* spore germination and outgrowth in bright field images. It is also available as
a napari plugin.

- :doc:`getting-started` -- install and track a sequence.
- :doc:`guide/index` -- inference, the napari plugin, and training your own models.
- :doc:`api` -- generated from the docstrings.

Citation
--------

   Kunzmann, M., Elizondo-Cantú, M. C., Bischofs, I. B., Rohr, K. Bacteria tracking and
   life cycle state classification using graph neural networks and pretrained vision
   transformers. *Medical Image Analysis*, 104275 (2026).
   `doi:10.1016/j.media.2026.104275 <https://doi.org/10.1016/j.media.2026.104275>`__

When using the models with DINO-pretrained ViT features, please also cite `Caron et al.,
ICCV 2021 <https://arxiv.org/abs/2104.14294>`__.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   guide/index
   api
