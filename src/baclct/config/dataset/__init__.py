"""Dataset-specific global config overrides.

Mandatory config group that defines global overrides suited for specific datasets.
Must provide a `dataset_name`. If multiple datasets should be combined during training,
this can be done by defining `included_datasets` as a list of dataset names. However, in
this case, only one set of graph parameters can be used.
"""
