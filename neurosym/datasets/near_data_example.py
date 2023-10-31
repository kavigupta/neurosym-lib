from .load_data import numpy_dataset_from_github

data = numpy_dataset_from_github(
    "https://github.com/trishullab/near/tree/master/near_code/data/example",
    "train_ex_data.npy",
    "train_ex_labels.npy",
    "test_ex_data.npy",
    "test_ex_labels.npy",
)
