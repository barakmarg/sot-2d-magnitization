"""Data generation component: simulator parameters, the stray-field forward model, random
synthetic samples, and the dataset / training-file / test-set builders."""
from sot.data.params import sot_default_params
from sot.data.physics import sot_forward
from sot.data.simulator import sot_make_sample, sot_preview, sot_run_checks
from sot.data.dataset import (IMAGE_FIELDS, sot_generate_dataset, save_dataset_mat, grid_params,
                              make_training_set, to_network_input, load_training_data,
                              make_data_file, get_test_set, make_test_set)
