"""Model component: the CNN, its loss, symmetry augmentation, training and prediction.
PyTorch is imported only when a network is built, loaded or run."""
from sot.model.network import build_cnn, save_net, load_net, best_file_of, load_best_network
from sot.model.loss import model_loss, dir_loss, phi_weight, loss_angle, angle_err
from sot.model.augment import apply_sym, augment, add_phi_channels
from sot.model.train import train_settings, train_cnn
from sot.model.predict import predict_dirs, predict_dirs_tta, show_examples
