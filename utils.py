import numpy as np
import warnings
import logging
import torch
import os

from scipy.spatial import KDTree


def compute_metrics(outputs, labels):
    outputs = (outputs > 0.5).float()
    labels = labels.float()

    intersection = (outputs * labels).sum()
    union = outputs.sum() + labels.sum()
    dice = 2.0 * intersection / (union + 1e-7)

    tp = (outputs * labels).sum().item()
    fp = (outputs * (1 - labels)).sum().item()
    fn = ((1 - outputs) * labels).sum().item()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return dice.item(), precision, recall


def compute_hd95_and_asd(outputs, labels):
    def get_surface_points(binary_mask):
        kernel = torch.ones(3, 3, 3, device=binary_mask.device)
        eroded_mask = torch.nn.functional.conv3d(
            binary_mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1
        )
        boundary_mask = binary_mask - (eroded_mask.squeeze(0).squeeze(0) > 0)
        surface_points = torch.nonzero(boundary_mask, as_tuple=False)
        return surface_points.cpu().numpy()

    outputs = (outputs > 0.5).float()
    labels = labels.float()

    output_points = get_surface_points(outputs)
    label_points = get_surface_points(labels)

    if len(output_points) == 0 or len(label_points) == 0:
        return float('inf'), float('inf')

    tree1 = KDTree(output_points)
    tree2 = KDTree(label_points)

    distances1, _ = tree2.query(output_points)
    distances2, _ = tree1.query(label_points)

    all_distances = np.concatenate([distances1, distances2])

    hd95 = np.percentile(all_distances, 95)
    asd = (np.mean(distances1) + np.mean(distances2)) / 2

    return hd95, asd


def setup_logging(log_dir):
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")

    log_file = os.path.join(log_dir, "train.log")
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def create_experiment_dir(base_dir=None, fold=None, run_id=None):
    # Default base directory
    if base_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_dir, 'experiment')

    # Define paths
    fold_dir = os.path.join(base_dir, f"fold_{fold}")

    run_dir = os.path.join(fold_dir, f"run_{run_id}")

    # Create directories
    os.makedirs(run_dir, exist_ok=True)

    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(checkpoints_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(run_dir)
    logger.info(f"Experiment directories created:\n"
                f"Base Directory: {base_dir}\n"
                f"Fold Directory: {fold_dir}\n"
                f"Run Directory: {run_dir}\n"
                f"Logs Directory: {run_dir}\n"
                f"Checkpoints Directory: {checkpoints_dir}")

    return {
        "run_dir": run_dir,
        "logs_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "logger": logger,
    }
