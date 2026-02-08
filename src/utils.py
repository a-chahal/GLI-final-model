"""
GLI-PLAPT Utilities — Reproducibility, logging, and helpers.
"""

import os
import json
import time
import random
import logging
import csv
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch

from src.config import SEED, OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR, EMBED_CACHE_DIR


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_dirs():
    """Create all output directories."""
    for d in [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR, EMBED_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def setup_logging(experiment_name: str) -> logging.Logger:
    """Configure logging to both file and console."""
    setup_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{experiment_name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging to {log_file}")
    return logger


class MetricsLogger:
    """Structured CSV logger for per-epoch and per-fold metrics."""

    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: Dict, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info(f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint


def log_config(config, experiment_name: str):
    """Save full config to JSON for reproducibility."""
    setup_dirs()
    config_dict = {}
    for field_name in ["paths", "encoder", "head", "pretrain", "domain_adapt", "gli_finetune"]:
        obj = getattr(config, field_name)
        config_dict[field_name] = {k: v for k, v in obj.__dict__.items()}
    config_dict["seed"] = config.seed
    config_dict["use_esm2"] = config.use_esm2

    filepath = os.path.join(OUTPUT_DIR, f"{experiment_name}_config.json")
    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logging.info(f"Config saved: {filepath}")


def log_environment():
    """Log package versions for reproducibility."""
    import transformers
    logging.info("=== Environment ===")
    logging.info(f"PyTorch: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"Transformers: {transformers.__version__}")
    logging.info(f"NumPy: {np.__version__}")


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            logging.info(f"[Timer] {self.label}: {self.elapsed:.2f}s")


def compute_gradient_norm(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for logging."""
    total_norm = 0.0
    max_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            count += 1
    total_norm = total_norm ** 0.5
    return {"grad_norm_total": total_norm, "grad_norm_max": max_norm, "grad_params": count}
