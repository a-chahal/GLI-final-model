"""
GLI-PLAPT Trainer — Three-stage training with extensive logging.

Stage 1: Pretraining on BindingDB (general protein-ligand interactions)
Stage 2: Domain adaptation on ZF data (zinc finger specificity)
Stage 3: GLI-specific LOOCV fine-tuning (GLI binder classification)
"""

import os
import logging
import copy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from src.config import Config, CHECKPOINT_DIR, LOG_DIR
from src.data import EmbeddedDataset, make_dataloader, compute_pos_weight
from src.model import BranchingPredictionHead
from src.utils import (
    MetricsLogger, save_checkpoint, load_checkpoint,
    compute_gradient_norm, Timer,
)


# ---------------------------------------------------------------------------
# Focal Loss (Phase 2B)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for binary classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified (easy) examples and focuses training
    on hard, misclassified examples. This is critical for our LOOCV
    setting where the Wen2023 cluster is easy and the original 6
    diverse compounds are hard.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from raw logits.

        Args:
            logits: (batch,) raw model output (before sigmoid)
            targets: (batch,) binary labels (0 or 1)
        """
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply pos_weight to positive examples (class imbalance correction)
        weight = torch.ones_like(targets)
        weight[targets == 1] = self.pos_weight

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # BCE component
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        loss = alpha_t * focal_weight * weight * bce
        return loss.mean()


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute all classification metrics."""
    preds = (probs >= threshold).astype(int)
    metrics = {}

    # Handle edge cases (all same class)
    if len(np.unique(labels)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["auprc"] = average_precision_score(labels, probs)

    metrics["mcc"] = matthews_corrcoef(labels, preds)
    metrics["precision"] = precision_score(labels, preds, zero_division=0)
    metrics["recall"] = recall_score(labels, preds, zero_division=0)
    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["accuracy"] = (preds == labels).mean()

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


class Trainer:
    """Handles training loop for a single stage."""

    def __init__(self, model: BranchingPredictionHead, config: Config,
                 device: torch.device, stage_name: str,
                 use_focal_loss: bool = False,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.75):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.stage_name = stage_name
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def train_stage(self, train_loader: DataLoader, val_loader: DataLoader,
                    lr: float, weight_decay: float, epochs: int, patience: int,
                    cosine_T0: int, pos_weight: float = 1.0) -> Dict:
        """Run a complete training stage with early stopping.

        Returns:
            dict with best metrics, best epoch, and training history
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"STAGE: {self.stage_name}")
        logging.info(f"{'='*60}")
        logging.info(f"  LR={lr}, WD={weight_decay}, epochs={epochs}, patience={patience}")
        logging.info(f"  Train batches={len(train_loader)}, Val batches={len(val_loader)}")
        logging.info(f"  pos_weight={pos_weight:.2f}")
        logging.info(f"  Trainable params: {self.model.count_parameters():,}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_T0, T_mult=1
        )
        if self.use_focal_loss:
            criterion = FocalLoss(
                gamma=self.focal_gamma, alpha=self.focal_alpha,
                pos_weight=pos_weight
            )
            logging.info(f"  Using Focal Loss (gamma={self.focal_gamma}, alpha={self.focal_alpha})")
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=self.device)
            )

        # Metrics logger
        csv_path = os.path.join(LOG_DIR, f"{self.stage_name}_epochs.csv")
        epoch_logger = MetricsLogger(csv_path, [
            "epoch", "train_loss", "val_loss", "train_auroc", "val_auroc",
            "val_auprc", "val_mcc", "val_precision", "val_recall", "val_f1",
            "lr", "grad_norm_total", "grad_norm_max", "epoch_time_s",
        ])

        best_val_loss = float("inf")
        best_metrics = {}
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            with Timer() as timer:
                # --- Train ---
                train_loss, train_labels, train_probs = self._train_epoch(
                    train_loader, optimizer, criterion
                )
                scheduler.step()  # Step per EPOCH, not per batch
                grad_info = compute_gradient_norm(self.model)

                # --- Validate ---
                val_loss, val_labels, val_probs = self._eval_epoch(val_loader, criterion)

            # Compute metrics
            train_metrics = compute_metrics(train_labels, train_probs)
            val_metrics = compute_metrics(val_labels, val_probs)

            current_lr = optimizer.param_groups[0]["lr"]

            # Log to CSV
            epoch_logger.log({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "train_auroc": f"{train_metrics['auroc']:.4f}" if not np.isnan(train_metrics['auroc']) else "nan",
                "val_auroc": f"{val_metrics['auroc']:.4f}" if not np.isnan(val_metrics['auroc']) else "nan",
                "val_auprc": f"{val_metrics['auprc']:.4f}" if not np.isnan(val_metrics['auprc']) else "nan",
                "val_mcc": f"{val_metrics['mcc']:.4f}",
                "val_precision": f"{val_metrics['precision']:.4f}",
                "val_recall": f"{val_metrics['recall']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "lr": f"{current_lr:.2e}",
                "grad_norm_total": f"{grad_info['grad_norm_total']:.4f}",
                "grad_norm_max": f"{grad_info['grad_norm_max']:.4f}",
                "epoch_time_s": f"{timer.elapsed:.1f}",
            })

            # Console log (concise)
            val_auroc_str = f"{val_metrics['auroc']:.4f}" if not np.isnan(val_metrics['auroc']) else "nan"
            logging.info(
                f"  [{self.stage_name}] Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"val_AUROC={val_auroc_str} val_MCC={val_metrics['mcc']:.4f} | "
                f"lr={current_lr:.2e} | {timer.elapsed:.1f}s"
            )

            # Early stopping on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{self.stage_name}_best.pt")
        save_checkpoint(self.model, optimizer, best_epoch, best_metrics, ckpt_path)

        logging.info(f"  Best epoch: {best_epoch} | val_loss={best_val_loss:.4f} | "
                     f"AUROC={best_metrics.get('auroc', 'nan')}")

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "checkpoint_path": ckpt_path,
        }

    def _unpack_batch(self, batch):
        """Unpack batch handling both 3-element (no Morgan) and 4-element (with Morgan) formats."""
        if len(batch) == 4:
            prot_emb, lig_emb, morgan_fp, labels = batch
            return prot_emb, lig_emb, morgan_fp, labels
        else:
            prot_emb, lig_emb, labels = batch
            return prot_emb, lig_emb, None, labels

    def _train_epoch(self, loader: DataLoader, optimizer, criterion):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        for batch in loader:
            prot_emb, lig_emb, morgan_fp, labels = self._unpack_batch(batch)
            prot_emb = prot_emb.to(self.device)
            lig_emb = lig_emb.to(self.device)
            labels = labels.float().to(self.device)
            if morgan_fp is not None:
                morgan_fp = morgan_fp.to(self.device)

            optimizer.zero_grad()
            logits = self.model(prot_emb, lig_emb, morgan_fp=morgan_fp).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        return avg_loss, np.array(all_labels), np.array(all_probs)

    def _eval_epoch(self, loader: DataLoader, criterion):
        """Single evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                prot_emb, lig_emb, morgan_fp, labels = self._unpack_batch(batch)
                prot_emb = prot_emb.to(self.device)
                lig_emb = lig_emb.to(self.device)
                labels = labels.float().to(self.device)
                if morgan_fp is not None:
                    morgan_fp = morgan_fp.to(self.device)

                logits = self.model(prot_emb, lig_emb, morgan_fp=morgan_fp).squeeze(-1)
                loss = criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        return avg_loss, np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Three-Stage Pipeline
# ---------------------------------------------------------------------------

def run_stage1_pretrain(model: BranchingPredictionHead, train_dataset: EmbeddedDataset,
                        config: Config, device: torch.device) -> Dict:
    """Stage 1: Pretrain on BindingDB."""
    cfg = config.pretrain
    n_val = int(len(train_dataset) * cfg.val_split)
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(config.seed))

    pw = compute_pos_weight([train_dataset.labels[i].item() for i in train_ds.indices])

    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False)

    trainer = Trainer(model, config, device, "stage1_pretrain")
    return trainer.train_stage(
        train_loader, val_loader,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        epochs=cfg.epochs, patience=cfg.patience,
        cosine_T0=cfg.cosine_T0, pos_weight=pw,
    )


def run_stage2_domain_adapt(model: BranchingPredictionHead, train_dataset: EmbeddedDataset,
                             config: Config, device: torch.device) -> Dict:
    """Stage 2: Domain adaptation on ZF data."""
    cfg = config.domain_adapt
    n_val = int(len(train_dataset) * cfg.val_split)
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(config.seed))

    pw = compute_pos_weight([train_dataset.labels[i].item() for i in train_ds.indices])

    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False)

    trainer = Trainer(model, config, device, "stage2_domain_adapt")
    return trainer.train_stage(
        train_loader, val_loader,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        epochs=cfg.epochs, patience=cfg.patience,
        cosine_T0=cfg.cosine_T0, pos_weight=pw,
    )


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                           metric: str = "youden") -> Tuple[float, Dict]:
    """Find the optimal classification threshold on a validation set.

    Strategies:
        'youden': Maximize Youden's J = sensitivity + specificity - 1
        'mcc': Maximize Matthews Correlation Coefficient

    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_score = -np.inf
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        if metric == "youden":
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            score = sensitivity + specificity - 1
        elif metric == "mcc":
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            score = (tp*tn - fp*fn) / max(denom, 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = t
            best_metrics = {
                "threshold": float(t),
                "score": float(score),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            }

    return best_thresh, best_metrics


def run_stage3_loocv_fold(model: BranchingPredictionHead,
                           train_dataset: EmbeddedDataset,
                           test_prot_emb: torch.Tensor,
                           test_lig_emb: torch.Tensor,
                           test_label: int,
                           config: Config, device: torch.device,
                           fold_id: int,
                           test_morgan_fp: Optional[torch.Tensor] = None) -> Dict:
    """Stage 3: Single LOOCV fold.

    Trains on all data except one held-out positive.
    Evaluates on the held-out positive + returns MC Dropout uncertainty.
    Also computes optimal threshold from validation set (Youden's J).
    Uses focal loss in Stage 3 when configured (Phase 2B).
    """
    cfg = config.gli_finetune
    # For single-fold, use 90/10 val split from training data
    n_val = max(1, int(len(train_dataset) * 0.1))
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(config.seed + fold_id))

    pw = compute_pos_weight([train_dataset.labels[i].item() for i in train_ds.indices])

    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False)

    trainer = Trainer(model, config, device, f"stage3_fold{fold_id}",
                      use_focal_loss=cfg.use_focal_loss,
                      focal_gamma=cfg.focal_gamma,
                      focal_alpha=cfg.focal_alpha)
    result = trainer.train_stage(
        train_loader, val_loader,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        epochs=cfg.epochs, patience=cfg.patience,
        cosine_T0=cfg.cosine_T0, pos_weight=pw,
    )

    # --- Find optimal threshold on validation set ---
    _, val_labels, val_probs = trainer._eval_epoch(
        val_loader, nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    )
    opt_thresh, thresh_info = find_optimal_threshold(val_labels, val_probs, metric="youden")
    result["optimal_threshold"] = opt_thresh
    result["threshold_info"] = thresh_info
    logging.info(f"  [Fold {fold_id}] Optimal threshold: {opt_thresh:.3f} "
                 f"(Youden's J={thresh_info['score']:.3f}, "
                 f"sens={thresh_info['sensitivity']:.3f}, "
                 f"spec={thresh_info['specificity']:.3f})")

    # MC Dropout prediction on held-out sample
    test_prot = test_prot_emb.unsqueeze(0).to(device)
    test_lig = test_lig_emb.unsqueeze(0).to(device)
    test_mfp = test_morgan_fp.unsqueeze(0).to(device) if test_morgan_fp is not None else None

    mc_result = model.mc_predict(test_prot, test_lig, n_samples=config.head.mc_samples,
                                  morgan_fp=test_mfp)

    result["held_out_prob"] = mc_result["mean_prob"].item()
    result["held_out_uncertainty"] = mc_result["std_prob"].item()
    result["held_out_label"] = test_label
    result["held_out_correct"] = int((mc_result["mean_prob"].item() >= 0.5) == test_label)
    result["held_out_correct_calibrated"] = int(
        (mc_result["mean_prob"].item() >= opt_thresh) == test_label
    )

    logging.info(
        f"  [Fold {fold_id}] Held-out: P(bind)={result['held_out_prob']:.4f} ± "
        f"{result['held_out_uncertainty']:.4f} | "
        f"correct@0.5={'YES' if result['held_out_correct'] else 'NO'} | "
        f"correct@{opt_thresh:.2f}={'YES' if result['held_out_correct_calibrated'] else 'NO'}"
    )

    return result
