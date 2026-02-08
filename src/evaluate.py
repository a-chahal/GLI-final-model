"""
GLI-PLAPT Evaluation — LOOCV orchestration, MC Dropout analysis, and statistical tests.
"""

import os
import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import Config, CHECKPOINT_DIR, LOG_DIR
from src.data import EmbeddedDataset, make_dataloader, compute_pos_weight, randomize_smiles
from src.model import BranchingPredictionHead, EncoderWrapper
from src.trainer import run_stage3_loocv_fold, compute_metrics
from src.utils import MetricsLogger, load_checkpoint, Timer


def run_loocv(model_template: BranchingPredictionHead,
              stage2_checkpoint: str,
              positive_prot_embs: torch.Tensor,
              positive_lig_embs: torch.Tensor,
              positive_names: List[str],
              negative_prot_embs: torch.Tensor,
              negative_lig_embs: torch.Tensor,
              augmented_positive_lig_embs: Dict[int, torch.Tensor],
              config: Config,
              device: torch.device,
              supp_pos_prot_embs: torch.Tensor = None,
              supp_pos_lig_embs: torch.Tensor = None,
              gli_neg_prot_embs: torch.Tensor = None,
              gli_neg_lig_embs: torch.Tensor = None,
              model_variant: str = "") -> Dict:
    """Run full Leave-One-Out Cross-Validation on GLI binders.

    For each fold:
        1. Hold out one positive
        2. Load Stage 2 checkpoint (fresh start per fold)
        3. Train on remaining positives (+ augmented + supplementary) + all negatives
        4. MC Dropout predict on held-out positive

    Args:
        model_template: Uninitialized model (for architecture)
        stage2_checkpoint: Path to Stage 2 best checkpoint
        positive_prot_embs: (n_pos, prot_dim) protein embeddings for positives
        positive_lig_embs: (n_pos, lig_dim) ligand embeddings for positives (canonical)
        positive_names: List of compound names
        negative_prot_embs: (n_neg, prot_dim) protein embeddings for negatives
        negative_lig_embs: (n_neg, lig_dim) ligand embeddings for negatives
        augmented_positive_lig_embs: {pos_idx: (n_aug, lig_dim)} augmented ligand embeddings
        config: Config object
        device: torch device
        supp_pos_prot_embs: ChEMBL GLI supplementary positives (always in training)
        supp_pos_lig_embs: ChEMBL GLI supplementary positive ligand embeddings
        gli_neg_prot_embs: ChEMBL GLI true negatives (tested, confirmed non-binders)
        gli_neg_lig_embs: ChEMBL GLI true negative ligand embeddings

    Returns:
        dict with per-fold results and aggregated metrics
    """
    n_pos = len(positive_names)
    logging.info(f"\n{'='*60}")
    logging.info(f"STAGE 3: LOOCV ({n_pos} folds)")
    logging.info(f"{'='*60}")

    # CSV logger for per-fold results (prefixed by model variant to avoid overwrites)
    prefix = f"{model_variant}_" if model_variant else ""
    fold_logger = MetricsLogger(
        os.path.join(LOG_DIR, f"{prefix}stage3_loocv_folds.csv"),
        ["fold", "compound", "held_out_prob", "held_out_uncertainty",
         "held_out_correct", "best_epoch", "best_val_loss",
         "val_auroc", "val_auprc", "val_mcc"]
    )

    fold_results = []

    for fold_idx in range(n_pos):
        logging.info(f"\n--- LOOCV Fold {fold_idx + 1}/{n_pos}: "
                     f"holding out {positive_names[fold_idx]} ---")

        # Fresh model from Stage 2 checkpoint
        model = copy.deepcopy(model_template)
        load_checkpoint(stage2_checkpoint, model)
        model = model.to(device)

        # Build training set: all positives except fold_idx + augmented + supplementary + all negatives
        train_prot_list = []
        train_lig_list = []
        train_labels = []

        for i in range(n_pos):
            if i == fold_idx:
                continue
            # Canonical positive
            train_prot_list.append(positive_prot_embs[i])
            train_lig_list.append(positive_lig_embs[i])
            train_labels.append(1)

            # Augmented positives
            if i in augmented_positive_lig_embs:
                aug_embs = augmented_positive_lig_embs[i]
                for j in range(len(aug_embs)):
                    train_prot_list.append(positive_prot_embs[i])
                    train_lig_list.append(aug_embs[j])
                    train_labels.append(1)

        # ChEMBL GLI supplementary positives (always in training, never held out)
        if supp_pos_prot_embs is not None and supp_pos_lig_embs is not None:
            for i in range(len(supp_pos_lig_embs)):
                train_prot_list.append(supp_pos_prot_embs[i])
                train_lig_list.append(supp_pos_lig_embs[i])
                train_labels.append(1)

        # All negatives (SMO + structural)
        for i in range(len(negative_lig_embs)):
            train_prot_list.append(negative_prot_embs[i])
            train_lig_list.append(negative_lig_embs[i])
            train_labels.append(0)

        # ChEMBL GLI true negatives (tested against GLI, confirmed non-binders)
        if gli_neg_prot_embs is not None and gli_neg_lig_embs is not None:
            for i in range(len(gli_neg_lig_embs)):
                train_prot_list.append(gli_neg_prot_embs[i])
                train_lig_list.append(gli_neg_lig_embs[i])
                train_labels.append(0)

        train_dataset = EmbeddedDataset(
            protein_embeds=torch.stack(train_prot_list),
            ligand_embeds=torch.stack(train_lig_list),
            labels=torch.tensor(train_labels, dtype=torch.float32),
        )

        logging.info(f"  Fold {fold_idx + 1} training set: {len(train_dataset)} samples "
                     f"(pos={sum(train_labels)}, neg={len(train_labels) - sum(train_labels)})")

        # Run fold
        result = run_stage3_loocv_fold(
            model=model,
            train_dataset=train_dataset,
            test_prot_emb=positive_prot_embs[fold_idx],
            test_lig_emb=positive_lig_embs[fold_idx],
            test_label=1,
            config=config,
            device=device,
            fold_id=fold_idx + 1,
        )

        result["compound"] = positive_names[fold_idx]
        fold_results.append(result)

        # Log fold to CSV
        bm = result.get("best_metrics", {})
        fold_logger.log({
            "fold": fold_idx + 1,
            "compound": positive_names[fold_idx],
            "held_out_prob": f"{result['held_out_prob']:.4f}",
            "held_out_uncertainty": f"{result['held_out_uncertainty']:.4f}",
            "held_out_correct": result["held_out_correct"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": f"{result['best_val_loss']:.4f}",
            "val_auroc": f"{bm.get('auroc', 'nan')}",
            "val_auprc": f"{bm.get('auprc', 'nan')}",
            "val_mcc": f"{bm.get('mcc', 'nan')}",
        })

    # Aggregate results
    agg = aggregate_loocv_results(fold_results)
    return agg


def aggregate_loocv_results(fold_results: List[Dict]) -> Dict:
    """Aggregate LOOCV fold results into summary statistics."""
    probs = [r["held_out_prob"] for r in fold_results]
    uncertainties = [r["held_out_uncertainty"] for r in fold_results]
    corrects = [r["held_out_correct"] for r in fold_results]
    compounds = [r["compound"] for r in fold_results]

    hit_rate = sum(corrects) / len(corrects)
    mean_prob = np.mean(probs)
    std_prob = np.std(probs)
    mean_uncertainty = np.mean(uncertainties)

    logging.info(f"\n{'='*60}")
    logging.info(f"LOOCV AGGREGATE RESULTS")
    logging.info(f"{'='*60}")
    logging.info(f"  Hit rate: {hit_rate:.2%} ({sum(corrects)}/{len(corrects)})")
    logging.info(f"  Mean P(binder): {mean_prob:.4f} ± {std_prob:.4f}")
    logging.info(f"  Mean uncertainty: {mean_uncertainty:.4f}")

    for i, (comp, prob, unc, correct) in enumerate(zip(compounds, probs, uncertainties, corrects)):
        status = "HIT" if correct else "MISS"
        logging.info(f"    {comp:25s} P={prob:.4f} σ={unc:.4f} [{status}]")

    return {
        "fold_results": fold_results,
        "hit_rate": hit_rate,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "mean_uncertainty": mean_uncertainty,
        "per_compound": dict(zip(compounds, probs)),
    }


def evaluate_on_negatives(model: BranchingPredictionHead,
                          negative_prot_embs: torch.Tensor,
                          negative_lig_embs: torch.Tensor,
                          config: Config,
                          device: torch.device) -> Dict:
    """Evaluate trained model on all negatives using MC Dropout.

    Returns distribution of P(binder) and uncertainty for negatives.
    """
    model = model.to(device)
    mc_result = model.mc_predict(
        negative_prot_embs.to(device),
        negative_lig_embs.to(device),
        n_samples=config.head.mc_samples,
    )

    neg_probs = mc_result["mean_prob"].cpu().numpy()
    neg_uncerts = mc_result["std_prob"].cpu().numpy()

    logging.info(f"  Negatives: mean P(bind)={neg_probs.mean():.4f} ± {neg_probs.std():.4f}")
    logging.info(f"  Negatives: mean uncertainty={neg_uncerts.mean():.4f}")
    logging.info(f"  Negatives: false positive rate (P>0.5)={( neg_probs > 0.5).mean():.2%}")

    return {
        "neg_probs": neg_probs,
        "neg_uncerts": neg_uncerts,
        "mean_prob": neg_probs.mean(),
        "fpr": (neg_probs > 0.5).mean(),
    }


# ---------------------------------------------------------------------------
# Statistical Comparison: Baseline vs Modified
# ---------------------------------------------------------------------------

def compare_models(baseline_results: Dict, modified_results: Dict) -> Dict:
    """Statistical comparison between baseline PLAPT and GLI-PLAPT.

    Tests:
        1. McNemar's test on LOOCV hit/miss patterns
        2. Paired t-test on per-fold probabilities
        3. Bootstrap CI for hit rate difference
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"STATISTICAL COMPARISON: Baseline vs Modified")
    logging.info(f"{'='*60}")

    b_folds = baseline_results["fold_results"]
    m_folds = modified_results["fold_results"]

    b_probs = np.array([r["held_out_prob"] for r in b_folds])
    m_probs = np.array([r["held_out_prob"] for r in m_folds])
    b_correct = np.array([r["held_out_correct"] for r in b_folds])
    m_correct = np.array([r["held_out_correct"] for r in m_folds])

    # 1. McNemar's test
    b_right_m_wrong = ((b_correct == 1) & (m_correct == 0)).sum()
    b_wrong_m_right = ((b_correct == 0) & (m_correct == 1)).sum()
    if b_right_m_wrong + b_wrong_m_right > 0:
        # Use exact binomial test for small N
        mcnemar_p = stats.binomtest(
            b_wrong_m_right, b_right_m_wrong + b_wrong_m_right, 0.5
        ).pvalue
    else:
        mcnemar_p = 1.0
    logging.info(f"  McNemar's test p-value: {mcnemar_p:.4f}")

    # 2. Paired t-test on probabilities
    t_stat, t_pval = stats.ttest_rel(m_probs, b_probs)
    logging.info(f"  Paired t-test: t={t_stat:.4f}, p={t_pval:.4f}")
    logging.info(f"  Mean prob diff (modified - baseline): {(m_probs - b_probs).mean():.4f}")

    # 3. Bootstrap 95% CI for hit rate difference
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(b_correct), size=len(b_correct), replace=True)
        b_hr = b_correct[idx].mean()
        m_hr = m_correct[idx].mean()
        diffs.append(m_hr - b_hr)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    logging.info(f"  Bootstrap 95% CI for hit rate diff: [{ci_low:.4f}, {ci_high:.4f}]")

    # Summary
    b_hr = baseline_results["hit_rate"]
    m_hr = modified_results["hit_rate"]
    logging.info(f"\n  Baseline hit rate:  {b_hr:.2%}")
    logging.info(f"  Modified hit rate:  {m_hr:.2%}")
    logging.info(f"  Baseline mean prob: {baseline_results['mean_prob']:.4f}")
    logging.info(f"  Modified mean prob: {modified_results['mean_prob']:.4f}")

    return {
        "mcnemar_p": mcnemar_p,
        "ttest_t": t_stat,
        "ttest_p": t_pval,
        "bootstrap_ci": (ci_low, ci_high),
        "baseline_hit_rate": b_hr,
        "modified_hit_rate": m_hr,
        "prob_diffs": m_probs - b_probs,
    }
