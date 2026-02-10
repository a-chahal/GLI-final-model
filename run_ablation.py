#!/usr/bin/env python
"""GLI-PLAPT Ablation Study — Systematic component contribution analysis.

Removes one component at a time and measures impact on LOOCV hit rate.

Conditions:
    full             Full model with all features (baseline)
    no_morgan        Remove Morgan fingerprint branch
    no_focal         Replace focal loss with BCE
    no_augmentation  Remove all SMILES augmentation
    no_pretrain      Skip Stage 1 BindingDB pretraining
    no_domain_adapt  Skip Stage 2 zinc finger domain adaptation

Usage:
    # Run a single condition on a specific GPU:
    CUDA_VISIBLE_DEVICES=0 python run_ablation.py --condition full

    # Collect results from all completed conditions:
    python run_ablation.py --collect
"""

import argparse
import copy
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# --- Imports from existing pipeline ---
from run import PrecomputedEmbeddings, precompute_all_embeddings, precompute_embeddings
from src.config import Config, OUTPUT_DIR
from src.data import EmbeddedDataset
from src.model import BranchingPredictionHead
from src.trainer import run_stage1_pretrain, run_stage2_domain_adapt
from src.evaluate import run_loocv
from src.utils import set_seed, setup_logging, get_device, Timer

# Module references for monkey-patching output directories
import src.config as _cfg
import src.trainer as _trainer
import src.evaluate as _evaluate
import src.utils as _utils


# ─── Constants ──────────────────────────────────────────────────────────────

ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")
SEED = 42

CONDITIONS = [
    "full",
    "no_morgan",
    "no_focal",
    "no_augmentation",
    "no_pretrain",
    "no_domain_adapt",
]

DESCRIPTIONS = {
    "full":            "Full model (all features)",
    "no_morgan":       "No Morgan fingerprints",
    "no_focal":        "No focal loss (BCE instead)",
    "no_augmentation": "No SMILES augmentation",
    "no_pretrain":     "No BindingDB pretraining (Stage 1 skipped)",
    "no_domain_adapt": "No ZF domain adaptation (Stage 2 skipped)",
}


# ─── Helper functions ───────────────────────────────────────────────────────

def setup_condition_dirs(condition: str):
    """Create output directories for a specific ablation condition."""
    cond_dir = os.path.join(ABLATION_DIR, condition)
    cond_ckpt = os.path.join(cond_dir, "checkpoints")
    cond_log = os.path.join(cond_dir, "logs")
    for d in [cond_dir, cond_ckpt, cond_log]:
        os.makedirs(d, exist_ok=True)
    return cond_dir, cond_ckpt, cond_log


def redirect_outputs(cond_dir: str, cond_ckpt: str, cond_log: str):
    """Monkey-patch output directories in all modules so checkpoints and
    CSV logs are written to condition-specific folders."""
    for mod in [_cfg, _utils]:
        mod.OUTPUT_DIR = cond_dir
        mod.LOG_DIR = cond_log
        mod.CHECKPOINT_DIR = cond_ckpt
    for mod in [_trainer, _evaluate]:
        mod.LOG_DIR = cond_log
        mod.CHECKPOINT_DIR = cond_ckpt


def make_config(condition: str) -> Config:
    """Create Config for a specific ablation condition."""
    config = Config(use_esm2=True, seed=SEED)
    if condition == "no_morgan":
        config.head.use_morgan_fp = False
    elif condition == "no_focal":
        config.gli_finetune.use_focal_loss = False
    elif condition == "no_augmentation":
        config.gli_finetune.smiles_augment_per_positive = 0
        config.gli_finetune.use_asymmetric_aug = False
    # no_pretrain and no_domain_adapt handled in pipeline logic
    return config


# ─── Run a single ablation condition ────────────────────────────────────────

def run_condition(condition: str, device: torch.device) -> Dict:
    """Run the full 3-stage pipeline for one ablation condition."""
    logging.info(f"\n{'#'*70}")
    logging.info(f"# ABLATION: {condition} -- {DESCRIPTIONS[condition]}")
    logging.info(f"{'#'*70}\n")

    config = make_config(condition)
    set_seed(SEED)

    # ── Precompute embeddings (cache-backed, fast after first run) ──────
    # For no_augmentation: use the full config so augmented SMILES are
    # generated and cached (avoids torch.stack([]) crash on 0 augments).
    # We simply won't pass the augmented dicts to LOOCV.
    if condition == "no_augmentation":
        embed_config = Config(use_esm2=True, seed=SEED)
    else:
        embed_config = make_config(condition)

    with Timer(f"Embedding precomputation [{condition}]"):
        embs = precompute_all_embeddings(embed_config, device)

    # ── Build model ─────────────────────────────────────────────────────
    model = BranchingPredictionHead(embs.prot_dim, embs.lig_dim, config.head)
    logging.info(f"Prediction head: {model.count_parameters():,} trainable params")

    # ── Build datasets ──────────────────────────────────────────────────
    s1_dataset = EmbeddedDataset(embs.s1_prot_embs, embs.s1_lig_embs, embs.s1_labels)
    s2_dataset = EmbeddedDataset(embs.s2_prot_embs, embs.s2_lig_embs, embs.s2_labels)

    # ── Stage 1: BindingDB pretraining ──────────────────────────────────
    if condition == "no_pretrain":
        logging.info("ABLATION: Skipping Stage 1 (random initialization)")
        s1_ckpt = None
    else:
        with Timer(f"Stage 1 [{condition}]"):
            s1_result = run_stage1_pretrain(model, s1_dataset, config, device)
        s1_ckpt = s1_result["checkpoint_path"]

    # ── Stage 2: ZF domain adaptation ───────────────────────────────────
    if condition == "no_domain_adapt":
        logging.info("ABLATION: Skipping Stage 2 (using Stage 1 checkpoint)")
        stage2_ckpt = s1_ckpt
    else:
        with Timer(f"Stage 2 [{condition}]"):
            s2_result = run_stage2_domain_adapt(model, s2_dataset, config, device)
        stage2_ckpt = s2_result["checkpoint_path"]

    # ── Stage 3: LOOCV ──────────────────────────────────────────────────
    # Override augmented dicts for no_augmentation
    aug_lig = embs.augmented_lig_embs
    aug_mfp = embs.augmented_morgan_fps
    if condition == "no_augmentation":
        aug_lig = {}
        aug_mfp = {}

    with Timer(f"Stage 3 LOOCV [{condition}]"):
        s3_result = run_loocv(
            model_template=model,
            stage2_checkpoint=stage2_ckpt,
            positive_prot_embs=embs.s3_pos_prot_embs,
            positive_lig_embs=embs.s3_pos_lig_embs,
            positive_names=embs.pos_names,
            negative_prot_embs=embs.s3_neg_prot_embs,
            negative_lig_embs=embs.s3_neg_lig_embs,
            augmented_positive_lig_embs=aug_lig,
            config=config,
            device=device,
            supp_pos_prot_embs=embs.s3_supp_pos_prot_embs,
            supp_pos_lig_embs=embs.s3_supp_pos_lig_embs,
            gli_neg_prot_embs=embs.s3_gli_neg_prot_embs,
            gli_neg_lig_embs=embs.s3_gli_neg_lig_embs,
            model_variant=f"ablation_{condition}",
            positive_morgan_fps=embs.pos_morgan_fps,
            negative_morgan_fps=embs.neg_morgan_fps,
            augmented_morgan_fps=aug_mfp,
            supp_pos_morgan_fps=embs.supp_pos_morgan_fps,
            gli_neg_morgan_fps=embs.gli_neg_morgan_fps,
        )

    # ── Save results ────────────────────────────────────────────────────
    result = {
        "condition": condition,
        "description": DESCRIPTIONS[condition],
        "seed": SEED,
        "hit_rate": s3_result["hit_rate"],
        "hit_rate_calibrated": s3_result.get("hit_rate_calibrated", s3_result["hit_rate"]),
        "mean_prob": s3_result["mean_prob"],
        "std_prob": s3_result.get("std_prob", 0),
        "mean_uncertainty": s3_result["mean_uncertainty"],
        "mean_fpr_default": s3_result.get("mean_fpr_default", None),
        "mean_fpr_calibrated": s3_result.get("mean_fpr_calibrated", None),
        "per_compound": s3_result["per_compound"],
        "fold_results": [{
            "compound": r["compound"],
            "prob": r["held_out_prob"],
            "uncertainty": r["held_out_uncertainty"],
            "correct": r["held_out_correct"],
            "correct_calibrated": r.get("held_out_correct_calibrated", r["held_out_correct"]),
        } for r in s3_result["fold_results"]],
    }

    cond_dir = os.path.join(ABLATION_DIR, condition)
    result_path = os.path.join(cond_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logging.info(f"\nResults saved: {result_path}")

    return result


# ─── Collect and display results ────────────────────────────────────────────

def collect_results():
    """Read all completed condition results and print summary table."""
    results = {}
    for condition in CONDITIONS:
        path = os.path.join(ABLATION_DIR, condition, "result.json")
        if os.path.exists(path):
            with open(path) as f:
                results[condition] = json.load(f)

    if not results:
        print("No ablation results found. Run conditions first.")
        return

    full = results.get("full")
    if full is None:
        print("WARNING: Full-model baseline not yet available. Deltas shown as N/A.\n")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("GLI-PLAPT ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Condition':<22} {'Hit Rate':>10} {'Delta':>8} "
          f"{'Mean P':>8} {'FPR':>7} {'Hits':>8}")
    print("  " + "-" * 68)

    for cond in CONDITIONS:
        if cond not in results:
            print(f"  {cond:<22} {'(pending)':>10}")
            continue
        r = results[cond]
        hr = r["hit_rate"]
        mp = r["mean_prob"]
        fpr = r.get("mean_fpr_default") or 0
        n_total = len(r.get("fold_results", []))
        n_hits = sum(1 for fr in r.get("fold_results", []) if fr["correct"])

        if full and cond != "full":
            delta = hr - full["hit_rate"]
            delta_str = f"{delta:+.1%}"
        elif cond == "full":
            delta_str = "  --"
        else:
            delta_str = " N/A"

        print(f"  {cond:<22} {hr:>9.1%} {delta_str:>8} "
              f"{mp:>8.3f} {fpr:>6.1%} {n_hits:>3}/{n_total}")

    # ── Per-compound breakdown ──────────────────────────────────────────
    if full:
        compounds = [fr["compound"] for fr in full["fold_results"]]
        avail = [c for c in CONDITIONS if c in results]

        print(f"\n  {'Compound':<22}", end="")
        for cond in avail:
            print(f" {cond[:13]:>13}", end="")
        print()
        print("  " + "-" * (20 + 14 * len(avail)))

        for comp in compounds:
            print(f"  {comp:<22}", end="")
            for cond in avail:
                r = results[cond]
                match = [fr for fr in r["fold_results"] if fr["compound"] == comp]
                if match:
                    prob = match[0]["prob"]
                    mark = " HIT" if prob >= 0.5 else "MISS"
                    print(f" {prob:>7.3f} {mark}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            print()

    # ── Save combined summary JSON ──────────────────────────────────────
    summary = {}
    for cond in CONDITIONS:
        if cond in results:
            r = results[cond]
            summary[cond] = {
                "description": DESCRIPTIONS[cond],
                "hit_rate": r["hit_rate"],
                "delta_hr": (r["hit_rate"] - full["hit_rate"]) if full and cond != "full" else 0,
                "mean_prob": r["mean_prob"],
                "mean_fpr": r.get("mean_fpr_default"),
            }

    summary_path = os.path.join(ABLATION_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print()


# ─── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLI-PLAPT Ablation Study")
    parser.add_argument("--condition", choices=CONDITIONS,
                        help="Ablation condition to run")
    parser.add_argument("--collect", action="store_true",
                        help="Collect and display results from all conditions")
    args = parser.parse_args()

    if args.collect:
        collect_results()
        return

    if args.condition is None:
        parser.error("Specify --condition <name> or --collect")

    # Setup condition-specific output directories BEFORE logging
    os.makedirs(ABLATION_DIR, exist_ok=True)
    cond_dir, cond_ckpt, cond_log = setup_condition_dirs(args.condition)
    redirect_outputs(cond_dir, cond_ckpt, cond_log)

    # Setup logging (uses the redirected LOG_DIR)
    setup_logging(f"ablation_{args.condition}")
    device = get_device()

    logging.info(f"Ablation condition: {args.condition}")
    logging.info(f"Description: {DESCRIPTIONS[args.condition]}")
    logging.info(f"Device: {device}")
    logging.info(f"Output: {cond_dir}")

    result = run_condition(args.condition, device)

    # Final summary
    hr = result["hit_rate"]
    mp = result["mean_prob"]
    fpr = result.get("mean_fpr_default") or 0
    n_total = len(result.get("fold_results", []))
    n_hits = sum(1 for fr in result["fold_results"] if fr["correct"])

    logging.info(f"\n{'='*60}")
    logging.info(f"ABLATION COMPLETE: {args.condition}")
    logging.info(f"  Hit Rate:     {hr:.1%} ({n_hits}/{n_total})")
    logging.info(f"  Mean P(bind): {mp:.4f}")
    logging.info(f"  FPR @0.5:     {fpr:.1%}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
