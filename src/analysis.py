"""
GLI-PLAPT Phase 1 Analysis — Cross-encoder consensus scoring and ESM-2 truncation analysis.

These are post-hoc analyses that do not require retraining.
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.config import OUTPUT_DIR, LOG_DIR


# ---------------------------------------------------------------------------
# Phase 1C: Cross-Encoder Consensus Scoring
# ---------------------------------------------------------------------------

def consensus_analysis(baseline_results: Dict, modified_results: Dict,
                       threshold: float = 0.5) -> Dict:
    """Compute cross-encoder consensus between ProtBERT and ESM-2 models.

    Consensus strategies:
        1. Agreement: both models predict same class → high confidence
        2. Mean ensemble: average probabilities from both models
        3. Conservative: only predict positive if BOTH models agree

    Args:
        baseline_results: LOOCV results from ProtBERT model
        modified_results: LOOCV results from ESM-2 model
        threshold: classification threshold (use 0.5 or calibrated)

    Returns:
        dict with consensus metrics, per-compound analysis
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"CROSS-ENCODER CONSENSUS ANALYSIS")
    logging.info(f"{'='*60}")

    b_folds = baseline_results["fold_results"]
    m_folds = modified_results["fold_results"]

    assert len(b_folds) == len(m_folds), "Fold count mismatch"

    compounds = []
    b_probs = []
    m_probs = []
    b_correct = []
    m_correct = []

    for bf, mf in zip(b_folds, m_folds):
        compounds.append(bf["compound"])
        b_probs.append(bf["held_out_prob"])
        m_probs.append(mf["held_out_prob"])
        b_correct.append(bf["held_out_correct"])
        m_correct.append(mf["held_out_correct"])

    b_probs = np.array(b_probs)
    m_probs = np.array(m_probs)
    b_correct = np.array(b_correct)
    m_correct = np.array(m_correct)

    # --- Strategy 1: Mean ensemble ---
    ensemble_probs = (b_probs + m_probs) / 2
    ensemble_preds = (ensemble_probs >= threshold).astype(int)
    ensemble_correct = ensemble_preds  # all labels are 1 (held-out positives)
    ensemble_hit_rate = ensemble_correct.mean()

    # --- Strategy 2: Conservative consensus (both must agree positive) ---
    both_positive = ((b_probs >= threshold) & (m_probs >= threshold)).astype(int)
    conservative_hit_rate = both_positive.mean()

    # --- Strategy 3: Optimistic consensus (either predicts positive) ---
    either_positive = ((b_probs >= threshold) | (m_probs >= threshold)).astype(int)
    optimistic_hit_rate = either_positive.mean()

    # --- Agreement analysis ---
    b_preds = (b_probs >= threshold).astype(int)
    m_preds = (m_probs >= threshold).astype(int)
    agreement = (b_preds == m_preds).mean()

    # --- Confidence tiers ---
    # High confidence: both agree + low uncertainty
    # Medium: one agrees
    # Low: both disagree or near threshold
    tiers = []
    for i, comp in enumerate(compounds):
        if b_preds[i] == 1 and m_preds[i] == 1:
            tier = "HIGH"
        elif b_preds[i] == 1 or m_preds[i] == 1:
            tier = "MEDIUM"
        else:
            tier = "LOW"
        tiers.append(tier)

    # Log results
    logging.info(f"\n  Threshold: {threshold:.3f}")
    logging.info(f"  Agreement rate: {agreement:.2%}")
    logging.info(f"  --- Hit Rates ---")
    logging.info(f"  ProtBERT alone:            {b_correct.mean():.2%} ({b_correct.sum()}/{len(b_correct)})")
    logging.info(f"  ESM-2 alone:               {m_correct.mean():.2%} ({m_correct.sum()}/{len(m_correct)})")
    logging.info(f"  Mean ensemble:             {ensemble_hit_rate:.2%} ({ensemble_correct.sum()}/{len(ensemble_correct)})")
    logging.info(f"  Conservative (both agree): {conservative_hit_rate:.2%} ({both_positive.sum()}/{len(both_positive)})")
    logging.info(f"  Optimistic (either):       {optimistic_hit_rate:.2%} ({either_positive.sum()}/{len(either_positive)})")

    logging.info(f"\n  --- Per-Compound Consensus ---")
    for i, comp in enumerate(compounds):
        logging.info(
            f"    {comp:25s} ProtBERT={b_probs[i]:.3f} ESM-2={m_probs[i]:.3f} "
            f"Ensemble={ensemble_probs[i]:.3f} Tier={tiers[i]}"
        )

    return {
        "threshold": threshold,
        "agreement_rate": float(agreement),
        "hit_rate_protbert": float(b_correct.mean()),
        "hit_rate_esm2": float(m_correct.mean()),
        "hit_rate_ensemble": float(ensemble_hit_rate),
        "hit_rate_conservative": float(conservative_hit_rate),
        "hit_rate_optimistic": float(optimistic_hit_rate),
        "per_compound": {
            comp: {
                "protbert_prob": float(b_probs[i]),
                "esm2_prob": float(m_probs[i]),
                "ensemble_prob": float(ensemble_probs[i]),
                "confidence_tier": tiers[i],
                "protbert_hit": bool(b_correct[i]),
                "esm2_hit": bool(m_correct[i]),
                "ensemble_hit": bool(ensemble_correct[i]),
            }
            for i, comp in enumerate(compounds)
        },
    }


def consensus_fpr_analysis(baseline_results: Dict, modified_results: Dict,
                           baseline_neg_probs: np.ndarray,
                           modified_neg_probs: np.ndarray,
                           threshold: float = 0.5) -> Dict:
    """Compute FPR for consensus strategies on negative compounds.

    Args:
        baseline_neg_probs: negative probabilities from ProtBERT (last fold or mean)
        modified_neg_probs: negative probabilities from ESM-2 (last fold or mean)
    """
    logging.info(f"\n  --- Consensus FPR Analysis ---")

    # Mean ensemble
    ensemble_neg = (baseline_neg_probs + modified_neg_probs) / 2
    fpr_ensemble = (ensemble_neg > threshold).mean()

    # Conservative: both predict positive
    fpr_conservative = ((baseline_neg_probs > threshold) & (modified_neg_probs > threshold)).mean()

    # Individual
    fpr_protbert = (baseline_neg_probs > threshold).mean()
    fpr_esm2 = (modified_neg_probs > threshold).mean()

    logging.info(f"  FPR ProtBERT alone:  {fpr_protbert:.2%}")
    logging.info(f"  FPR ESM-2 alone:     {fpr_esm2:.2%}")
    logging.info(f"  FPR Ensemble (mean): {fpr_ensemble:.2%}")
    logging.info(f"  FPR Conservative:    {fpr_conservative:.2%}")

    return {
        "fpr_protbert": float(fpr_protbert),
        "fpr_esm2": float(fpr_esm2),
        "fpr_ensemble": float(fpr_ensemble),
        "fpr_conservative": float(fpr_conservative),
    }


# ---------------------------------------------------------------------------
# Phase 1D: ESM-2 Truncation Analysis
# ---------------------------------------------------------------------------

# GLI1 (P08151) domain annotations from UniProt
GLI1_DOMAINS = {
    "Zinc finger C2H2 1": (234, 258),
    "Zinc finger C2H2 2": (264, 288),
    "Zinc finger C2H2 3": (296, 318),
    "Zinc finger C2H2 4": (324, 348),
    "Zinc finger C2H2 5": (354, 376),
    "DNA-binding region": (234, 392),
    "N-terminal repressor domain": (1, 130),
    "C-terminal activation domain": (1020, 1106),
}

# Known binding sites for GLI inhibitors
GLI1_BINDING_SITES = {
    "GANT61/GANT61-D": {"site": "ZF2-3", "residues": (264, 318),
                         "note": "Binds between zinc fingers 2 and 3, disrupts DNA binding"},
    "GlaB": {"site": "ZF4-5", "residues": (324, 376),
             "note": "Binds between zinc fingers 4 and 5"},
    "JC19": {"site": "ZF4-5", "residues": (324, 376),
             "note": "Validated ZF4-5 binder"},
    "BAS07019774": {"site": "ZF4-5", "residues": (324, 376),
                     "note": "Validated ZF4-5 binder"},
    "Compound_1": {"site": "ZF2-3", "residues": (264, 318),
                    "note": "Validated ZF2-3 binder"},
    "Wen2023 compounds": {"site": "ZF2-3_inferred", "residues": (264, 318),
                           "note": "Scaffold match to Compound_1, Kd 8-28 μM"},
}


def esm2_truncation_analysis(gli1_sequence: str, esm2_max_length: int = 1024) -> Dict:
    """Analyze the impact of ESM-2's context window on GLI1 representation.

    ESM-2 tokenizes each amino acid as one token, plus [CLS] and [EOS].
    With max_length=1024, it encodes at most 1022 amino acids.

    GLI1 (P08151) is 1106 AA long, so ESM-2 loses the last 84 residues
    (positions 1023-1106), which includes the C-terminal activation domain.

    Args:
        gli1_sequence: Full GLI1 amino acid sequence
        esm2_max_length: ESM-2 max token length (default 1024)

    Returns:
        dict with truncation impact analysis
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"ESM-2 TRUNCATION ANALYSIS FOR GLI1")
    logging.info(f"{'='*60}")

    seq_length = len(gli1_sequence)
    # ESM-2 uses 2 special tokens: [CLS] at start, [EOS] at end
    max_aa = esm2_max_length - 2  # 1022 amino acids
    truncated_residues = max(0, seq_length - max_aa)
    truncated_pct = truncated_residues / seq_length * 100

    logging.info(f"  GLI1 sequence length: {seq_length} AA")
    logging.info(f"  ESM-2 max tokens: {esm2_max_length} (= {max_aa} AA after special tokens)")
    logging.info(f"  Truncated: {truncated_residues} AA ({truncated_pct:.1f}%)")
    logging.info(f"  Truncated region: positions {max_aa + 1}-{seq_length}")

    # Check which domains are affected
    affected_domains = []
    unaffected_domains = []
    for domain_name, (start, end) in GLI1_DOMAINS.items():
        if end > max_aa:
            overlap_with_truncation = max(0, end - max(start, max_aa))
            total_domain_length = end - start + 1
            pct_lost = overlap_with_truncation / total_domain_length * 100
            affected_domains.append({
                "name": domain_name,
                "range": f"{start}-{end}",
                "residues_lost": overlap_with_truncation,
                "total_length": total_domain_length,
                "pct_lost": pct_lost,
            })
            logging.info(f"  ⚠ AFFECTED: {domain_name} ({start}-{end}): "
                         f"{overlap_with_truncation}/{total_domain_length} residues lost ({pct_lost:.0f}%)")
        else:
            unaffected_domains.append(domain_name)

    # Check binding sites
    logging.info(f"\n  --- Binding Site Coverage ---")
    binding_site_analysis = {}
    for compound, info in GLI1_BINDING_SITES.items():
        start, end = info["residues"]
        fully_covered = end <= max_aa
        status = "FULLY COVERED" if fully_covered else "PARTIALLY TRUNCATED"
        binding_site_analysis[compound] = {
            "site": info["site"],
            "residues": f"{start}-{end}",
            "covered_by_esm2": fully_covered,
            "note": info["note"],
        }
        logging.info(f"    {compound:25s} {info['site']:15s} ({start}-{end}) → {status}")

    # ProtBERT comparison
    protbert_max_aa = 3200 - 2  # 3198 AA capacity
    protbert_truncated = max(0, seq_length - protbert_max_aa)
    logging.info(f"\n  --- Comparison ---")
    logging.info(f"  ProtBERT max AA: {protbert_max_aa} → {protbert_truncated} residues truncated (NONE)")
    logging.info(f"  ESM-2 max AA:    {max_aa} → {truncated_residues} residues truncated")
    logging.info(f"  ProtBERT sees FULL GLI1 sequence; ESM-2 misses C-terminal activation domain")

    # Key finding
    key_finding = (
        f"ESM-2 truncates GLI1 at position {max_aa}, losing {truncated_residues} C-terminal "
        f"residues ({truncated_pct:.1f}%). This includes the C-terminal activation domain "
        f"(residues 1020-{seq_length}), which is important for GLI1 transcriptional activity "
        f"and protein-compound interactions. All zinc finger binding sites (ZF2-5, residues "
        f"234-376) are fully retained. ProtBERT (max 3198 AA) sees the complete sequence. "
        f"This truncation may explain ESM-2's lower sensitivity: while it retains the direct "
        f"binding site information, it loses long-range context from the activation domain "
        f"that ProtBERT captures."
    )
    logging.info(f"\n  KEY FINDING: {key_finding}")

    return {
        "seq_length": seq_length,
        "esm2_max_aa": max_aa,
        "truncated_residues": truncated_residues,
        "truncated_pct": truncated_pct,
        "affected_domains": affected_domains,
        "unaffected_domains": unaffected_domains,
        "binding_site_analysis": binding_site_analysis,
        "protbert_truncated": protbert_truncated,
        "key_finding": key_finding,
    }


# ---------------------------------------------------------------------------
# Multi-Seed Aggregation
# ---------------------------------------------------------------------------

def aggregate_multi_seed_results(seed_results: Dict[int, Dict],
                                 model_variant: str = "") -> Dict:
    """Aggregate results across multiple random seeds.

    Args:
        seed_results: {seed: pipeline_result} dict
        model_variant: "esm2" or "protbert" for logging

    Returns:
        dict with mean ± std for all key metrics, per-compound consistency
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"MULTI-SEED AGGREGATION: {model_variant} ({len(seed_results)} seeds)")
    logging.info(f"{'='*60}")

    seeds = sorted(seed_results.keys())
    n_seeds = len(seeds)

    # Collect per-seed metrics
    hit_rates = []
    hit_rates_cal = []
    mean_probs = []
    mean_fprs = []
    mean_fprs_cal = []
    per_compound_probs = {}  # {compound: [prob_seed1, prob_seed2, ...]}
    per_compound_hits = {}   # {compound: [hit_seed1, hit_seed2, ...]}

    for seed in seeds:
        s3 = seed_results[seed]["stage3"]
        hit_rates.append(s3["hit_rate"])
        hit_rates_cal.append(s3.get("hit_rate_calibrated", s3["hit_rate"]))
        mean_probs.append(s3["mean_prob"])
        mean_fprs.append(s3.get("mean_fpr_default", np.nan))
        mean_fprs_cal.append(s3.get("mean_fpr_calibrated", np.nan))

        for fold_r in s3["fold_results"]:
            comp = fold_r["compound"]
            if comp not in per_compound_probs:
                per_compound_probs[comp] = []
                per_compound_hits[comp] = []
            per_compound_probs[comp].append(fold_r["held_out_prob"])
            per_compound_hits[comp].append(fold_r["held_out_correct"])

    hit_rates = np.array(hit_rates)
    hit_rates_cal = np.array(hit_rates_cal)
    mean_probs = np.array(mean_probs)
    mean_fprs = np.array(mean_fprs)
    mean_fprs_cal = np.array(mean_fprs_cal)

    # Summary stats
    logging.info(f"  Seeds: {seeds}")
    logging.info(f"  Hit rate @0.5:    {hit_rates.mean():.2%} ± {hit_rates.std():.2%} "
                 f"(range: {hit_rates.min():.2%}-{hit_rates.max():.2%})")
    logging.info(f"  Hit rate @cal:    {hit_rates_cal.mean():.2%} ± {hit_rates_cal.std():.2%}")
    logging.info(f"  Mean P(binder):   {mean_probs.mean():.4f} ± {mean_probs.std():.4f}")
    logging.info(f"  FPR @0.5:         {np.nanmean(mean_fprs):.2%} ± {np.nanstd(mean_fprs):.2%}")
    logging.info(f"  FPR @calibrated:  {np.nanmean(mean_fprs_cal):.2%} ± {np.nanstd(mean_fprs_cal):.2%}")

    # Per-compound consistency
    logging.info(f"\n  --- Per-Compound Consistency ({n_seeds} seeds) ---")
    compound_summary = {}
    for comp in per_compound_probs:
        probs_arr = np.array(per_compound_probs[comp])
        hits_arr = np.array(per_compound_hits[comp])
        consistency = hits_arr.mean()  # fraction of seeds where it was a HIT
        compound_summary[comp] = {
            "mean_prob": float(probs_arr.mean()),
            "std_prob": float(probs_arr.std()),
            "hit_consistency": float(consistency),
            "hits_per_seed": int(hits_arr.sum()),
            "total_seeds": n_seeds,
        }
        status = "STABLE HIT" if consistency == 1.0 else \
                 "STABLE MISS" if consistency == 0.0 else \
                 f"VARIABLE ({consistency:.0%})"
        logging.info(
            f"    {comp:25s} P={probs_arr.mean():.4f}±{probs_arr.std():.4f} "
            f"Hit {hits_arr.sum()}/{n_seeds} seeds → {status}"
        )

    result = {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "hit_rate_mean": float(hit_rates.mean()),
        "hit_rate_std": float(hit_rates.std()),
        "hit_rate_cal_mean": float(hit_rates_cal.mean()),
        "hit_rate_cal_std": float(hit_rates_cal.std()),
        "mean_prob_mean": float(mean_probs.mean()),
        "mean_prob_std": float(mean_probs.std()),
        "fpr_default_mean": float(np.nanmean(mean_fprs)),
        "fpr_default_std": float(np.nanstd(mean_fprs)),
        "fpr_cal_mean": float(np.nanmean(mean_fprs_cal)),
        "fpr_cal_std": float(np.nanstd(mean_fprs_cal)),
        "per_compound": compound_summary,
    }

    return result


def save_phase1_results(all_results: Dict, output_dir: str = OUTPUT_DIR):
    """Save all Phase 1 analysis results to a structured JSON file."""
    filepath = os.path.join(output_dir, "phase1_results.json")
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"\nPhase 1 results saved: {filepath}")
    return filepath
