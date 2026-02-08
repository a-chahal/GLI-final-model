"""GLI-PLAPT Main Pipeline — Multi-seed training + LOOCV + Phase 1+2 analyses.

Usage:
    python run.py --mode both          # Run baseline + modified + comparison
    python run.py --mode modified      # Run only modified (ESM-2) model
    python run.py --mode baseline      # Run only baseline (ProtBERT) model
    python run.py --seeds 42           # Single seed (default)
    python run.py --seeds all          # All 5 seeds for statistical robustness
    python run.py --seeds 42,123,456   # Custom seed list
"""

import argparse
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

from src.config import Config, LOG_DIR, OUTPUT_DIR, MULTI_SEEDS
from src.data import (
    load_bindingdb, load_zf_data, load_gli_data,
    EmbeddedDataset, randomize_smiles, EmbeddingCache,
    compute_morgan_fps_batch, compute_asymmetric_aug_counts,
    compute_morgan_fp,
)
from src.model import BranchingPredictionHead, EncoderWrapper
from src.trainer import run_stage1_pretrain, run_stage2_domain_adapt
from src.evaluate import run_loocv, compare_models
from src.analysis import (
    consensus_analysis, esm2_truncation_analysis,
    aggregate_multi_seed_results, save_all_results,
    tanimoto_similarity_analysis,
)
from src.utils import (
    set_seed, setup_dirs, setup_logging, get_device,
    log_config, log_environment, Timer,
)


@dataclass
class PrecomputedEmbeddings:
    """All precomputed embeddings for one encoder variant (deterministic)."""
    prot_dim: int
    lig_dim: int
    # Stage 1
    s1_prot_embs: torch.Tensor
    s1_lig_embs: torch.Tensor
    s1_labels: torch.Tensor
    # Stage 2
    s2_prot_embs: torch.Tensor
    s2_lig_embs: torch.Tensor
    s2_labels: torch.Tensor
    # Stage 3 positives
    s3_pos_prot_embs: torch.Tensor
    s3_pos_lig_embs: torch.Tensor
    pos_names: List[str]
    # Stage 3 negatives
    s3_neg_prot_embs: torch.Tensor
    s3_neg_lig_embs: torch.Tensor
    # Stage 3 augmented
    augmented_lig_embs: Dict[int, torch.Tensor]
    # Stage 3 supplementary
    s3_supp_pos_prot_embs: Optional[torch.Tensor]
    s3_supp_pos_lig_embs: Optional[torch.Tensor]
    s3_gli_neg_prot_embs: Optional[torch.Tensor]
    s3_gli_neg_lig_embs: Optional[torch.Tensor]
    # GLI1 sequence (for truncation analysis)
    gli1_seq: str
    # Morgan fingerprints (Phase 2C)
    pos_morgan_fps: Optional[torch.Tensor] = None
    neg_morgan_fps: Optional[torch.Tensor] = None
    augmented_morgan_fps: Optional[Dict[int, torch.Tensor]] = None
    supp_pos_morgan_fps: Optional[torch.Tensor] = None
    gli_neg_morgan_fps: Optional[torch.Tensor] = None
    # Compound SMILES for Tanimoto analysis (Phase 2A)
    pos_smiles: Optional[List[str]] = None


def precompute_embeddings(encoder: EncoderWrapper, smiles_list: List[str],
                          protein_seqs: List[str], cache: EmbeddingCache,
                          stage_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute and cache all embeddings for a dataset.

    This runs the frozen encoders once and caches results to disk,
    so training only uses the lightweight prediction head.
    """
    logging.info(f"  Pre-computing embeddings for {stage_name}...")

    # Deduplicate proteins (many pairs share the same protein)
    unique_prots = list(set(protein_seqs))
    prot_to_emb = {}

    logging.info(f"    Encoding {len(unique_prots)} unique proteins...")
    for i, seq in enumerate(unique_prots):
        cached = cache.get(seq, encoder.prot_model_name)
        if cached is not None:
            prot_to_emb[seq] = cached
        else:
            emb = encoder.encode_protein(seq)
            cache.put(seq, encoder.prot_model_name, emb)
            prot_to_emb[seq] = emb
        if (i + 1) % 100 == 0 or i == len(unique_prots) - 1:
            logging.info(f"    Proteins: {i + 1}/{len(unique_prots)}")

    # Deduplicate ligands
    unique_smiles = list(set(smiles_list))
    smi_to_emb = {}

    logging.info(f"    Encoding {len(unique_smiles)} unique ligands...")
    for i, smi in enumerate(unique_smiles):
        cached = cache.get(smi, encoder.mol_model_name)
        if cached is not None:
            smi_to_emb[smi] = cached
        else:
            emb = encoder.encode_ligand(smi)
            cache.put(smi, encoder.mol_model_name, emb)
            smi_to_emb[smi] = emb
        if (i + 1) % 500 == 0 or i == len(unique_smiles) - 1:
            logging.info(f"    Ligands: {i + 1}/{len(unique_smiles)}")

    # Assemble in order
    prot_embs = torch.stack([prot_to_emb[s] for s in protein_seqs])
    lig_embs = torch.stack([smi_to_emb[s] for s in smiles_list])

    logging.info(f"    Done: prot_embs={prot_embs.shape}, lig_embs={lig_embs.shape}")
    return prot_embs, lig_embs


def precompute_all_embeddings(config: Config, device: torch.device) -> PrecomputedEmbeddings:
    """Precompute ALL embeddings for one encoder variant.

    Embeddings are deterministic (frozen encoders) and cached to disk,
    so this only needs to run once per encoder variant regardless of seed count.
    """
    variant = "ESM-2" if config.use_esm2 else "ProtBERT"
    logging.info(f"\n{'#'*60}")
    logging.info(f"# PRECOMPUTING EMBEDDINGS: {variant}")
    logging.info(f"{'#'*60}")

    cache = EmbeddingCache()

    with Timer("Encoder loading"):
        encoder = EncoderWrapper(config, device)

    prot_dim = encoder.prot_dim
    lig_dim = encoder.mol_dim

    # --- Stage 1: BindingDB ---
    logging.info("\n=== Loading Stage 1 data (BindingDB) ===")
    bindingdb_df = load_bindingdb(config)
    with Timer("Stage 1 embedding"):
        s1_prot_embs, s1_lig_embs = precompute_embeddings(
            encoder,
            bindingdb_df["ligand_smiles"].tolist(),
            bindingdb_df["protein_sequence"].tolist(),
            cache, "stage1_bindingdb"
        )
    s1_labels = torch.tensor(bindingdb_df["label"].values, dtype=torch.float32)

    # --- Stage 2: ZF ---
    logging.info("\n=== Loading Stage 2 data (ZF) ===")
    zf_df = load_zf_data(config)
    with Timer("Stage 2 embedding"):
        s2_prot_embs, s2_lig_embs = precompute_embeddings(
            encoder,
            zf_df["smiles"].tolist(),
            zf_df["protein_sequence"].tolist(),
            cache, "stage2_zf"
        )
    s2_labels = torch.tensor(zf_df["label"].values, dtype=torch.float32)

    # --- Stage 3: GLI ---
    logging.info("\n=== Loading Stage 3 data (GLI) ===")
    gli_pos_df, gli_neg_df, chembl_gli_pos, chembl_gli_neg, gli1_seq = load_gli_data(config)

    pos_smiles = gli_pos_df["smiles"].tolist()
    name_col = next((c for c in gli_pos_df.columns if "name" in c.lower() or "id" in c.lower()), gli_pos_df.columns[0])
    pos_names = gli_pos_df[name_col].tolist()
    pos_prots = [gli1_seq] * len(pos_smiles)

    with Timer("Stage 3 positive embedding"):
        s3_pos_prot_embs, s3_pos_lig_embs = precompute_embeddings(
            encoder, pos_smiles, pos_prots, cache, "stage3_positives"
        )

    # Supplementary positives
    s3_supp_pos_prot_embs = None
    s3_supp_pos_lig_embs = None
    if len(chembl_gli_pos) > 0:
        supp_pos_smiles = chembl_gli_pos["smiles"].tolist()
        supp_pos_prots = [gli1_seq] * len(supp_pos_smiles)
        with Timer("Stage 3 supplementary positive embedding"):
            s3_supp_pos_prot_embs, s3_supp_pos_lig_embs = precompute_embeddings(
                encoder, supp_pos_smiles, supp_pos_prots, cache, "stage3_supp_positives"
            )
        logging.info(f"  ChEMBL GLI supplementary positives: {len(supp_pos_smiles)} compounds")

    # True GLI negatives
    s3_gli_neg_prot_embs = None
    s3_gli_neg_lig_embs = None
    if len(chembl_gli_neg) > 0:
        gli_neg_smiles = chembl_gli_neg["smiles"].tolist()
        gli_neg_prots = [gli1_seq] * len(gli_neg_smiles)
        with Timer("Stage 3 GLI-tested negative embedding"):
            s3_gli_neg_prot_embs, s3_gli_neg_lig_embs = precompute_embeddings(
                encoder, gli_neg_smiles, gli_neg_prots, cache, "stage3_gli_neg"
            )
        logging.info(f"  ChEMBL GLI true negatives: {len(gli_neg_smiles)} compounds")

    # Augmented positive ligand embeddings (with asymmetric augmentation, Phase 2D)
    logging.info("  Generating SMILES augmentations for positives...")
    augmented_lig_embs = {}
    augmented_morgan_fps_dict = {}
    cfg_ft = config.gli_finetune
    use_morgan = config.head.use_morgan_fp

    if cfg_ft.use_asymmetric_aug:
        aug_counts = compute_asymmetric_aug_counts(
            pos_smiles, base_aug=cfg_ft.smiles_augment_per_positive,
            min_aug=cfg_ft.asym_aug_min, max_aug=cfg_ft.asym_aug_max
        )
        logging.info(f"  Asymmetric augmentation: {dict(zip(pos_names, aug_counts))}")
    else:
        aug_counts = [cfg_ft.smiles_augment_per_positive] * len(pos_smiles)

    for i, smi in enumerate(pos_smiles):
        n_aug = aug_counts[i]
        aug_smiles = randomize_smiles(smi, n_augments=n_aug)
        aug_embs = []
        aug_mfps = []
        for asmi in aug_smiles:
            cached = cache.get(asmi, encoder.mol_model_name)
            if cached is not None:
                aug_embs.append(cached)
            else:
                emb = encoder.encode_ligand(asmi)
                cache.put(asmi, encoder.mol_model_name, emb)
                aug_embs.append(emb)
            if use_morgan:
                aug_mfps.append(torch.tensor(compute_morgan_fp(
                    asmi, config.head.morgan_fp_bits, config.head.morgan_fp_radius
                )))
        augmented_lig_embs[i] = torch.stack(aug_embs)
        if use_morgan:
            augmented_morgan_fps_dict[i] = torch.stack(aug_mfps)
        logging.info(f"    {pos_names[i]}: {n_aug} augmented SMILES encoded")

    # Negative embeddings
    neg_smiles = gli_neg_df["smiles"].tolist()
    neg_prots = [gli1_seq] * len(neg_smiles)
    with Timer("Stage 3 negative embedding"):
        s3_neg_prot_embs, s3_neg_lig_embs = precompute_embeddings(
            encoder, neg_smiles, neg_prots, cache, "stage3_negatives"
        )

    # Offload encoders to free GPU
    encoder.offload()

    # --- Morgan fingerprints (Phase 2C) ---
    pos_morgan_fps = None
    neg_morgan_fps = None
    supp_pos_mfps = None
    gli_neg_mfps = None

    if use_morgan:
        logging.info("  Computing Morgan fingerprints (Phase 2C)...")
        pos_morgan_fps = compute_morgan_fps_batch(
            pos_smiles, config.head.morgan_fp_bits, config.head.morgan_fp_radius
        )
        neg_morgan_fps = compute_morgan_fps_batch(
            neg_smiles, config.head.morgan_fp_bits, config.head.morgan_fp_radius
        )
        if len(chembl_gli_pos) > 0:
            supp_pos_mfps = compute_morgan_fps_batch(
                chembl_gli_pos["smiles"].tolist(),
                config.head.morgan_fp_bits, config.head.morgan_fp_radius
            )
        if len(chembl_gli_neg) > 0:
            gli_neg_mfps = compute_morgan_fps_batch(
                chembl_gli_neg["smiles"].tolist(),
                config.head.morgan_fp_bits, config.head.morgan_fp_radius
            )
        logging.info(f"    Morgan FPs: pos={pos_morgan_fps.shape}, neg={neg_morgan_fps.shape}")

    return PrecomputedEmbeddings(
        prot_dim=prot_dim, lig_dim=lig_dim,
        s1_prot_embs=s1_prot_embs, s1_lig_embs=s1_lig_embs, s1_labels=s1_labels,
        s2_prot_embs=s2_prot_embs, s2_lig_embs=s2_lig_embs, s2_labels=s2_labels,
        s3_pos_prot_embs=s3_pos_prot_embs, s3_pos_lig_embs=s3_pos_lig_embs,
        pos_names=pos_names,
        s3_neg_prot_embs=s3_neg_prot_embs, s3_neg_lig_embs=s3_neg_lig_embs,
        augmented_lig_embs=augmented_lig_embs,
        s3_supp_pos_prot_embs=s3_supp_pos_prot_embs,
        s3_supp_pos_lig_embs=s3_supp_pos_lig_embs,
        s3_gli_neg_prot_embs=s3_gli_neg_prot_embs,
        s3_gli_neg_lig_embs=s3_gli_neg_lig_embs,
        gli1_seq=gli1_seq,
        pos_morgan_fps=pos_morgan_fps,
        neg_morgan_fps=neg_morgan_fps,
        augmented_morgan_fps=augmented_morgan_fps_dict if use_morgan else None,
        supp_pos_morgan_fps=supp_pos_mfps,
        gli_neg_morgan_fps=gli_neg_mfps,
        pos_smiles=pos_smiles,
    )


def run_training_pipeline(config: Config, device: torch.device,
                          embs: PrecomputedEmbeddings, seed: int,
                          experiment_name: str) -> Dict:
    """Run the 3-stage training pipeline for one seed using precomputed embeddings."""
    logging.info(f"\n{'#'*60}")
    logging.info(f"# TRAINING: {experiment_name} (seed={seed})")
    logging.info(f"{'#'*60}")

    # Set seed for this run
    config.seed = seed
    set_seed(seed)

    results = {}

    # Build datasets from precomputed embeddings
    s1_dataset = EmbeddedDataset(embs.s1_prot_embs, embs.s1_lig_embs, embs.s1_labels)
    s2_dataset = EmbeddedDataset(embs.s2_prot_embs, embs.s2_lig_embs, embs.s2_labels)

    # Build and train prediction head
    model = BranchingPredictionHead(embs.prot_dim, embs.lig_dim, config.head)
    logging.info(f"Prediction head: {model.count_parameters():,} trainable parameters")

    # --- Stage 1: Pretrain ---
    with Timer("Stage 1 training"):
        s1_result = run_stage1_pretrain(model, s1_dataset, config, device)
    results["stage1"] = s1_result

    # --- Stage 2: Domain adapt ---
    with Timer("Stage 2 training"):
        s2_result = run_stage2_domain_adapt(model, s2_dataset, config, device)
    results["stage2"] = s2_result

    # --- Stage 3: LOOCV ---
    model_variant = "esm2" if config.use_esm2 else "protbert"
    with Timer("Stage 3 LOOCV"):
        s3_result = run_loocv(
            model_template=model,
            stage2_checkpoint=s2_result["checkpoint_path"],
            positive_prot_embs=embs.s3_pos_prot_embs,
            positive_lig_embs=embs.s3_pos_lig_embs,
            positive_names=embs.pos_names,
            negative_prot_embs=embs.s3_neg_prot_embs,
            negative_lig_embs=embs.s3_neg_lig_embs,
            augmented_positive_lig_embs=embs.augmented_lig_embs,
            config=config,
            device=device,
            supp_pos_prot_embs=embs.s3_supp_pos_prot_embs,
            supp_pos_lig_embs=embs.s3_supp_pos_lig_embs,
            gli_neg_prot_embs=embs.s3_gli_neg_prot_embs,
            gli_neg_lig_embs=embs.s3_gli_neg_lig_embs,
            model_variant=f"{model_variant}_seed{seed}",
            positive_morgan_fps=embs.pos_morgan_fps,
            negative_morgan_fps=embs.neg_morgan_fps,
            augmented_morgan_fps=embs.augmented_morgan_fps,
            supp_pos_morgan_fps=embs.supp_pos_morgan_fps,
            gli_neg_morgan_fps=embs.gli_neg_morgan_fps,
        )
    results["stage3"] = s3_result

    return results


def parse_seeds(seeds_arg: str) -> List[int]:
    """Parse --seeds argument into list of integers."""
    if seeds_arg == "all":
        return MULTI_SEEDS
    return [int(s.strip()) for s in seeds_arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description="GLI-PLAPT Pipeline")
    parser.add_argument("--mode", choices=["both", "modified", "baseline"],
                        default="both", help="Which model(s) to run")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Seeds: 'all' for 5 seeds, or comma-separated (e.g. '42,123,456')")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1

    setup_dirs()
    set_seed(seeds[0])
    logger = setup_logging("gli_plapt")
    device = get_device()
    log_environment()

    logging.info(f"Seeds: {seeds} ({'multi-seed' if multi_seed else 'single-seed'})")

    all_results = {"seeds": seeds, "multi_seed": multi_seed}

    # =========================================================================
    # Step 0: Precompute embeddings (once per encoder, deterministic)
    # =========================================================================
    embs_mod = None
    embs_base = None

    if args.mode in ("both", "modified"):
        config_mod = Config(use_esm2=True)
        log_config(config_mod, "modified_esm2")
        with Timer("Modified embeddings (ESM-2)"):
            embs_mod = precompute_all_embeddings(config_mod, device)

    if args.mode in ("both", "baseline"):
        config_base = Config(use_esm2=False)
        log_config(config_base, "baseline_protbert")
        with Timer("Baseline embeddings (ProtBERT)"):
            embs_base = precompute_all_embeddings(config_base, device)

    # =========================================================================
    # Step 1: Tanimoto similarity analysis (Phase 2A — diagnostic)
    # =========================================================================
    # Run on first available embeddings (compound SMILES are the same for both encoders)
    tanimoto_embs = embs_mod or embs_base
    if tanimoto_embs is not None and tanimoto_embs.pos_smiles is not None:
        tanimoto_result = tanimoto_similarity_analysis(
            tanimoto_embs.pos_names, tanimoto_embs.pos_smiles
        )
        all_results["tanimoto_analysis"] = tanimoto_result

    # =========================================================================
    # Step 2: Train across all seeds
    # =========================================================================
    mod_seed_results = {}  # {seed: pipeline_result}
    base_seed_results = {}

    for seed in seeds:
        logging.info(f"\n{'*'*60}")
        logging.info(f"* SEED: {seed}")
        logging.info(f"{'*'*60}")

        if embs_mod is not None:
            config_mod = Config(use_esm2=True, seed=seed)
            with Timer(f"Modified pipeline seed={seed}"):
                mod_seed_results[seed] = run_training_pipeline(
                    config_mod, device, embs_mod, seed, f"GLI-PLAPT (ESM-2) seed={seed}"
                )

        if embs_base is not None:
            config_base = Config(use_esm2=False, seed=seed)
            with Timer(f"Baseline pipeline seed={seed}"):
                base_seed_results[seed] = run_training_pipeline(
                    config_base, device, embs_base, seed, f"Baseline (ProtBERT) seed={seed}"
                )

    # =========================================================================
    # Step 3: Multi-seed aggregation (Phase 1A)
    # =========================================================================
    if multi_seed:
        if mod_seed_results:
            all_results["esm2_multi_seed"] = aggregate_multi_seed_results(
                mod_seed_results, "ESM-2"
            )
        if base_seed_results:
            all_results["protbert_multi_seed"] = aggregate_multi_seed_results(
                base_seed_results, "ProtBERT"
            )

    # =========================================================================
    # Step 4: Per-seed statistical comparison (use first seed for backward compat)
    # =========================================================================
    primary_seed = seeds[0]
    if args.mode == "both" and primary_seed in mod_seed_results and primary_seed in base_seed_results:
        comparison = compare_models(
            base_seed_results[primary_seed]["stage3"],
            mod_seed_results[primary_seed]["stage3"],
        )
        all_results["comparison_primary_seed"] = {
            "seed": primary_seed,
            "mcnemar_p": comparison["mcnemar_p"],
            "ttest_p": comparison["ttest_p"],
            "bootstrap_ci": list(comparison["bootstrap_ci"]),
            "baseline_hit_rate": comparison["baseline_hit_rate"],
            "modified_hit_rate": comparison["modified_hit_rate"],
        }

    # =========================================================================
    # Step 5: Cross-encoder consensus analysis (Phase 1C)
    # =========================================================================
    if args.mode == "both" and mod_seed_results and base_seed_results:
        logging.info("\n" + "="*60)
        logging.info("PHASE 1C: CROSS-ENCODER CONSENSUS ANALYSIS")
        logging.info("="*60)

        # Run consensus on each seed
        consensus_per_seed = {}
        for seed in seeds:
            if seed in mod_seed_results and seed in base_seed_results:
                cons = consensus_analysis(
                    base_seed_results[seed]["stage3"],
                    mod_seed_results[seed]["stage3"],
                    threshold=0.5,
                )
                consensus_per_seed[seed] = cons

        # Aggregate consensus across seeds
        if consensus_per_seed:
            ensemble_hrs = [c["hit_rate_ensemble"] for c in consensus_per_seed.values()]
            conservative_hrs = [c["hit_rate_conservative"] for c in consensus_per_seed.values()]
            optimistic_hrs = [c["hit_rate_optimistic"] for c in consensus_per_seed.values()]

            all_results["consensus"] = {
                "per_seed": {str(k): v for k, v in consensus_per_seed.items()},
                "ensemble_hit_rate_mean": float(np.mean(ensemble_hrs)),
                "ensemble_hit_rate_std": float(np.std(ensemble_hrs)),
                "conservative_hit_rate_mean": float(np.mean(conservative_hrs)),
                "conservative_hit_rate_std": float(np.std(conservative_hrs)),
                "optimistic_hit_rate_mean": float(np.mean(optimistic_hrs)),
                "optimistic_hit_rate_std": float(np.std(optimistic_hrs)),
            }

            if multi_seed:
                logging.info(f"\n  --- Consensus Aggregated ({len(seeds)} seeds) ---")
                logging.info(f"  Ensemble hit rate:      {np.mean(ensemble_hrs):.2%} ± {np.std(ensemble_hrs):.2%}")
                logging.info(f"  Conservative hit rate:  {np.mean(conservative_hrs):.2%} ± {np.std(conservative_hrs):.2%}")
                logging.info(f"  Optimistic hit rate:    {np.mean(optimistic_hrs):.2%} ± {np.std(optimistic_hrs):.2%}")

    # =========================================================================
    # Step 6: ESM-2 truncation analysis (Phase 1D)
    # =========================================================================
    if embs_mod is not None:
        logging.info("\n" + "="*60)
        logging.info("PHASE 1D: ESM-2 TRUNCATION ANALYSIS")
        logging.info("="*60)

        trunc = esm2_truncation_analysis(
            embs_mod.gli1_seq,
            esm2_max_length=Config().encoder.esm2_max_length,
        )
        all_results["esm2_truncation"] = trunc

    # =========================================================================
    # Save all results
    # =========================================================================
    # Legacy results_summary.json (backward compatible)
    summary = {}
    for variant, seed_results in [("modified", mod_seed_results), ("baseline", base_seed_results)]:
        if primary_seed in seed_results:
            s3 = seed_results[primary_seed]["stage3"]
            summary[variant] = {
                "hit_rate": s3["hit_rate"],
                "hit_rate_calibrated": s3.get("hit_rate_calibrated", s3["hit_rate"]),
                "mean_prob": s3["mean_prob"],
                "mean_uncertainty": s3["mean_uncertainty"],
                "mean_fpr_default": s3.get("mean_fpr_default", None),
                "mean_fpr_calibrated": s3.get("mean_fpr_calibrated", None),
                "mean_optimal_threshold": s3.get("mean_optimal_threshold", 0.5),
                "per_compound": s3["per_compound"],
            }
    if "comparison_primary_seed" in all_results:
        summary["comparison"] = all_results["comparison_primary_seed"]

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"\nResults summary saved: {summary_path}")

    # Phase 1 comprehensive results
    save_all_results(all_results)

    logging.info("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
