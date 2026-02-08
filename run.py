"""
GLI-PLAPT Main Pipeline — Runs the complete 3-stage training + LOOCV evaluation.

Usage:
    python run.py --mode both          # Run baseline + modified + comparison
    python run.py --mode modified      # Run only modified (ESM-2) model
    python run.py --mode baseline      # Run only baseline (ProtBERT) model
"""

import argparse
import logging
import os
import json
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.config import Config, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, NEGATIVE_SOURCES_KEEP
from src.data import (
    load_bindingdb, load_zf_data, load_gli_data,
    EmbeddedDataset, ProteinLigandDataset, randomize_smiles,
    EmbeddingCache,
)
from src.model import BranchingPredictionHead, EncoderWrapper
from src.trainer import run_stage1_pretrain, run_stage2_domain_adapt
from src.evaluate import run_loocv, evaluate_on_negatives, compare_models
from src.utils import (
    set_seed, setup_dirs, setup_logging, get_device,
    log_config, log_environment, Timer,
)


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


def run_pipeline(config: Config, device: torch.device, experiment_name: str) -> Dict:
    """Run the complete 3-stage pipeline for one model variant."""
    logging.info(f"\n{'#'*60}")
    logging.info(f"# PIPELINE: {experiment_name}")
    logging.info(f"# Protein encoder: {'ESM-2 650M' if config.use_esm2 else 'ProtBERT'}")
    logging.info(f"{'#'*60}")

    cache = EmbeddingCache()
    results = {}

    # -------------------------------------------------------------------------
    # Load encoder and pre-compute embeddings
    # -------------------------------------------------------------------------
    with Timer("Encoder loading"):
        encoder = EncoderWrapper(config, device)

    prot_dim = encoder.prot_dim
    lig_dim = encoder.mol_dim

    # --- Stage 1: BindingDB embeddings ---
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
    s1_dataset = EmbeddedDataset(s1_prot_embs, s1_lig_embs, s1_labels)

    # --- Stage 2: ZF embeddings ---
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
    s2_dataset = EmbeddedDataset(s2_prot_embs, s2_lig_embs, s2_labels)

    # --- Stage 3: GLI embeddings ---
    logging.info("\n=== Loading Stage 3 data (GLI) ===")
    gli_pos_df, gli_neg_df, chembl_gli_pos, chembl_gli_neg, gli1_seq = load_gli_data(config)

    # Positive embeddings (canonical)
    pos_smiles = gli_pos_df["smiles"].tolist()
    # gli_inhibitors.csv uses "compound_name" as first column
    name_col = next((c for c in gli_pos_df.columns if "name" in c.lower() or "id" in c.lower()), gli_pos_df.columns[0])
    pos_names = gli_pos_df[name_col].tolist()
    pos_prots = [gli1_seq] * len(pos_smiles)

    with Timer("Stage 3 positive embedding"):
        s3_pos_prot_embs, s3_pos_lig_embs = precompute_embeddings(
            encoder, pos_smiles, pos_prots, cache, "stage3_positives"
        )

    # ChEMBL GLI supplementary positives (always in training, never held out)
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

    # ChEMBL GLI true negatives (tested against GLI, confirmed non-binders)
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

    # Augmented positive ligand embeddings
    logging.info("  Generating SMILES augmentations for positives...")
    augmented_lig_embs = {}
    n_aug = config.gli_finetune.smiles_augment_per_positive
    for i, smi in enumerate(pos_smiles):
        aug_smiles = randomize_smiles(smi, n_augments=n_aug)
        aug_embs = []
        for asmi in aug_smiles:
            cached = cache.get(asmi, encoder.mol_model_name)
            if cached is not None:
                aug_embs.append(cached)
            else:
                emb = encoder.encode_ligand(asmi)
                cache.put(asmi, encoder.mol_model_name, emb)
                aug_embs.append(emb)
        augmented_lig_embs[i] = torch.stack(aug_embs)
        logging.info(f"    {pos_names[i]}: {len(aug_embs)} augmented SMILES encoded")

    # Negative embeddings (SMO + structural)
    neg_smiles = gli_neg_df["smiles"].tolist()
    neg_prots = [gli1_seq] * len(neg_smiles)

    with Timer("Stage 3 negative embedding"):
        s3_neg_prot_embs, s3_neg_lig_embs = precompute_embeddings(
            encoder, neg_smiles, neg_prots, cache, "stage3_negatives"
        )

    # Offload encoders to free GPU
    encoder.offload()

    # -------------------------------------------------------------------------
    # Build and train prediction head
    # -------------------------------------------------------------------------
    model = BranchingPredictionHead(prot_dim, lig_dim, config.head)
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
    with Timer("Stage 3 LOOCV"):
        s3_result = run_loocv(
            model_template=model,
            stage2_checkpoint=s2_result["checkpoint_path"],
            positive_prot_embs=s3_pos_prot_embs,
            positive_lig_embs=s3_pos_lig_embs,
            positive_names=pos_names,
            negative_prot_embs=s3_neg_prot_embs,
            negative_lig_embs=s3_neg_lig_embs,
            augmented_positive_lig_embs=augmented_lig_embs,
            config=config,
            device=device,
            supp_pos_prot_embs=s3_supp_pos_prot_embs,
            supp_pos_lig_embs=s3_supp_pos_lig_embs,
            gli_neg_prot_embs=s3_gli_neg_prot_embs,
            gli_neg_lig_embs=s3_gli_neg_lig_embs,
            model_variant="esm2" if config.use_esm2 else "protbert",
        )
    results["stage3"] = s3_result

    # Evaluate final model on negatives
    neg_eval = evaluate_on_negatives(
        model, s3_neg_prot_embs, s3_neg_lig_embs, config, device
    )
    results["negative_eval"] = neg_eval

    return results


def main():
    parser = argparse.ArgumentParser(description="GLI-PLAPT Pipeline")
    parser.add_argument("--mode", choices=["both", "modified", "baseline"],
                        default="both", help="Which model(s) to run")
    args = parser.parse_args()

    setup_dirs()
    set_seed()
    logger = setup_logging("gli_plapt")
    device = get_device()
    log_environment()

    all_results = {}

    if args.mode in ("both", "modified"):
        config_mod = Config(use_esm2=True)
        log_config(config_mod, "modified_esm2")
        with Timer("Modified pipeline (ESM-2)"):
            all_results["modified"] = run_pipeline(config_mod, device, "GLI-PLAPT (ESM-2)")

    if args.mode in ("both", "baseline"):
        config_base = Config(use_esm2=False)
        log_config(config_base, "baseline_protbert")
        with Timer("Baseline pipeline (ProtBERT)"):
            all_results["baseline"] = run_pipeline(config_base, device, "Baseline PLAPT (ProtBERT)")

    if args.mode == "both" and "modified" in all_results and "baseline" in all_results:
        comparison = compare_models(
            all_results["baseline"]["stage3"],
            all_results["modified"]["stage3"],
        )
        all_results["comparison"] = comparison

    # Save final results summary
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    summary = {}
    for key in all_results:
        if key == "comparison":
            summary["comparison"] = {
                "mcnemar_p": all_results["comparison"]["mcnemar_p"],
                "ttest_p": all_results["comparison"]["ttest_p"],
                "bootstrap_ci": list(all_results["comparison"]["bootstrap_ci"]),
                "baseline_hit_rate": all_results["comparison"]["baseline_hit_rate"],
                "modified_hit_rate": all_results["comparison"]["modified_hit_rate"],
            }
        elif "stage3" in all_results[key]:
            s3 = all_results[key]["stage3"]
            summary[key] = {
                "hit_rate": s3["hit_rate"],
                "mean_prob": s3["mean_prob"],
                "mean_uncertainty": s3["mean_uncertainty"],
                "per_compound": s3["per_compound"],
            }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"\nResults summary saved: {summary_path}")
    logging.info("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
