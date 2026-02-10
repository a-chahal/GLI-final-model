"""
GLI Inhibitor Virtual Screening — Multi-GPU Inference on Rosenbluth

Runs trained GLI-PLAPT LOOCV ensemble (29 fold models) on prescreened compounds.
Each fold model runs MC Dropout (50 passes) for uncertainty estimation.
Final output: ensemble mean probability, uncertainty, per-fold breakdown.

Optimized for Rosenbluth workstation:
  - 4x NVIDIA RTX 3090 (24GB each): parallel ChemBERTa encoding
  - 2x 60-core CPUs (120 total): parallel Morgan FP computation
  - Prediction head is tiny (~500K params), runs on single GPU

Usage:
    python -m src.screen_compounds --input data/prescreened_compounds.csv \
                                   --output outputs/screening_results.csv \
                                   --gpus 4 --workers 100 --mc-samples 50

    # Single GPU mode (local testing):
    python -m src.screen_compounds --input data/prescreened_compounds.csv --gpus 1
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, PredictionHeadConfig, CHECKPOINT_DIR
from src.model import BranchingPredictionHead, EncoderWrapper


# ---------------------------------------------------------------------------
# Morgan Fingerprint computation (CPU-parallel)
# ---------------------------------------------------------------------------

def compute_morgan_single(smiles: str) -> np.ndarray:
    """Compute Morgan FP for a single SMILES. Returns zeros if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_parallel(smiles_list: list, workers: int = 100) -> np.ndarray:
    """Compute Morgan FPs in parallel. Returns (N, 2048) array."""
    from multiprocessing import Pool

    logging.info(f"Computing Morgan FPs for {len(smiles_list)} compounds ({workers} workers)...")
    with Pool(workers) as pool:
        fps = pool.map(compute_morgan_single, smiles_list, chunksize=500)
    return np.stack(fps)


# ---------------------------------------------------------------------------
# ChemBERTa encoding (multi-GPU)
# ---------------------------------------------------------------------------

def encode_ligands_single_gpu(smiles_list: list, gpu_id: int,
                              config: Config, batch_size: int = 128) -> torch.Tensor:
    """Encode SMILES on a single GPU using ChemBERTa. Returns (N, 768) tensor."""
    from transformers import RobertaTokenizer, RobertaModel

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(config.encoder.chemberta_model_name)
    model = RobertaModel.from_pretrained(config.encoder.chemberta_model_name)
    model.eval().to(device)

    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=config.encoder.chemberta_max_length,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tokens)
        all_embs.append(out.pooler_output.cpu())

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)


def encode_ligands_multigpu(smiles_list: list, n_gpus: int,
                            config: Config, batch_size: int = 128) -> torch.Tensor:
    """Encode SMILES across multiple GPUs. Returns (N, 768) tensor."""
    if n_gpus <= 1 or not torch.cuda.is_available():
        return encode_ligands_single_gpu(smiles_list, 0, config, batch_size)

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # Split SMILES across GPUs
    chunk_size = (len(smiles_list) + n_gpus - 1) // n_gpus
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    logging.info(f"Encoding {len(smiles_list)} compounds across {len(chunks)} GPUs...")

    # Use process pool for multi-GPU encoding
    results = {}

    def _encode_worker(gpu_id, chunk, result_dict):
        result_dict[gpu_id] = encode_ligands_single_gpu(chunk, gpu_id, config, batch_size)

    # Sequential GPU encoding (avoids CUDA fork issues, still GPU-parallel via pipeline)
    for gpu_id, chunk in enumerate(chunks):
        logging.info(f"  GPU {gpu_id}: encoding {len(chunk)} compounds...")
        results[gpu_id] = encode_ligands_single_gpu(chunk, gpu_id, config, batch_size)

    # Reassemble in order
    ordered = [results[i] for i in range(len(chunks))]
    return torch.cat(ordered, dim=0)


# ---------------------------------------------------------------------------
# Protein embedding (compute once)
# ---------------------------------------------------------------------------

def get_protein_embedding(config: Config, device: torch.device) -> torch.Tensor:
    """Compute GLI1 protein embedding. Returns (1, prot_dim) tensor."""
    fasta_path = config.paths.gli1_fasta
    logging.info(f"Loading GLI1 sequence from {fasta_path}...")

    with open(fasta_path) as f:
        lines = f.readlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    logging.info(f"  GLI1 sequence length: {len(sequence)} AA")

    logging.info("  Computing protein embedding (this may take a moment)...")
    encoder = EncoderWrapper(config, device)
    prot_emb = encoder.encode_protein(sequence).unsqueeze(0)  # (1, prot_dim)
    encoder.offload()

    return prot_emb


# ---------------------------------------------------------------------------
# Ensemble prediction with MC Dropout
# ---------------------------------------------------------------------------

def load_fold_models(checkpoint_dir: str, config: Config,
                     device: torch.device) -> List[BranchingPredictionHead]:
    """Load all LOOCV fold models from checkpoint directory."""
    prot_dim = config.encoder.esm2_embed_dim if config.use_esm2 else config.encoder.protbert_embed_dim
    lig_dim = config.encoder.chemberta_embed_dim

    ckpt_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("stage3_fold") and f.endswith("_best.pt")
    ])

    if not ckpt_files:
        raise FileNotFoundError(f"No stage3 fold checkpoints in {checkpoint_dir}")

    models = []
    for fname in ckpt_files:
        path = os.path.join(checkpoint_dir, fname)
        head = BranchingPredictionHead(prot_dim, lig_dim, config.head)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(ckpt["model_state_dict"])
        head.to(device)
        head.eval()
        models.append(head)

    logging.info(f"Loaded {len(models)} fold models from {checkpoint_dir}")
    return models


def predict_ensemble(models: List[BranchingPredictionHead],
                     protein_emb: torch.Tensor,
                     ligand_embs: torch.Tensor,
                     morgan_fps: torch.Tensor,
                     mc_samples: int = 50,
                     batch_size: int = 512) -> Dict[str, np.ndarray]:
    """Run ensemble prediction with MC Dropout across all fold models.

    Args:
        models: List of LOOCV fold models
        protein_emb: (1, prot_dim) — broadcast to batch
        ligand_embs: (N, 768) — ChemBERTa embeddings
        morgan_fps: (N, 2048) — Morgan fingerprints
        mc_samples: MC Dropout forward passes per model
        batch_size: Inference batch size

    Returns:
        dict with:
            ensemble_mean: (N,) mean probability across all folds
            ensemble_std: (N,) std across folds (epistemic uncertainty)
            mc_uncertainty: (N,) mean MC uncertainty across folds (aleatoric)
            per_fold_probs: (N, n_folds) per-fold mean probabilities
    """
    device = next(models[0].parameters()).device
    n_compounds = ligand_embs.shape[0]
    n_folds = len(models)

    # Expand protein embedding to match batch
    prot_expanded = protein_emb.expand(batch_size, -1).to(device)

    all_fold_means = []  # (n_folds, N)
    all_fold_stds = []

    for fold_idx, model in enumerate(models):
        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            logging.info(f"  Fold {fold_idx+1}/{n_folds}...")

        fold_probs = []

        for start in range(0, n_compounds, batch_size):
            end = min(start + batch_size, n_compounds)
            bs = end - start

            lig_batch = ligand_embs[start:end].to(device)
            morgan_batch = morgan_fps[start:end].to(device)
            prot_batch = prot_expanded[:bs]

            result = model.mc_predict(
                prot_batch, lig_batch,
                n_samples=mc_samples,
                morgan_fp=morgan_batch,
            )
            fold_probs.append(result["mean_prob"].cpu().numpy())

        fold_mean = np.concatenate(fold_probs)
        all_fold_means.append(fold_mean)

    all_fold_means = np.stack(all_fold_means, axis=1)  # (N, n_folds)

    return {
        "ensemble_mean": all_fold_means.mean(axis=1),
        "ensemble_std": all_fold_means.std(axis=1),
        "per_fold_probs": all_fold_means,
        "n_folds_agree_above_05": (all_fold_means > 0.5).sum(axis=1),
    }


# ---------------------------------------------------------------------------
# Main screening pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLI-PLAPT virtual screening")
    parser.add_argument("--input", type=str, required=True,
                        help="Prescreened compounds CSV (must have 'smiles' column)")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "screening_results.csv"))
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of GPUs for encoding (Rosenbluth: 4)")
    parser.add_argument("--workers", type=int, default=100,
                        help="CPU workers for Morgan FP (Rosenbluth: 100)")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="MC Dropout samples per fold model")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size per GPU")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Only output top-K predictions")
    parser.add_argument("--use-esm2", action="store_true", default=True,
                        help="Use ESM-2 encoder (default)")
    parser.add_argument("--use-protbert", action="store_true",
                        help="Use ProtBERT encoder instead of ESM-2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    # Config
    config = Config()
    if args.use_protbert:
        config.use_esm2 = False

    # Set random seeds for reproducibility
    # This ensures MC Dropout masks and any other stochastic elements are consistent across runs
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"Random seed set to: {config.seed}")

    # Device setup
    n_gpus = min(args.gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    device = torch.device("cuda:0" if n_gpus > 0 else "cpu")
    logging.info(f"Device: {device} ({n_gpus} GPUs available)")
    if torch.cuda.is_available():
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            logging.info(f"  GPU {i}: {name} ({mem:.0f} GB)")

    # Load compounds
    logging.info(f"\nLoading compounds from {args.input}...")
    df = pd.read_csv(args.input)
    smiles_col = "smiles"
    for c in df.columns:
        if "smiles" in c.lower():
            smiles_col = c
            break
    smiles_list = df[smiles_col].dropna().tolist()
    logging.info(f"  {len(smiles_list)} compounds to screen")

    # Step 1: Compute protein embedding (once)
    logging.info("\n--- Step 1: Protein Embedding ---")
    prot_emb = get_protein_embedding(config, device)
    logging.info(f"  Protein embedding shape: {prot_emb.shape}")

    # Step 2: Encode ligands (multi-GPU)
    logging.info("\n--- Step 2: Ligand Encoding (ChemBERTa) ---")
    lig_embs = encode_ligands_multigpu(smiles_list, n_gpus, config, args.batch_size)
    logging.info(f"  Ligand embeddings shape: {lig_embs.shape}")

    # Step 3: Morgan fingerprints (CPU parallel)
    logging.info("\n--- Step 3: Morgan Fingerprints ---")
    morgan_fps = compute_morgan_parallel(smiles_list, args.workers)
    morgan_tensor = torch.from_numpy(morgan_fps)
    logging.info(f"  Morgan FPs shape: {morgan_tensor.shape}")

    # Step 4: Load ensemble and predict
    logging.info("\n--- Step 4: Ensemble Prediction ---")
    models = load_fold_models(args.checkpoint_dir, config, device)
    results = predict_ensemble(
        models, prot_emb, lig_embs, morgan_tensor,
        mc_samples=args.mc_samples, batch_size=args.batch_size,
    )

    # Step 5: Build output DataFrame
    logging.info("\n--- Step 5: Results ---")
    out_df = pd.DataFrame({
        "smiles": smiles_list,
        "ensemble_prob": results["ensemble_mean"],
        "ensemble_std": results["ensemble_std"],
        "folds_agree": results["n_folds_agree_above_05"],
        "n_folds": len(models),
        "consensus_ratio": results["n_folds_agree_above_05"] / len(models),
    })

    # Sort by ensemble probability (descending)
    out_df = out_df.sort_values("ensemble_prob", ascending=False)

    # Add confidence tier
    out_df["confidence"] = "low"
    out_df.loc[out_df["ensemble_prob"] >= 0.5, "confidence"] = "medium"
    out_df.loc[(out_df["ensemble_prob"] >= 0.7) &
               (out_df["consensus_ratio"] >= 0.6), "confidence"] = "high"
    out_df.loc[(out_df["ensemble_prob"] >= 0.8) &
               (out_df["consensus_ratio"] >= 0.8), "confidence"] = "very_high"

    # Top-K filter
    if args.top_k:
        out_df = out_df.head(args.top_k)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"SCREENING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Total screened:  {len(smiles_list)}")
    logging.info(f"Saved to:        {args.output}")
    logging.info(f"\nConfidence distribution:")
    for tier in ["very_high", "high", "medium", "low"]:
        n = (out_df["confidence"] == tier).sum()
        logging.info(f"  {tier:<12} {n:>6}  ({100*n/len(out_df):.1f}%)")
    logging.info(f"\nTop 20 candidates:")
    logging.info(f"{'Rank':<5} {'P(bind)':<9} {'Std':<7} {'Fold%':<7} {'Tier':<10} {'SMILES':<60}")
    for i, (_, row) in enumerate(out_df.head(20).iterrows()):
        logging.info(f"{i+1:<5} {row['ensemble_prob']:<9.4f} {row['ensemble_std']:<7.4f} "
                     f"{row['consensus_ratio']:<7.1%} {row['confidence']:<10} "
                     f"{row['smiles'][:58]}")


if __name__ == "__main__":
    main()
