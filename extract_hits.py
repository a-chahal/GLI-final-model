"""
Hit Extraction & Post-Processing for GLI Inhibitor Virtual Screening.

Takes ML screening results, applies novelty analysis, prepares docking input,
and (after docking) merges ML + docking scores for final candidate ranking.

Pipeline:
  1. Load screening results from screen_compounds.py
  2. Filter by confidence tier (ensemble_prob >= threshold)
  3. Novelty analysis: Tanimoto distance to all known GLI binders
  4. Diversity selection: Butina clustering to avoid redundant hits
  5. Output: docking-ready CSV + full analysis report

Usage:
    # Step 1: Extract hits from screening results
    python extract_hits.py extract \
        --screening outputs/screening_results.csv \
        --known gli_inhibitors.csv \
        --output outputs/hits_for_docking.csv \
        --top-k 500 --min-prob 0.5

    # Step 2: After docking, merge and rank
    python extract_hits.py merge \
        --hits outputs/hits_for_docking.csv \
        --docking docking_results/docking_results.csv \
        --output outputs/final_candidates.csv
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Novelty Analysis
# ---------------------------------------------------------------------------

def compute_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    """Compute Morgan fingerprint. Returns None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto_to_known(fp, known_fps: list) -> float:
    """Compute maximum Tanimoto similarity to any known binder."""
    if fp is None or not known_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, known_fps)
    return max(sims) if sims else 0.0


def novelty_analysis(hit_smiles: List[str], known_smiles: List[str]) -> Dict[str, np.ndarray]:
    """Compute novelty metrics for hits relative to known binders.
    
    Returns:
        max_tanimoto: max similarity to any known binder (lower = more novel)
        nearest_known: index of most similar known binder
        novel_flag: True if max_tanimoto < 0.4 (genuinely novel scaffold)
    """
    logging.info(f"Computing novelty: {len(hit_smiles)} hits vs {len(known_smiles)} known binders")
    
    known_fps = [compute_fingerprint(s) for s in known_smiles]
    known_fps = [fp for fp in known_fps if fp is not None]
    
    max_tanimotos = []
    nearest_indices = []
    
    for i, smi in enumerate(hit_smiles):
        fp = compute_fingerprint(smi)
        if fp is None:
            max_tanimotos.append(0.0)
            nearest_indices.append(-1)
            continue
        
        sims = DataStructs.BulkTanimotoSimilarity(fp, known_fps)
        if sims:
            max_sim = max(sims)
            nearest_idx = sims.index(max_sim)
        else:
            max_sim = 0.0
            nearest_idx = -1
        
        max_tanimotos.append(max_sim)
        nearest_indices.append(nearest_idx)
        
        if (i + 1) % 100 == 0:
            logging.info(f"  Processed {i + 1}/{len(hit_smiles)}")
    
    max_tanimotos = np.array(max_tanimotos)
    
    return {
        "max_tanimoto": max_tanimotos,
        "nearest_known_idx": np.array(nearest_indices),
        "novel_flag": max_tanimotos < 0.4,
    }


# ---------------------------------------------------------------------------
# Diversity Selection (Butina Clustering)
# ---------------------------------------------------------------------------

def butina_cluster(smiles_list: List[str], cutoff: float = 0.35) -> List[int]:
    """Cluster compounds by Tanimoto similarity using Butina algorithm.
    
    Returns cluster ID for each compound. Cluster 0 is largest.
    """
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        fp = compute_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    
    n = len(fps)
    logging.info(f"Clustering {n} compounds (cutoff={cutoff})...")
    
    # Compute distance matrix (upper triangle)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    
    # Map back to original indices
    cluster_ids = np.full(len(smiles_list), -1, dtype=int)
    for cluster_idx, members in enumerate(clusters):
        for member in members:
            orig_idx = valid_indices[member]
            cluster_ids[orig_idx] = cluster_idx
    
    n_clusters = len(clusters)
    logging.info(f"  {n_clusters} clusters formed")
    logging.info(f"  Largest cluster: {len(clusters[0])} compounds")
    logging.info(f"  Singletons: {sum(1 for c in clusters if len(c) == 1)}")
    
    return cluster_ids


def select_diverse_representatives(df: pd.DataFrame, n_select: int,
                                    cluster_col: str = "cluster_id") -> pd.DataFrame:
    """Select diverse representatives: best-scoring compound from each cluster."""
    # Sort by score descending
    df_sorted = df.sort_values("ensemble_prob", ascending=False)
    
    selected = []
    seen_clusters = set()
    
    # First pass: one per cluster (best scoring)
    for _, row in df_sorted.iterrows():
        cid = row[cluster_col]
        if cid not in seen_clusters:
            selected.append(row)
            seen_clusters.add(cid)
        if len(selected) >= n_select:
            break
    
    # Second pass: fill remaining with highest scoring compounds
    if len(selected) < n_select:
        for _, row in df_sorted.iterrows():
            if row.name not in [s.name for s in selected]:
                selected.append(row)
            if len(selected) >= n_select:
                break
    
    return pd.DataFrame(selected).head(n_select)


# ---------------------------------------------------------------------------
# Extract hits
# ---------------------------------------------------------------------------

def extract_hits(args):
    """Extract and analyze top hits from screening results."""
    logging.info("=" * 60)
    logging.info("HIT EXTRACTION & NOVELTY ANALYSIS")
    logging.info("=" * 60)
    
    # Load screening results
    df = pd.read_csv(args.screening)
    logging.info(f"Loaded {len(df)} screening results")
    
    # Filter by probability threshold
    hits = df[df["ensemble_prob"] >= args.min_prob].copy()
    logging.info(f"Hits above P >= {args.min_prob}: {len(hits)}")
    
    if len(hits) == 0:
        logging.warning("No hits found! Try lowering --min-prob")
        return
    
    # Load known binders
    known_df = pd.read_csv(args.known)
    known_smiles = known_df["smiles"].dropna().tolist()
    name_col = next((c for c in known_df.columns if "name" in c.lower()), known_df.columns[0])
    known_names = known_df[name_col].tolist()
    logging.info(f"Known binders: {len(known_smiles)}")
    
    # Novelty analysis
    novelty = novelty_analysis(hits["smiles"].tolist(), known_smiles)
    hits["max_tanimoto_known"] = novelty["max_tanimoto"]
    hits["nearest_known_idx"] = novelty["nearest_known_idx"]
    hits["nearest_known"] = hits["nearest_known_idx"].map(
        lambda i: known_names[i] if 0 <= i < len(known_names) else "none"
    )
    hits["novel"] = novelty["novel_flag"]
    
    n_novel = hits["novel"].sum()
    logging.info(f"\nNovelty: {n_novel}/{len(hits)} ({100*n_novel/len(hits):.1f}%) are novel (Tc < 0.4)")
    
    # Diversity clustering
    if len(hits) > 50:
        cluster_ids = butina_cluster(hits["smiles"].tolist(), cutoff=0.35)
        hits["cluster_id"] = cluster_ids
        
        # Select diverse top-K
        if args.top_k and len(hits) > args.top_k:
            hits = select_diverse_representatives(hits, args.top_k)
            logging.info(f"Selected {len(hits)} diverse representatives")
    else:
        hits["cluster_id"] = range(len(hits))
    
    # Sort by ensemble probability
    hits = hits.sort_values("ensemble_prob", ascending=False)
    
    # Compute additional descriptors for docking prep
    logging.info("Computing molecular properties...")
    mws = []
    logps = []
    hbds = []
    hbas = []
    for smi in hits["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mws.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            hbds.append(rdMolDescriptors.CalcNumHBD(mol))
            hbas.append(rdMolDescriptors.CalcNumHBA(mol))
        else:
            mws.append(np.nan)
            logps.append(np.nan)
            hbds.append(np.nan)
            hbas.append(np.nan)
    
    hits["mw"] = mws
    hits["logp"] = logps
    hits["hbd"] = hbds
    hits["hba"] = hbas
    
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    hits.to_csv(args.output, index=False)
    logging.info(f"\nSaved {len(hits)} hits to {args.output}")
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"HIT EXTRACTION SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total hits (P >= {args.min_prob}): {len(hits)}")
    logging.info(f"Novel scaffolds (Tc < 0.4): {hits['novel'].sum()}")
    logging.info(f"Unique clusters: {hits['cluster_id'].nunique()}")
    
    if "confidence" in hits.columns:
        for tier in ["very_high", "high", "medium"]:
            n = (hits["confidence"] == tier).sum()
            if n > 0:
                logging.info(f"  {tier}: {n}")
    
    if "source" in hits.columns:
        for src in hits["source"].unique():
            n = (hits["source"] == src).sum()
            logging.info(f"  {src}: {n}")
    
    logging.info(f"\nTop 20 hits:")
    logging.info(f"{'Rank':<5} {'P(bind)':<9} {'Tc_known':<9} {'Novel':<6} {'Nearest':<20} {'SMILES':<50}")
    for i, (_, row) in enumerate(hits.head(20).iterrows()):
        logging.info(f"{i+1:<5} {row['ensemble_prob']:<9.4f} {row['max_tanimoto_known']:<9.3f} "
                     f"{'Y' if row['novel'] else 'N':<6} {str(row['nearest_known'])[:18]:<20} "
                     f"{row['smiles'][:48]}")


# ---------------------------------------------------------------------------
# Merge ML + Docking results
# ---------------------------------------------------------------------------

def merge_results(args):
    """Merge ML screening scores with docking results for final ranking."""
    logging.info("=" * 60)
    logging.info("MERGING ML + DOCKING RESULTS")
    logging.info("=" * 60)
    
    hits = pd.read_csv(args.hits)
    docking = pd.read_csv(args.docking)
    
    logging.info(f"ML hits: {len(hits)}")
    logging.info(f"Docking results: {len(docking)}")
    
    # Merge on SMILES
    merged = hits.merge(
        docking[["smiles", "zf23_score", "zf45_score", "best_site", "best_score",
                 "zf23_status", "zf45_status"]],
        on="smiles", how="left"
    )
    
    # Compute composite score
    # Normalize ML prob to [0, 1] (already is)
    # Normalize docking score: more negative = better, cap at [-12, 0] -> [0, 1]
    def normalize_dock(score, min_score=-12.0, max_score=0.0):
        if pd.isna(score):
            return 0.0
        clamped = max(min_score, min(max_score, score))
        return (max_score - clamped) / (max_score - min_score)
    
    merged["dock_score_norm"] = merged["best_score"].apply(normalize_dock)
    
    # Composite: weighted combination
    # ML weight = 0.6, Docking weight = 0.3, Novelty bonus = 0.1
    w_ml = 0.6
    w_dock = 0.3
    w_novel = 0.1
    
    merged["composite_score"] = (
        w_ml * merged["ensemble_prob"] +
        w_dock * merged["dock_score_norm"] +
        w_novel * merged["novel"].astype(float)
    )
    
    # Final ranking
    merged = merged.sort_values("composite_score", ascending=False)
    merged["final_rank"] = range(1, len(merged) + 1)
    
    # Confidence classification
    merged["final_tier"] = "candidate"
    merged.loc[
        (merged["ensemble_prob"] >= 0.7) &
        (merged["best_score"].fillna(0) <= -6.0) &
        (merged["novel"] == True),
        "final_tier"
    ] = "priority"
    merged.loc[
        (merged["ensemble_prob"] >= 0.8) &
        (merged["best_score"].fillna(0) <= -7.0),
        "final_tier"
    ] = "top_priority"
    
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    merged.to_csv(args.output, index=False)
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL RANKING SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total candidates: {len(merged)}")
    logging.info(f"Docked successfully: {(merged['best_score'].notna()).sum()}")
    
    for tier in ["top_priority", "priority", "candidate"]:
        n = (merged["final_tier"] == tier).sum()
        logging.info(f"  {tier}: {n}")
    
    logging.info(f"\nTop 30 final candidates:")
    logging.info(f"{'Rank':<5} {'Composite':<10} {'P(bind)':<9} {'Dock':<8} {'Novel':<6} {'Tier':<14} {'SMILES':<45}")
    for _, row in merged.head(30).iterrows():
        dock_str = f"{row['best_score']:.1f}" if pd.notna(row['best_score']) else "N/A"
        logging.info(
            f"{row['final_rank']:<5} {row['composite_score']:<10.4f} "
            f"{row['ensemble_prob']:<9.4f} {dock_str:<8} "
            f"{'Y' if row['novel'] else 'N':<6} {row['final_tier']:<14} "
            f"{row['smiles'][:43]}"
        )
    
    logging.info(f"\nSaved to {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLI hit extraction and post-processing")
    subparsers = parser.add_subparsers(dest="command")
    
    # Extract subcommand
    p_extract = subparsers.add_parser("extract", help="Extract hits from screening results")
    p_extract.add_argument("--screening", required=True, help="Screening results CSV")
    p_extract.add_argument("--known", default="gli_inhibitors.csv", help="Known binders CSV")
    p_extract.add_argument("--output", default="outputs/hits_for_docking.csv")
    p_extract.add_argument("--top-k", type=int, default=500, help="Max hits to extract")
    p_extract.add_argument("--min-prob", type=float, default=0.5, help="Min ensemble probability")
    
    # Merge subcommand
    p_merge = subparsers.add_parser("merge", help="Merge ML + docking results")
    p_merge.add_argument("--hits", required=True, help="Hits CSV (from extract)")
    p_merge.add_argument("--docking", required=True, help="Docking results CSV")
    p_merge.add_argument("--output", default="outputs/final_candidates.csv")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        extract_hits(args)
    elif args.command == "merge":
        merge_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
