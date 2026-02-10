#!/usr/bin/env python3
"""Analyze expanded screening results: hit counts, novelty check, source breakdown."""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canonical(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None


def load_smiles_from_csv(path, smiles_col="smiles"):
    import os
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    col = smiles_col
    for c in df.columns:
        if "smiles" in c.lower():
            col = c
            break
    raw = df[col].dropna().tolist()
    return set(filter(None, [canonical(s) for s in raw]))


def main():
    r = pd.read_csv("outputs/screening_expanded_results.csv")
    hits = r[r["ensemble_prob"] >= 0.5].copy()

    print("=" * 70)
    print("EXPANDED SCREENING RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total screened: {len(r)}")
    print(f"Hits (P>=0.5): {len(hits)} ({100*len(hits)/len(r):.1f}%)")

    for tier in ["very_high", "high", "medium"]:
        n = (hits["confidence"] == tier).sum()
        print(f"  {tier}: {n}")

    # Source breakdown of hits
    print()
    print("=" * 70)
    print("HIT SOURCE BREAKDOWN")
    print("=" * 70)
    collected = pd.read_csv("data/collected_expanded/all_collected_compounds.csv")
    # Build SMILES -> metadata lookup
    meta = {}
    for _, row in collected.iterrows():
        smi = canonical(str(row.get("smiles", "")))
        if smi:
            meta[smi] = {
                "source": row.get("source", ""),
                "all_sources": row.get("all_sources", row.get("source", "")),
                "compound_id": row.get("compound_id", ""),
                "activity_type": row.get("activity_type", ""),
                "activity_value": row.get("activity_value", ""),
            }

    # Tag each hit with source
    sources = {}
    for _, row in hits.iterrows():
        smi = canonical(str(row["smiles"]))
        info = meta.get(smi, {})
        src = str(info.get("all_sources", info.get("source", "unknown")))
        for s in src.split("|"):
            s = s.strip()
            if s:
                sources[s] = sources.get(s, 0) + 1

    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    # NOVELTY CHECK
    print()
    print("=" * 70)
    print("NOVELTY CHECK: HITS vs ALL TRAINING DATA")
    print("=" * 70)

    gli_smiles = load_smiles_from_csv("gli_inhibitors.csv")
    neg_smiles = load_smiles_from_csv("negatives.csv")
    zf_smiles = load_smiles_from_csv("zf_training_combined.csv")
    bdb_smiles = load_smiles_from_csv("bindingdb_200k.csv")

    all_train = gli_smiles | neg_smiles | zf_smiles | bdb_smiles
    print(f"Training data: {len(all_train)} unique SMILES")
    print(f"  GLI positives: {len(gli_smiles)}")
    print(f"  Negatives: {len(neg_smiles)}")
    print(f"  ZF domain adapt: {len(zf_smiles)}")
    print(f"  BindingDB pretrain: {len(bdb_smiles)}")

    novel = 0
    leaked = 0
    novel_hits = []
    leaked_hits = []

    for _, row in hits.iterrows():
        smi = canonical(str(row["smiles"]))
        if smi and smi in all_train:
            leaked += 1
            leaked_hits.append(row)
        else:
            novel += 1
            novel_hits.append(row)

    print(f"\nResults:")
    print(f"  NOVEL (not in training): {novel}")
    print(f"  IN TRAINING DATA:       {leaked}")
    print(f"  Novelty rate:           {100*novel/len(hits):.1f}%")

    # Where do leaked hits come from?
    print()
    print("LEAKED HITS - training set breakdown:")
    leak_sources = {"GLI_POS": 0, "NEG": 0, "ZF": 0, "BDB": 0}
    for row in leaked_hits:
        smi = canonical(str(row["smiles"]))
        if smi in gli_smiles:
            leak_sources["GLI_POS"] += 1
        if smi in neg_smiles:
            leak_sources["NEG"] += 1
        if smi in zf_smiles:
            leak_sources["ZF"] += 1
        if smi in bdb_smiles:
            leak_sources["BDB"] += 1
    for k, v in leak_sources.items():
        print(f"  {k}: {v}")

    # Top 20 NOVEL hits only
    print()
    print("=" * 70)
    print("TOP 20 TRULY NOVEL HITS (not in any training data)")
    print("=" * 70)
    novel_df = pd.DataFrame(novel_hits)
    novel_df = novel_df.sort_values("ensemble_prob", ascending=False)

    for i, (_, row) in enumerate(novel_df.head(20).iterrows()):
        smi = row["smiles"]
        prob = row["ensemble_prob"]
        std = row["ensemble_std"]
        conf = row["confidence"]
        info = meta.get(canonical(str(smi)), {})
        cid = info.get("compound_id", "")
        src = info.get("all_sources", info.get("source", ""))
        act_type = info.get("activity_type", "")
        act_val = info.get("activity_value", "")
        act_str = ""
        if pd.notna(act_val) and str(act_val).strip():
            act_str = f" | {act_type}={act_val}nM"
        print(f"  {i+1:2d}. {cid:18s} P={prob:.4f} std={std:.4f} [{conf}] src={src}{act_str}")
        print(f"      {smi[:80]}")

    # Confidence breakdown for novel vs leaked
    print()
    print("=" * 70)
    print("CONFIDENCE TIER: NOVEL vs LEAKED")
    print("=" * 70)
    novel_df_all = pd.DataFrame(novel_hits)
    leaked_df_all = pd.DataFrame(leaked_hits)
    for tier in ["very_high", "high", "medium"]:
        n_novel = (novel_df_all["confidence"] == tier).sum() if len(novel_df_all) > 0 else 0
        n_leak = (leaked_df_all["confidence"] == tier).sum() if len(leaked_df_all) > 0 else 0
        print(f"  {tier:12s}: {n_novel:5d} novel, {n_leak:5d} leaked")


if __name__ == "__main__":
    main()
