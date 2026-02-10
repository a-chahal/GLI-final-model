#!/usr/bin/env python3
"""Check overlap between screening hits and ALL training data."""
import pandas as pd
import numpy as np
import os
import glob
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
    """Load and canonicalize SMILES from a CSV file."""
    if not os.path.exists(path):
        return set(), 0
    df = pd.read_csv(path)
    # Try to find the smiles column
    col = smiles_col
    for c in df.columns:
        if "smiles" in c.lower():
            col = c
            break
    raw = df[col].dropna().tolist()
    canon = set(filter(None, [canonical(s) for s in raw]))
    return canon, len(raw)


def main():
    print("=" * 70)
    print("NOVELTY ANALYSIS: SCREENING HITS vs ALL TRAINING DATA")
    print("=" * 70)

    # ---- Load all training data sources ----
    datasets = {}

    # 1. GLI inhibitors (the 28 positives)
    s, n = load_smiles_from_csv("gli_inhibitors.csv")
    datasets["GLI_POSITIVES"] = s
    print(f"  GLI inhibitors (28 LOOCV positives): {len(s)} unique canonical")

    # 2. GLI finetuning data (augmented)
    s, n = load_smiles_from_csv("gli_finetuning_data_augmented.csv")
    datasets["GLI_FT_AUG"] = s
    print(f"  GLI finetuning augmented: {len(s)} unique canonical (from {n} rows)")

    # 3. GLI finetuning (non-augmented)
    s, n = load_smiles_from_csv("gli_finetuning_data.csv")
    datasets["GLI_FT"] = s
    print(f"  GLI finetuning data: {len(s)} unique canonical")

    # 4. GLI binding data (ChEMBL supplement)
    s, n = load_smiles_from_csv("gli_binding_data.csv")
    datasets["GLI_CHEMBL_SUPP"] = s
    print(f"  GLI binding data (ChEMBL supplement): {len(s)} unique canonical")

    # 5. Negatives
    s, n = load_smiles_from_csv("negatives.csv")
    datasets["NEGATIVES"] = s
    print(f"  Negatives: {len(s)} unique canonical")

    # 6. ZF domain adaptation
    s, n = load_smiles_from_csv("zf_training_combined.csv")
    datasets["ZF_DOMAIN_ADAPT"] = s
    print(f"  ZF domain adaptation: {len(s)} unique canonical")

    # 7. ZF binding only
    s, n = load_smiles_from_csv("zf_training_binding_only.csv")
    datasets["ZF_BINDING"] = s
    print(f"  ZF binding only: {len(s)} unique canonical")

    # 8. BindingDB pretraining (large - canonical all of them)
    s, n = load_smiles_from_csv("bindingdb_200k.csv")
    datasets["BINDINGDB_PRETRAIN"] = s
    print(f"  BindingDB pretrain: {len(s)} unique canonical (from {n} rows)")

    # 9. Augmentation mapping
    s, n = load_smiles_from_csv("augmentation_mapping.csv")
    datasets["AUG_MAPPING"] = s
    print(f"  Augmentation mapping: {len(s)} unique canonical")

    all_train = set()
    for k, v in datasets.items():
        all_train |= v
    print(f"\n  TOTAL unique training SMILES (all sources): {len(all_train)}")

    # ---- Load screening hits ----
    results = pd.read_csv("outputs/screening_results_enriched.csv")
    hits = results[results["ensemble_prob"] >= 0.5].copy()

    # ---- Also check ALL 1119 screened compounds ----
    all_screened = set(filter(None, [canonical(s) for s in results["smiles"]]))
    overlap_all = all_screened & all_train
    print(f"\n  ALL 1119 screened: {len(overlap_all)} overlap with training ({100*len(overlap_all)/len(all_screened):.1f}%)")

    # ---- Check each hit ----
    print()
    print("=" * 70)
    print("HIT-BY-HIT OVERLAP ANALYSIS")
    print("=" * 70)

    novel_count = 0
    leak_count = 0
    for _, row in hits.iterrows():
        smi = canonical(row["smiles"])
        if smi is None:
            continue
        cid = str(row.get("compound_id", "?"))
        prob = row["ensemble_prob"]

        overlaps = []
        for name, smiles_set in datasets.items():
            if smi in smiles_set:
                overlaps.append(name)

        if overlaps:
            leak_count += 1
            tag = "LEAK"
        else:
            novel_count += 1
            tag = "NOVEL"

        src = " | ".join(overlaps) if overlaps else "NOT in any training data"
        print(f"  {cid:18s} P={prob:.4f} [{tag:5s}] {src}")

    print(f"\n  RESULT: {novel_count} NOVEL, {leak_count} IN TRAINING DATA (of {len(hits)} hits)")

    # ---- Deeper analysis: what's in gli_binding_data? ----
    if os.path.exists("gli_binding_data.csv"):
        print()
        print("=" * 70)
        print("ANALYSIS OF gli_binding_data.csv (ChEMBL supplement)")
        print("=" * 70)
        gbd = pd.read_csv("gli_binding_data.csv")
        print(f"  Rows: {len(gbd)}")
        print(f"  Columns: {list(gbd.columns)}")
        for col in gbd.columns:
            if "label" in col.lower() or "active" in col.lower() or "bind" in col.lower():
                print(f"  {col} value counts:")
                print(gbd[col].value_counts().to_string())

    # ---- Check what the collected compounds actually are ----
    print()
    print("=" * 70)
    print("WHAT ARE WE ACTUALLY SCREENING?")
    print("=" * 70)
    collected = pd.read_csv("data/collected/all_collected_compounds.csv")
    print(f"  Total collected: {len(collected)}")
    print(f"  Source breakdown:")
    if "source" in collected.columns:
        for src, cnt in collected["source"].value_counts().items():
            print(f"    {src}: {cnt}")

    # How many collected have activity data?
    if "activity_value" in collected.columns:
        has_act = collected["activity_value"].notna().sum()
        print(f"\n  With activity values: {has_act}/{len(collected)} ({100*has_act/len(collected):.1f}%)")
        actives = collected[collected["activity_value"].notna()]
        if len(actives) > 0:
            print(f"  Activity value stats (nM):")
            print(f"    Mean: {actives['activity_value'].mean():.0f}")
            print(f"    Median: {actives['activity_value'].median():.0f}")
            print(f"    < 1000 nM: {(actives['activity_value'] < 1000).sum()}")
            print(f"    < 10000 nM: {(actives['activity_value'] < 10000).sum()}")
            print(f"    >= 10000 nM: {(actives['activity_value'] >= 10000).sum()}")

    # ---- Cross-reference hits that have known activity ----
    print()
    print("=" * 70)
    print("HITS WITH KNOWN EXPERIMENTAL ACTIVITY")
    print("=" * 70)
    for _, row in hits.iterrows():
        act_val = row.get("activity_value", None)
        act_type = row.get("activity_type", None)
        if pd.notna(act_val):
            cid = str(row.get("compound_id", "?"))
            prob = row["ensemble_prob"]
            print(f"  {cid:18s} P={prob:.4f} {act_type}={act_val} nM  [{row['confidence']}]")


if __name__ == "__main__":
    main()
