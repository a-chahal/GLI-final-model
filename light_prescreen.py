"""
Light Prescreen Filter for Large Compound Libraries.

Minimal, physics-justified filters that do NOT replicate ML model decisions:
  1. Valid SMILES (RDKit parseable)
  2. MW 120-800 Da
  3. Heavy atoms >= 8
  4. Formal charge -3 to +3
  5. Desalt (keep largest fragment)

NO scaffold SMARTS, NO Tanimoto similarity, NO LogP, NO HBD/HBA, NO aromatic ring requirement.
The model carries the burden of selectivity.

PAINS are soft-flagged in metadata, never used for elimination.

Usage:
    python light_prescreen.py \
        --enamine /path/to/enamine.csv \
        --coconut /path/to/coconut.csv \
        --output data/libraries/all_filtered.csv \
        --workers 60
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MW_MIN = 120
MW_MAX = 800
HEAVY_ATOMS_MIN = 8
CHARGE_MIN = -3
CHARGE_MAX = 3

# ---------------------------------------------------------------------------
# PAINS catalog (soft flag only)
# ---------------------------------------------------------------------------
def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


# ---------------------------------------------------------------------------
# Single-compound filter
# ---------------------------------------------------------------------------
def filter_single(smiles: str) -> Optional[Tuple[str, float, int, int, bool]]:
    """Apply light filter to a single SMILES.

    Returns:
        (canonical_smiles, mw, heavy_atoms, formal_charge, pains_flag) or None if rejected.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # Desalt: keep largest fragment
    if "." in smiles:
        frags = smiles.split(".")
        smiles = max(frags, key=len)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # MW
    mw = Descriptors.MolWt(mol)
    if mw < MW_MIN or mw > MW_MAX:
        return None

    # Heavy atoms
    ha = mol.GetNumHeavyAtoms()
    if ha < HEAVY_ATOMS_MIN:
        return None

    # Formal charge
    fc = Chem.GetFormalCharge(mol)
    if fc < CHARGE_MIN or fc > CHARGE_MAX:
        return None

    # Canonical SMILES
    canon = Chem.MolToSmiles(mol)

    return (canon, mw, ha, fc, False)  # PAINS checked in batch later


def filter_batch(smiles_list):
    """Filter a batch of SMILES. For multiprocessing."""
    results = []
    for smi in smiles_list:
        result = filter_single(smi)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Library loaders
# ---------------------------------------------------------------------------
def load_enamine(path: str) -> pd.DataFrame:
    """Load Enamine Hit Locator Library CSV or SMILES file."""
    logging.info(f"Loading Enamine data from {path}...")

    if path.endswith(".smiles"):
        # Tab-separated: SMILES  Catalog ID  MW  ...
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        # CSV format
        # First line might be "sep=," directive
        with open(path, "r") as f:
            first_line = f.readline().strip()

        if first_line.startswith("sep="):
            df = pd.read_csv(path, skiprows=1, low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)

    # Find SMILES column
    smiles_col = None
    id_col = None
    for c in df.columns:
        cl = c.strip().upper()
        if cl == "SMILES":
            smiles_col = c
        elif "CATALOG" in cl or cl == "ID":
            id_col = c

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in Enamine file. Columns: {list(df.columns)}")

    result = pd.DataFrame({
        "smiles": df[smiles_col].astype(str),
        "compound_id": df[id_col].astype(str) if id_col else [f"EN_{i}" for i in range(len(df))],
        "source": "Enamine_HitLocator",
    })

    logging.info(f"  Loaded {len(result)} Enamine compounds")
    return result.dropna(subset=["smiles"])


def load_coconut(path: str) -> pd.DataFrame:
    """Load COCONUT natural products CSV."""
    logging.info(f"Loading COCONUT data from {path}...")

    df = pd.read_csv(path, low_memory=False)

    # Find SMILES and ID columns
    smiles_col = None
    id_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if "canonical_smiles" in cl:
            smiles_col = c
        elif cl == "smiles":
            smiles_col = c
        elif "identifier" in cl:
            id_col = c
        elif cl == "id":
            id_col = c

    if smiles_col is None:
        raise ValueError(f"No SMILES column in COCONUT file. Columns: {list(df.columns)[:20]}")

    result = pd.DataFrame({
        "smiles": df[smiles_col].astype(str),
        "compound_id": df[id_col].astype(str) if id_col else [f"CNP_{i}" for i in range(len(df))],
        "source": "COCONUT",
    })

    logging.info(f"  Loaded {len(result)} COCONUT compounds")
    return result.dropna(subset=["smiles"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Light prescreen for compound libraries")
    parser.add_argument("--enamine", type=str, help="Path to Enamine .csv or .smiles file")
    parser.add_argument("--coconut", type=str, help="Path to COCONUT .csv file")
    parser.add_argument("--output", type=str, required=True, help="Output filtered CSV path")
    parser.add_argument("--workers", type=int, default=min(60, cpu_count()),
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for multiprocessing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.enamine and not args.coconut:
        parser.error("Provide at least one of --enamine or --coconut")

    # Load all libraries
    dfs = []
    if args.enamine:
        dfs.append(load_enamine(args.enamine))
    if args.coconut:
        dfs.append(load_coconut(args.coconut))

    combined = pd.concat(dfs, ignore_index=True)
    logging.info(f"\nTotal raw compounds: {len(combined)}")

    # Deduplicate by SMILES before filtering
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["smiles"], keep="first")
    logging.info(f"After dedup: {len(combined)} (removed {n_before - len(combined)} duplicates)")

    # Filter with multiprocessing
    all_smiles = combined["smiles"].tolist()
    n = len(all_smiles)
    batch_size = args.batch_size
    batches = [all_smiles[i:i + batch_size] for i in range(0, n, batch_size)]

    logging.info(f"Filtering {n} compounds with {args.workers} workers, "
                 f"{len(batches)} batches of {batch_size}...")

    passed = []
    with Pool(args.workers) as pool:
        for i, batch_results in enumerate(pool.imap(filter_batch, batches)):
            passed.extend(batch_results)
            if (i + 1) % 20 == 0 or i == len(batches) - 1:
                logging.info(f"  Batch {i + 1}/{len(batches)}: "
                             f"{len(passed)} passed so far ({len(passed) / ((i + 1) * batch_size) * 100:.1f}%)")

    logging.info(f"\nFilter results: {len(passed)}/{n} passed ({len(passed)/n*100:.1f}%)")

    # Build result DataFrame
    filtered_df = pd.DataFrame(passed, columns=["smiles", "mw", "heavy_atoms", "formal_charge", "pains_flag"])

    # Merge back source and compound_id
    combined_lookup = combined.set_index("smiles")
    source_map = combined_lookup["source"].to_dict()
    id_map = combined_lookup["compound_id"].to_dict()

    # Use canonical SMILES for lookup — need original SMILES too
    # Since canonicalization may change SMILES, also try original
    original_smiles = combined["smiles"].tolist()
    original_canon = {}
    for smi, src, cid in zip(combined["smiles"], combined["source"], combined["compound_id"]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canon = Chem.MolToSmiles(mol)
            original_canon[canon] = (src, cid)

    filtered_df["source"] = filtered_df["smiles"].map(lambda s: original_canon.get(s, ("unknown", "unknown"))[0])
    filtered_df["compound_id"] = filtered_df["smiles"].map(lambda s: original_canon.get(s, ("unknown", "unknown"))[1])

    # PAINS soft flagging (single-threaded, fast on filtered set)
    logging.info("Applying PAINS soft flags...")
    pains_catalog = _build_pains_catalog()
    pains_flags = []
    for smi in filtered_df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            pains_flags.append(pains_catalog.HasMatch(mol))
        else:
            pains_flags.append(False)
    filtered_df["pains_flag"] = pains_flags
    n_pains = sum(pains_flags)
    logging.info(f"  PAINS flagged: {n_pains} ({n_pains/len(filtered_df)*100:.1f}%) — soft flag only, NOT removed")

    # Final dedup on canonical SMILES
    n_before = len(filtered_df)
    filtered_df = filtered_df.drop_duplicates(subset=["smiles"], keep="first")
    logging.info(f"Final dedup: {len(filtered_df)} unique compounds (removed {n_before - len(filtered_df)})")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    filtered_df.to_csv(args.output, index=False)
    logging.info(f"\nSaved to {args.output}")

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"PRESCREEN SUMMARY")
    logging.info(f"{'='*60}")
    for src in filtered_df["source"].unique():
        n_src = (filtered_df["source"] == src).sum()
        logging.info(f"  {src}: {n_src}")
    logging.info(f"  Total: {len(filtered_df)}")
    logging.info(f"  MW range: {filtered_df['mw'].min():.1f} - {filtered_df['mw'].max():.1f}")
    logging.info(f"  PAINS flagged: {n_pains} (kept, soft flag)")


if __name__ == "__main__":
    main()
