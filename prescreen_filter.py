"""
GLI Inhibitor Pre-Screening Filter Cascade

4-stage filter calibrated against all 28 known GLI binders:
  1. Physicochemical: Calibrated to binder convex hull (MW 150-650, LogP 0.5-8.5, etc.)
  2. Aromatic ring check: >= 1 aromatic ring (all GLI binders have aromatic systems)
  3. Structural relevance: Match any scaffold SMARTS OR Tanimoto >= threshold to known binder
  4. PAINS flagging: Soft flag (not elimination — Compound_1 and BAS07019774 are validated
     binders that trigger PAINS false positives)

Optimized for Rosenbluth (120 CPU cores) via multiprocessing.

Usage:
    python -m src.prescreen_filter --input data/collected/all_collected_compounds.csv \
                                   --output data/prescreened_compounds.csv \
                                   --workers 100
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Filter Constants
# ---------------------------------------------------------------------------

# Stage 1: Physicochemical bounds
# Calibrated to ALL 28 known GLI binders:
#   MW: JC19=175 to Lospinoso=563 → 150-650
#   LogP: BAS09681156=1.87 to Lospinoso=7.94 → 0.5-8.5
#   HBD: GlaB/NTs=0, validated binders max=2 → ≤5 (generous)
#   HBA: BAS=2 to GlaB-family=6 → 1-10
#   RotBonds: JC19=1 to GANT61-D/SST0704=10 → ≤12
PHYSCHEM = {
    "mw_min": 150, "mw_max": 650,
    "logp_min": 0.5, "logp_max": 8.5,
    "hbd_max": 5,
    "hba_min": 1, "hba_max": 10,
    "rotbonds_max": 12,
}

# Stage 3: Scaffold SMARTS for GLI-relevant chemotypes
# A compound passes if it matches ANY SMARTS OR has Tanimoto >= threshold
SCAFFOLD_SMARTS = [
    "[OH]c1cccc2ncccc12",               # 8-Hydroxyquinoline (JC19, BAS01923177)
    "O=c1c(-c2ccccc2)coc2ccccc12",      # Isoflavone / 3-arylchromenone (GlaB, NTs)
    "O=c1ccoc2ccccc12",                  # Chromenone (broader GlaB family)
    "c1ccnc2ccccc12",                    # Quinoline (Wen2023, Manetti)
    "c1ccncc1",                          # Pyridine (BAS07019774)
    "c1cc[nH]n1",                        # Pyrazole (NH form)
    "c1ccnn1",                           # Pyrazole (N-substituted, SST0704)
    "c1cnc[nH]1",                        # Imidazole
    "c1ccc2[nH]ccc2c1",                  # Indole
    "[NX3]c1ccccc1",                     # Arylamino (GANT61-D, BAS compounds)
    "C(=O)NC",                           # Amide linker (Z27610715)
    "[NX3]C([cR1])[cR1]",               # Diarylmethylamine
    "c1ccc(CN)cc1",                      # Benzylamine (BAS06348344, BAS09681156)
    "C1CCNCC1",                          # Piperidine (BAS09681156)
    "C1CCNC1",                           # Pyrrolidine (BAS07019774)
]

# Tanimoto threshold for structural relevance
TANIMOTO_THRESHOLD = 0.3
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ---------------------------------------------------------------------------
# Build PAINS filter
# ---------------------------------------------------------------------------

def _build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


# ---------------------------------------------------------------------------
# Pre-compile scaffold patterns
# ---------------------------------------------------------------------------

def _compile_scaffolds() -> list:
    patterns = []
    for sma in SCAFFOLD_SMARTS:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            patterns.append(pat)
    return patterns


# ---------------------------------------------------------------------------
# Per-compound filter
# ---------------------------------------------------------------------------

def filter_single(smiles: str, ref_fps: list, scaffold_patterns: list,
                  pains_catalog: FilterCatalog) -> Tuple[Optional[str], str, bool]:
    """Apply filter cascade to a single SMILES.

    Returns:
        (canonical_smiles, "pass", is_pains_flagged) if compound passes
        (None, stage_name, False) if compound fails at stage_name
    """
    # Parse
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, "invalid", False)

    can_smi = Chem.MolToSmiles(mol, canonical=True)

    # Stage 1: Physicochemical
    mw = Descriptors.MolWt(mol)
    if not (PHYSCHEM["mw_min"] <= mw <= PHYSCHEM["mw_max"]):
        return (None, "physchem_mw", False)
    logp = Descriptors.MolLogP(mol)
    if not (PHYSCHEM["logp_min"] <= logp <= PHYSCHEM["logp_max"]):
        return (None, "physchem_logp", False)
    hbd = Descriptors.NumHDonors(mol)
    if hbd > PHYSCHEM["hbd_max"]:
        return (None, "physchem_hbd", False)
    hba = Descriptors.NumHAcceptors(mol)
    if not (PHYSCHEM["hba_min"] <= hba <= PHYSCHEM["hba_max"]):
        return (None, "physchem_hba", False)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    if rotbonds > PHYSCHEM["rotbonds_max"]:
        return (None, "physchem_rotbonds", False)

    # Stage 2: Must have >= 1 aromatic ring
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    if n_aromatic_rings < 1:
        return (None, "no_aromatic", False)

    # Stage 3: Structural relevance — pass if EITHER condition met:
    #   (a) Matches ANY scaffold SMARTS, OR
    #   (b) Tanimoto >= threshold to ANY known binder
    has_scaffold = any(mol.HasSubstructMatch(pat) for pat in scaffold_patterns)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    max_sim = max(DataStructs.TanimotoSimilarity(fp, ref) for ref in ref_fps) if ref_fps else 0.0

    if not has_scaffold and max_sim < TANIMOTO_THRESHOLD:
        return (None, "no_structural_relevance", False)

    # Stage 4: PAINS — flag only, do not eliminate
    # Validated GLI binders (Compound_1, BAS07019774) trigger PAINS false positives.
    is_pains = pains_catalog.HasMatch(mol)

    return (can_smi, "pass", is_pains)


# ---------------------------------------------------------------------------
# Worker init (multiprocessing — each worker builds its own catalogs)
# ---------------------------------------------------------------------------

_worker_state = {}

def _worker_init(ref_smiles_list: list):
    """Initialize per-worker state (avoids pickling RDKit objects)."""
    _worker_state["pains"] = _build_pains_catalog()
    _worker_state["scaffolds"] = _compile_scaffolds()

    ref_fps = []
    for smi in ref_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_fps.append(
                AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
            )
    _worker_state["ref_fps"] = ref_fps


def _worker_filter(smiles: str) -> Tuple[Optional[str], str, bool]:
    """Worker function: filter a single SMILES using worker-local state."""
    return filter_single(
        smiles,
        _worker_state["ref_fps"],
        _worker_state["scaffolds"],
        _worker_state["pains"],
    )


# ---------------------------------------------------------------------------
# Run cascade
# ---------------------------------------------------------------------------

def run_cascade(input_csv: str, output_csv: str, reference_csv: str,
                workers: int = None) -> pd.DataFrame:
    """Run the filter cascade with multiprocessing."""

    if workers is None:
        workers = min(max(cpu_count() - 2, 1), 100)

    # Load input
    logging.info(f"Loading compounds from {input_csv}...")
    df = pd.read_csv(input_csv)
    smiles_col = "smiles"
    if smiles_col not in df.columns:
        for c in df.columns:
            if "smiles" in c.lower():
                smiles_col = c
                break
    smiles_list = df[smiles_col].dropna().tolist()
    logging.info(f"  Input: {len(smiles_list)} compounds")

    # Load reference compounds (known GLI binders)
    logging.info(f"Loading reference binders from {reference_csv}...")
    ref_df = pd.read_csv(reference_csv)
    ref_smiles = ref_df["smiles"].tolist()
    logging.info(f"  Reference: {len(ref_smiles)} known binders")

    # Run parallel filtering
    logging.info(f"Running filter cascade with {workers} workers...")
    with Pool(workers, initializer=_worker_init, initargs=(ref_smiles,)) as pool:
        results = pool.map(_worker_filter, smiles_list, chunksize=500)

    # Collect results
    passed = []
    stage_counts = {}
    n_pains_flagged = 0
    for smi, (can_smi, stage, is_pains) in zip(smiles_list, results):
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if stage == "pass":
            passed.append({
                "smiles": can_smi,
                "original_smiles": smi,
                "pains_flag": is_pains,
            })
            if is_pains:
                n_pains_flagged += 1

    # Report
    logging.info(f"\n{'='*50}")
    logging.info(f"Filter Cascade Results:")
    logging.info(f"  Input:          {len(smiles_list):>8}")
    logging.info(f"  Passed:         {len(passed):>8}  ({100*len(passed)/max(len(smiles_list),1):.1f}%)")
    logging.info(f"  PAINS flagged:  {n_pains_flagged:>8}  (kept, flagged for review)")
    logging.info(f"\n  Rejections by stage:")
    for stage in ["invalid", "physchem_mw", "physchem_logp", "physchem_hbd",
                   "physchem_hba", "physchem_rotbonds", "no_aromatic",
                   "no_structural_relevance"]:
        n = stage_counts.get(stage, 0)
        if n > 0:
            logging.info(f"    {stage:<28} {n:>8}")

    # Save
    out_df = pd.DataFrame(passed)
    out_df = out_df.drop_duplicates(subset=["smiles"])
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    logging.info(f"\nSaved {len(out_df)} prescreened compounds to {output_csv}")

    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLI pre-screening filter cascade")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV with SMILES column")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "prescreened_compounds.csv"))
    parser.add_argument("--reference", type=str,
                        default=str(PROJECT_ROOT / "gli_inhibitors.csv"),
                        help="CSV of known GLI binders for Tanimoto filter")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count-2, max 100)")
    parser.add_argument("--tanimoto-threshold", type=float, default=0.3,
                        help="Minimum Tanimoto similarity to any known binder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    global TANIMOTO_THRESHOLD
    TANIMOTO_THRESHOLD = args.tanimoto_threshold

    run_cascade(args.input, args.output, args.reference, args.workers)


if __name__ == "__main__":
    main()
