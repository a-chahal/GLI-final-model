"""
GLI Inhibitor Compound Collection Pipeline

Pulls from freely accessible databases via API:
  - ChEMBL (GLI1, GLI2, SMO, Hedgehog pathway assays)
  - PubChem BioAssay (GLI-luciferase, Hh pathway screens)
  - ZINC20 (substructure/similarity search for GLI scaffolds)
  - DGIdb (drug repurposing for Hh pathway genes)

Provides file parsers for download-based sources:
  - COCONUT (natural products)
  - BindingDB (zinc finger targets)
  - Enamine REAL, Specs, Asinex, MolPort (vendor libraries)

Usage:
    python -m src.collect_compounds --output-dir data/collected
    python -m src.collect_compounds --sources chembl pubchem zinc dgidb
    python -m src.collect_compounds --local-files path/to/coconut.csv path/to/bindingdb.tsv
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set

import requests
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
ZINC_BASE = "https://zinc20.docking.org"
DGIDB_BASE = "https://dgidb.org/api/v2"

API_DELAY = {"chembl": 0.4, "pubchem": 0.25, "zinc": 0.5, "dgidb": 0.3}

# GLI/Hh target UniProt accessions for ChEMBL
GLI_UNIPROT = {
    "GLI1": "P08151", "GLI2": "P10070", "SMO": "Q99835",
    "PTCH1": "Q13635", "SHH": "Q15465",
}

# PubChem Hedgehog/GLI bioassay AIDs
HH_ASSAY_AIDS = [2551, 504339, 504523, 504524, 504525, 2517, 485341, 588489]

# Representative compounds for ZINC similarity search (one per scaffold family)
REFERENCE_SMILES = {
    "8HQ": "COC1=C2C=CC=C(C2=NC=C1)O",                                    # JC19
    "GANT": "CN(C)C1=CC=CC=C1CNCCCNCC2=CC=CC=C2N(C)C",                     # GANT61-D
    "isoflavone": "COc1cc(OC)c2c(=O)c(-c3ccc(OC)c(OC)c3)coc2c1",           # GlaB-like
    "quinoline": "Cc1ccc2c(C(Nc3ccccn3)c4ccncc4)ccc(O)c2n1",               # Wen2023 scaffold
    "diarylamine": "C1CCN(C1)C2=CC=C(C=C2)NCC3=CC=NC=C3",                  # BAS07019774
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def api_get(url: str, params: dict = None, source: str = "chembl",
            retries: int = 3) -> Optional[dict]:
    """GET with retry and rate limiting. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30,
                                headers={"Accept": "application/json"})
            if resp.status_code == 200:
                time.sleep(API_DELAY.get(source, 0.3))
                return resp.json()
            if resp.status_code == 404:
                return None
            logging.warning(f"  API {resp.status_code} from {url} (attempt {attempt+1})")
        except requests.RequestException as e:
            logging.warning(f"  Request error: {e} (attempt {attempt+1})")
        time.sleep(2 ** attempt)
    return None


def canonicalize(smiles: str) -> Optional[str]:
    """Canonicalize SMILES with RDKit. Returns None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ---------------------------------------------------------------------------
# ChEMBL Collection
# ---------------------------------------------------------------------------

def collect_chembl(output_dir: Path) -> pd.DataFrame:
    """Mine ChEMBL for GLI1, GLI2, SMO, PTCH1, SHH activities + Hh pathway assays."""
    cache = output_dir / "chembl_raw.csv"
    if cache.exists():
        logging.info(f"ChEMBL: loading cached {cache}")
        return pd.read_csv(cache)

    all_rows = []

    # Step 1: Get ChEMBL target IDs from UniProt accessions
    target_ids = {}
    for name, uniprot in GLI_UNIPROT.items():
        data = api_get(f"{CHEMBL_BASE}/target.json",
                       {"target_components__accession": uniprot, "limit": 5}, "chembl")
        if data and data.get("targets"):
            tid = data["targets"][0].get("target_chembl_id")
            if tid:
                target_ids[name] = tid
                logging.info(f"  ChEMBL target {name}: {tid}")

    # Step 2: Pull activities for each target
    for name, tid in target_ids.items():
        logging.info(f"  Querying activities for {name} ({tid})...")
        offset = 0
        while True:
            data = api_get(f"{CHEMBL_BASE}/activity.json",
                           {"target_chembl_id": tid, "limit": 1000, "offset": offset},
                           "chembl")
            if not data:
                break
            activities = data.get("activities", [])
            if not activities:
                break
            for act in activities:
                smi = act.get("canonical_smiles")
                if smi:
                    all_rows.append({
                        "smiles": smi,
                        "source": f"ChEMBL_{name}",
                        "compound_id": act.get("molecule_chembl_id", ""),
                        "activity_type": act.get("standard_type", ""),
                        "activity_value": act.get("standard_value", ""),
                        "activity_units": act.get("standard_units", ""),
                    })
            total = data.get("page_meta", {}).get("total_count", 0)
            offset += 1000
            if offset >= total:
                break

    # Step 3: Hedgehog pathway assays (keyword search)
    logging.info("  Searching Hedgehog pathway assays...")
    data = api_get(f"{CHEMBL_BASE}/assay/search.json",
                   {"q": "hedgehog", "limit": 500}, "chembl")
    if data:
        assay_ids = [a["assay_chembl_id"] for a in data.get("assays", [])
                     if a.get("assay_chembl_id")]
        for aid in assay_ids[:50]:  # Cap at 50 assays to avoid excessive API calls
            act_data = api_get(f"{CHEMBL_BASE}/activity.json",
                               {"assay_chembl_id": aid, "limit": 1000}, "chembl")
            if act_data:
                for act in act_data.get("activities", []):
                    smi = act.get("canonical_smiles")
                    if smi:
                        all_rows.append({
                            "smiles": smi,
                            "source": "ChEMBL_Hh_pathway",
                            "compound_id": act.get("molecule_chembl_id", ""),
                            "activity_type": act.get("standard_type", ""),
                            "activity_value": act.get("standard_value", ""),
                            "activity_units": act.get("standard_units", ""),
                        })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    logging.info(f"ChEMBL: {len(df)} records collected")
    return df


# ---------------------------------------------------------------------------
# PubChem Collection
# ---------------------------------------------------------------------------

def collect_pubchem(output_dir: Path) -> pd.DataFrame:
    """Mine PubChem BioAssay for Hh/GLI screens (active + borderline compounds)."""
    cache = output_dir / "pubchem_raw.csv"
    if cache.exists():
        logging.info(f"PubChem: loading cached {cache}")
        return pd.read_csv(cache)

    all_rows = []

    for aid in HH_ASSAY_AIDS:
        logging.info(f"  PubChem AID {aid}...")

        # Try active CIDs first, then all tested
        cids = []
        for cids_type in ["active", "all"]:
            params = {"cids_type": cids_type} if cids_type == "active" else {}
            data = api_get(f"{PUBCHEM_BASE}/assay/aid/{aid}/cids/JSON",
                           params, "pubchem")
            if data:
                try:
                    info_list = data.get("InformationList", {}).get("Information", [])
                    for info in info_list:
                        cids.extend(info.get("CID", []))
                except (KeyError, TypeError):
                    pass
            if cids:
                break

        if not cids:
            logging.info(f"    AID {aid}: no compounds found")
            continue

        cids = list(set(cids))
        logging.info(f"    AID {aid}: {len(cids)} CIDs")

        # Get SMILES in batches of 200 (PubChem URL length limit)
        for i in range(0, len(cids), 200):
            batch = cids[i:i+200]
            cid_str = ",".join(str(c) for c in batch)
            data = api_get(
                f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/CanonicalSMILES,MolecularWeight,XLogP/JSON",
                source="pubchem"
            )
            if data:
                for prop in data.get("PropertyTable", {}).get("Properties", []):
                    smi = prop.get("CanonicalSMILES")
                    if smi:
                        all_rows.append({
                            "smiles": smi,
                            "source": f"PubChem_AID{aid}",
                            "compound_id": f"CID{prop.get('CID', '')}",
                        })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    logging.info(f"PubChem: {len(df)} records collected")
    return df


# ---------------------------------------------------------------------------
# ZINC20 Collection (API-based similarity search)
# ---------------------------------------------------------------------------

def collect_zinc(output_dir: Path) -> pd.DataFrame:
    """Search ZINC20 for compounds similar to known GLI inhibitor scaffolds.

    Uses similarity search (Tanimoto >= 0.3) with property filters
    matching our physicochemical window (MW 250-600, LogP 1-5.5).
    """
    cache = output_dir / "zinc_raw.csv"
    if cache.exists():
        logging.info(f"ZINC: loading cached {cache}")
        return pd.read_csv(cache)

    all_rows = []

    for name, smi in REFERENCE_SMILES.items():
        logging.info(f"  ZINC similarity search: {name}...")
        # ZINC20 substances search API
        params = {
            "smiles": smi,
            "type": "sim",
            "threshold": 0.3,
            "mwt_min": 250, "mwt_max": 600,
            "logp_min": 1.0, "logp_max": 5.5,
            "count": 5000,   # Max per query
        }
        data = api_get(f"{ZINC_BASE}/substances/search.json", params, "zinc")
        if data:
            substances = data if isinstance(data, list) else data.get("substances", data.get("results", []))
            if isinstance(substances, list):
                for sub in substances:
                    sub_smi = sub.get("smiles", sub.get("smi", ""))
                    if sub_smi:
                        all_rows.append({
                            "smiles": sub_smi,
                            "source": f"ZINC_{name}",
                            "compound_id": sub.get("zinc_id", sub.get("id", "")),
                        })
            logging.info(f"    {name}: {len(substances) if isinstance(substances, list) else 0} hits")
        else:
            logging.warning(f"    ZINC API failed for {name} — try downloading tranches manually")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    logging.info(f"ZINC: {len(df)} records collected")
    return df


# ---------------------------------------------------------------------------
# DGIdb Collection (drug repurposing)
# ---------------------------------------------------------------------------

def collect_dgidb(output_dir: Path) -> pd.DataFrame:
    """Query DGIdb for approved/investigational drugs interacting with Hh pathway genes.

    Cross-references ChEMBL IDs to get SMILES.
    """
    cache = output_dir / "dgidb_raw.csv"
    if cache.exists():
        logging.info(f"DGIdb: loading cached {cache}")
        return pd.read_csv(cache)

    genes = ",".join(GLI_UNIPROT.keys())
    logging.info(f"  DGIdb query: {genes}")
    data = api_get(f"{DGIDB_BASE}/interactions.json", {"genes": genes}, "dgidb")

    all_rows = []
    chembl_ids_to_lookup = []

    if data:
        for term in data.get("matchedTerms", []):
            gene = term.get("geneName", "")
            for interaction in term.get("interactions", []):
                drug = interaction.get("drugName", "")
                chembl_id = interaction.get("drugChemblId", "")
                if chembl_id:
                    chembl_ids_to_lookup.append((chembl_id, gene, drug))

    # Look up SMILES from ChEMBL molecule endpoint
    logging.info(f"  Looking up SMILES for {len(chembl_ids_to_lookup)} DGIdb drugs...")
    for chembl_id, gene, drug in chembl_ids_to_lookup:
        mol_data = api_get(f"{CHEMBL_BASE}/molecule/{chembl_id}.json", source="chembl")
        if mol_data:
            structs = mol_data.get("molecule_structures")
            if structs:
                smi = structs.get("canonical_smiles")
                if smi:
                    all_rows.append({
                        "smiles": smi,
                        "source": f"DGIdb_{gene}",
                        "compound_id": chembl_id,
                        "drug_name": drug,
                    })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    logging.info(f"DGIdb: {len(df)} records collected")
    return df


# ---------------------------------------------------------------------------
# Local File Parsers (COCONUT, BindingDB, vendor libraries)
# ---------------------------------------------------------------------------

def load_local_csv(filepath: str, source_name: str, smiles_col: str = "smiles",
                   id_col: str = None) -> pd.DataFrame:
    """Parse a CSV/TSV file containing SMILES.

    Works for:
      - COCONUT (download from coconut.naturalproducts.net/download)
      - BindingDB (download TSV from bindingdb.org/bind/downloads.jsp)
      - Any vendor CSV/TSV with a SMILES column
    """
    logging.info(f"  Loading local file: {filepath}")
    sep = "\t" if filepath.endswith((".tsv", ".txt")) else ","

    try:
        df = pd.read_csv(filepath, sep=sep, usecols=lambda c: c.lower() in
                         {smiles_col.lower(), (id_col or "").lower(), "smiles",
                          "canonical_smiles", "ligand_smiles", "coconut_id",
                          "monoisotopic_mass", "zinc_id", "molecule_chembl_id",
                          "ligand_inchi_key"},
                         dtype=str, on_bad_lines="skip")
    except Exception as e:
        logging.error(f"  Failed to load {filepath}: {e}")
        return pd.DataFrame()

    # Find SMILES column (case-insensitive)
    smi_candidates = ["smiles", "canonical_smiles", "ligand_smiles",
                      "Ligand SMILES", "SMILES", smiles_col]
    smi_col = None
    for c in df.columns:
        if c.lower().replace(" ", "_") in [s.lower().replace(" ", "_") for s in smi_candidates]:
            smi_col = c
            break
    if smi_col is None:
        logging.error(f"  No SMILES column found in {filepath}. Columns: {list(df.columns)}")
        return pd.DataFrame()

    # Find ID column
    found_id_col = None
    for c in df.columns:
        if c != smi_col:
            found_id_col = c
            break

    rows = []
    for _, row in df.iterrows():
        smi = row.get(smi_col)
        if smi and isinstance(smi, str):
            rows.append({
                "smiles": smi.strip(),
                "source": source_name,
                "compound_id": row.get(found_id_col, "") if found_id_col else "",
            })

    result = pd.DataFrame(rows)
    logging.info(f"  {source_name}: {len(result)} records loaded")
    return result


def load_sdf(filepath: str, source_name: str) -> pd.DataFrame:
    """Parse an SDF file to extract SMILES. For Enamine, Specs, Asinex, MolPort files."""
    from rdkit.Chem import SDMolSupplier

    logging.info(f"  Loading SDF: {filepath}")
    supplier = SDMolSupplier(filepath)
    rows = []
    for mol in supplier:
        if mol is not None:
            smi = Chem.MolToSmiles(mol, canonical=True)
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            rows.append({"smiles": smi, "source": source_name, "compound_id": mol_id})

    result = pd.DataFrame(rows)
    logging.info(f"  {source_name}: {len(result)} records loaded")
    return result


# ---------------------------------------------------------------------------
# BindingDB: zinc finger target filter
# ---------------------------------------------------------------------------

def filter_bindingdb_zf(filepath: str) -> pd.DataFrame:
    """Load BindingDB TSV and filter for zinc finger protein targets.

    Download from: https://www.bindingdb.org/bind/downloads.jsp
    Select: 'BindingDB_All' → download TSV
    """
    logging.info("  Filtering BindingDB for zinc finger targets...")
    zf_keywords = ["zinc finger", "zinc-finger", "C2H2", "PHD finger",
                   "RING finger", "LIM domain", "GATA"]

    rows = []
    try:
        # BindingDB is huge — read in chunks
        for chunk in pd.read_csv(filepath, sep="\t", chunksize=50000,
                                 dtype=str, on_bad_lines="skip"):
            # Find target name column
            target_col = None
            for c in chunk.columns:
                if "target" in c.lower() and "name" in c.lower():
                    target_col = c
                    break
            if target_col is None:
                continue

            smi_col = None
            for c in chunk.columns:
                if "smiles" in c.lower() and "ligand" in c.lower():
                    smi_col = c
                    break
            if smi_col is None:
                continue

            mask = chunk[target_col].str.lower().str.contains(
                "|".join(zf_keywords), na=False
            )
            zf_chunk = chunk[mask]
            for _, row in zf_chunk.iterrows():
                smi = row.get(smi_col)
                if smi and isinstance(smi, str):
                    rows.append({
                        "smiles": smi.strip(),
                        "source": "BindingDB_ZF",
                        "compound_id": row.get("BindingDB MonomerID", ""),
                    })
    except Exception as e:
        logging.error(f"  BindingDB filtering failed: {e}")

    result = pd.DataFrame(rows)
    logging.info(f"  BindingDB ZF: {len(result)} records")
    return result


# ---------------------------------------------------------------------------
# Merge & Deduplicate
# ---------------------------------------------------------------------------

def canonicalize_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize SMILES with RDKit, remove invalids, deduplicate."""
    logging.info(f"Canonicalizing {len(df)} compounds...")
    df = df.copy()
    df["canonical_smiles"] = df["smiles"].apply(canonicalize)
    n_invalid = df["canonical_smiles"].isna().sum()
    df = df.dropna(subset=["canonical_smiles"])
    logging.info(f"  Removed {n_invalid} invalid SMILES")

    # Deduplicate by canonical SMILES, keeping track of all sources
    source_map = df.groupby("canonical_smiles")["source"].apply(
        lambda x: "|".join(sorted(set(x)))
    ).to_dict()

    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    df["all_sources"] = df["canonical_smiles"].map(source_map)
    df["smiles"] = df["canonical_smiles"]
    df = df.drop(columns=["canonical_smiles"])

    logging.info(f"  After dedup: {len(df)} unique compounds")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect GLI inhibitor candidates")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "collected"))
    parser.add_argument("--sources", nargs="+",
                        default=["chembl", "pubchem", "zinc", "dgidb"],
                        choices=["chembl", "pubchem", "zinc", "dgidb", "all"],
                        help="API sources to query")
    parser.add_argument("--local-files", nargs="*", default=[],
                        help="Local CSV/TSV/SDF files to include (format: path:source_name)")
    parser.add_argument("--bindingdb-file", type=str, default=None,
                        help="Path to BindingDB_All TSV for zinc finger filtering")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip sources that already have cached results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources
    if "all" in sources:
        sources = ["chembl", "pubchem", "zinc", "dgidb"]

    frames = []

    # API-based collection
    collectors = {
        "chembl": collect_chembl,
        "pubchem": collect_pubchem,
        "zinc": collect_zinc,
        "dgidb": collect_dgidb,
    }
    for src in sources:
        if src in collectors:
            logging.info(f"{'='*60}")
            logging.info(f"Collecting from {src.upper()}...")
            try:
                df = collectors[src](output_dir)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logging.error(f"{src} collection failed: {e}")

    # BindingDB zinc finger filtering
    if args.bindingdb_file:
        logging.info(f"{'='*60}")
        logging.info("Processing BindingDB zinc finger subset...")
        df = filter_bindingdb_zf(args.bindingdb_file)
        if not df.empty:
            frames.append(df)

    # Local file loading
    for file_spec in args.local_files:
        if ":" in file_spec:
            fpath, sname = file_spec.rsplit(":", 1)
        else:
            fpath = file_spec
            sname = Path(fpath).stem

        logging.info(f"{'='*60}")
        if fpath.endswith(".sdf"):
            df = load_sdf(fpath, sname)
        else:
            df = load_local_csv(fpath, sname)
        if not df.empty:
            frames.append(df)

    # Merge and deduplicate
    if not frames:
        logging.error("No compounds collected from any source!")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    logging.info(f"\n{'='*60}")
    logging.info(f"Total raw records: {len(merged)}")

    final = canonicalize_and_deduplicate(merged)

    # Save
    out_path = output_dir / "all_collected_compounds.csv"
    final.to_csv(out_path, index=False)
    logging.info(f"\nSaved {len(final)} unique compounds to {out_path}")

    # Source breakdown
    logging.info("\nSource breakdown:")
    for src, count in final["source"].value_counts().items():
        logging.info(f"  {src}: {count}")

    print(f"\n=== DOWNLOAD INSTRUCTIONS FOR BULK DATABASES ===")
    print(f"For maximum coverage, also download and include these files:")
    print(f"")
    print(f"1. COCONUT (Natural Products, ~400K compounds):")
    print(f"   Download: https://coconut.naturalproducts.net/download")
    print(f"   Run: python -m src.collect_compounds --local-files coconut.csv:COCONUT")
    print(f"")
    print(f"2. BindingDB (Zinc Finger targets):")
    print(f"   Download: https://www.bindingdb.org/bind/downloads.jsp")
    print(f"   Run: python -m src.collect_compounds --bindingdb-file BindingDB_All.tsv")
    print(f"")
    print(f"3. ZINC22 Goldilocks (drug-like, MW 300-550, LogP 1-5):")
    print(f"   Download tranches: https://cartblanche22.docking.org/tranches")
    print(f"   Run: python -m src.collect_compounds --local-files zinc22_goldilocks.smi:ZINC22")
    print(f"")
    print(f"4. Vendor libraries (Enamine REAL, Specs, Asinex, MolPort):")
    print(f"   Download SDF/CSV from vendor websites")
    print(f"   Run: python -m src.collect_compounds --local-files library.sdf:Enamine_REAL")


if __name__ == "__main__":
    main()
