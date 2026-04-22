#!/usr/bin/env python3
"""
Dock GANT61-D and GlaB at their validated binding sites on GLI1,
then create a reference CSV for the MD pipeline.
"""

import os, sys, csv, subprocess, tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
import numpy as np

PROJECT = Path(os.environ.get("PROJECT_DIR", os.path.expanduser("~/GLI-final-model")))
RECEPTOR_PDBQT = PROJECT / "2gli_apo_prepared.pdbqt"
RECEPTOR_PDB   = PROJECT / "2gli_with_zinc.pdb"
DOCK_OUT       = PROJECT / "docking_results_v2"

# ---- Reference compounds ----
REFERENCES = [
    {
        "compound_id": "GANT61-D",
        "smiles": "CN(C)C1=CC=CC=C1CNCCCNCC2=CC=CC=C2N(C)C",
        "site": "ZF2-3",
        "exhaustiveness": 128,   # 14 rotatable bonds → needs extensive sampling
    },
    {
        "compound_id": "GlaB",
        "smiles": "CC(=CCOC1=C(C=C(C=C1)C2=COC3=C(C2=O)C(=CC(=C3)OC)OC)OCC=C(C)C)C",
        "site": "ZF4-5",
        "exhaustiveness": 128,   # prenyl chains are flexible
    },
]

# Binding site centers (from batch_dock_dual_sites.py setup)
# These are computed from E119/E167 midpoint (ZF2-3) and K209/K219 midpoint (ZF4-5)
from Bio.PDB import PDBParser

def get_site_centers():
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('GLI', str(RECEPTOR_PDB))
    coords = {}
    for chain in structure[0]:
        if chain.id == 'A':
            for res in chain:
                if res.id[0] == ' ':
                    try:
                        ca = res['CA'].coord
                        rid = res.id[1]
                        if res.resname == 'GLU' and rid in (119, 167):
                            coords[f"E{rid}"] = ca
                        elif res.resname == 'LYS' and rid in (209, 219):
                            coords[f"K{rid}"] = ca
                    except:
                        pass
    zf23 = (coords["E119"] + coords["E167"]) / 2
    zf45 = (coords["K209"] + coords["K219"]) / 2
    return {
        "ZF2-3": [float(zf23[0]), float(zf23[1]), float(zf23[2])],
        "ZF4-5": [float(zf45[0]), float(zf45[1]), float(zf45[2])],
    }


def smiles_to_pdbqt(smiles, name, outdir):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    pdb = outdir / f"{name}.pdb"
    pdbqt = outdir / f"{name}.pdbqt"
    Chem.MolToPDBFile(mol, str(pdb))
    subprocess.run(f"obabel {pdb} -O {pdbqt}", shell=True,
                   capture_output=True, text=True)
    return str(pdbqt)


def dock(pdbqt_lig, center, out_pdbqt, exhaustiveness=32):
    v = Vina(sf_name='vina')
    v.set_receptor(str(RECEPTOR_PDBQT))
    v.compute_vina_maps(center=center, box_size=[24, 24, 24])
    v.set_ligand_from_file(pdbqt_lig)
    v.dock(exhaustiveness=exhaustiveness, n_poses=20)
    energies = v.energies(n_poses=20)
    score = float(energies[0][0]) if isinstance(energies[0], (list, tuple, np.ndarray)) else float(energies[0])
    v.write_poses(out_pdbqt, n_poses=20, overwrite=True)
    return score


def main():
    centers = get_site_centers()
    print(f"ZF2-3 center: {centers['ZF2-3']}")
    print(f"ZF4-5 center: {centers['ZF4-5']}")

    tmpdir = Path(tempfile.mkdtemp())
    results = []

    for ref in REFERENCES:
        cid = ref["compound_id"]
        site = ref["site"]
        smi = ref["smiles"]
        print(f"\n{'='*60}")
        print(f"Docking {cid} → {site}")
        print(f"SMILES: {smi}")

        # Prepare ligand
        lig_pdbqt = smiles_to_pdbqt(smi, cid.replace("-", "_"), tmpdir)

        exh = ref.get("exhaustiveness", 32)
        print(f"  Exhaustiveness: {exh}")

        # Dock at primary site
        site_dir = DOCK_OUT / site
        site_dir.mkdir(parents=True, exist_ok=True)
        tag = "zf23" if site == "ZF2-3" else "zf45"
        out_pdbqt = site_dir / f"{cid}.0_{tag}.pdbqt"
        score_primary = dock(lig_pdbqt, centers[site], str(out_pdbqt),
                             exhaustiveness=exh)
        print(f"  {site} score: {score_primary:.3f} kcal/mol")

        # Dock at other site too
        other_site = "ZF4-5" if site == "ZF2-3" else "ZF2-3"
        other_dir = DOCK_OUT / other_site
        other_dir.mkdir(parents=True, exist_ok=True)
        other_tag = "zf45" if tag == "zf23" else "zf23"
        out_pdbqt2 = other_dir / f"{cid}.0_{other_tag}.pdbqt"
        score_other = dock(lig_pdbqt, centers[other_site], str(out_pdbqt2),
                           exhaustiveness=exh)
        print(f"  {other_site} score: {score_other:.3f} kcal/mol")

        # Compute properties
        mol = Chem.MolFromSmiles(smi)
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        zf23_score = score_primary if site == "ZF2-3" else score_other
        zf45_score = score_other if site == "ZF2-3" else score_primary
        best = min(zf23_score, zf45_score)
        best_site = "ZF2-3" if zf23_score <= zf45_score else "ZF4-5"

        results.append({
            "smiles": smi,
            "ensemble_prob": 1.0,  # known binder
            "ensemble_std": 0.0,
            "folds_agree": 28,
            "n_folds": 28,
            "consensus_ratio": 1.0,
            "mw": f"{mw:.1f}",
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formal_charge": 0,
            "pains_flag": False,
            "source": "reference",
            "compound_id": f"{cid}.0",
            "confidence": "reference",
            "max_tanimoto_known": 1.0,
            "nearest_known_idx": 0,
            "nearest_known": cid,
            "novel": False,
            "cluster_id": 0,
            "logp": f"{logp:.2f}",
            "hbd": hbd,
            "hba": hba,
            "zf23_score": f"{zf23_score:.3f}",
            "zf45_score": f"{zf45_score:.3f}",
            "best_dock_score": f"{best:.3f}",
            "best_site": best_site,
        })

        print(f"  Best: {best:.3f} @ {best_site}")

    # Write reference CSV
    out_csv = PROJECT / "outputs" / "reference_compounds_for_md.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nReference CSV: {out_csv}")
    print("Done! Now run the MD pipeline with --candidates-csv outputs/reference_compounds_for_md.csv")


if __name__ == "__main__":
    main()
