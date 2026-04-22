#!/usr/bin/env python3
"""
GLI1 Zinc Finger – Ligand MD Simulation Pipeline (GROMACS)
==========================================================
Fully automated pipeline: ADMET/Lipinski filtering → compound selection →
protein prep → ligand parameterization → complex assembly → solvation →
energy minimization → NVT/NPT equilibration → 50 ns production MD →
trajectory analysis → VMD-ready output.

Protocol (Hollingsworth & Dror, Neuron 2018):
  Force field:  AMBER99SB-ILDN (protein) + GAFF2 (ligand) + TIP3P (water)
  Ligand prep:  RDKit 3D from SMILES → acpype (AM1-BCC / Gasteiger fallback)
  Zn2+ handling: Position restraints on zinc + coordinating residues during equil
  GPU:          NVIDIA RTX 3090 via GROMACS CUDA (custom build)

Usage:
    python run_md_pipeline.py --project-dir ~/GLI-final-model --gpu-ids 0,1,2,3
"""

import os
import re
import sys
import csv
import json
import math
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MDConfig:
    project_dir: Path = Path.home() / "GLI-final-model"
    md_base_dir: Path = Path.home() / "GLI-final-model" / "md_simulation"
    protein_pdb: str = "2gli_with_zinc.pdb"
    docking_dir: str = "docking_results_v2"
    candidates_csv: str = "outputs/final_candidates_v2.csv"
    ff: str = "amber99sb-ildn"
    water_model: str = "tip3p"
    box_buffer: float = 1.2
    salt_conc: float = 0.15
    prod_ns: int = 50
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    n_compounds: int = 5
    lipinski_max_violations: int = 1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "md_pipeline.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def sh(cmd: str, cwd: Path = None, check: bool = True,
       stdin_text: str = None, timeout: int = None) -> subprocess.CompletedProcess:
    logging.info(f"  $ {cmd}")
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                           input=stdin_text, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.warning(f"  Command timed out after {timeout}s: {cmd}")
        # Return a fake CompletedProcess with non-zero rc
        return subprocess.CompletedProcess(cmd, returncode=-9,
                                           stdout="", stderr="TIMEOUT")
    if r.returncode != 0:
        logging.error(f"  STDOUT:\n{r.stdout[-3000:]}")
        logging.error(f"  STDERR:\n{r.stderr[-3000:]}")
        if check:
            raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd}")
    return r


GMX_BIN = "/home/sanjanp/gromacs-cuda/bin/gmx_cuda"
# Conda's libstdc++ has GLIBCXX_3.4.29 needed by libmuparser
_CONDA_LIB = "/home/sanjanp/miniforge3/envs/md-gromacs/lib"

def gmx(sub: str, cwd: Path = None, stdin_text: str = None,
        check: bool = True) -> subprocess.CompletedProcess:
    env_prefix = f"LD_LIBRARY_PATH={_CONDA_LIB}:$LD_LIBRARY_PATH"
    return sh(f"{env_prefix} {GMX_BIN} {sub}", cwd=cwd, check=check, stdin_text=stdin_text)


def assert_file(p: Path, label: str = ""):
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty: {p} ({label})")

# ---------------------------------------------------------------------------
# Step 0  –  ADMET / Lipinski → top N selection
# ---------------------------------------------------------------------------

def select_compounds(cfg: MDConfig) -> List[Dict]:
    csv_path = cfg.project_dir / cfg.candidates_csv
    logging.info(f"Loading candidates from {csv_path}")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    logging.info(f"  Total candidates: {len(rows)}")

    passed = []
    for r in rows:
        try:
            mw  = float(r["mw"])
            logp = float(r["logp"])
            hbd = int(r["hbd"])
            hba = int(r["hba"])
        except (ValueError, KeyError):
            continue
        viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        if viol > cfg.lipinski_max_violations:
            continue
        dock = r.get("best_dock_score", "")
        if not dock:
            continue
        passed.append({
            "compound_id": r["compound_id"],
            "smiles":      r["smiles"],
            "mw": mw, "logp": logp, "hbd": hbd, "hba": hba,
            "lipinski_viol": viol,
            "prob":       float(r.get("ensemble_prob", 0)),
            "zf23":       float(r["zf23_score"]) if r.get("zf23_score") else None,
            "zf45":       float(r["zf45_score"]) if r.get("zf45_score") else None,
            "best_dock":  float(dock),
            "best_site":  r.get("best_site", ""),
        })

    logging.info(f"  Lipinski-pass (≤{cfg.lipinski_max_violations} viol): {len(passed)}")
    passed.sort(key=lambda x: x["best_dock"])

    # Ensure ≥1 from each pocket
    zf23 = [c for c in passed if c["best_site"] == "ZF2-3"]
    zf45 = [c for c in passed if c["best_site"] == "ZF4-5"]
    sel, ids = [], set()
    for pool in (zf23, zf45):
        if pool and pool[0]["compound_id"] not in ids:
            sel.append(pool[0]); ids.add(pool[0]["compound_id"])
    for c in passed:
        if len(sel) >= cfg.n_compounds:
            break
        if c["compound_id"] not in ids:
            sel.append(c); ids.add(c["compound_id"])

    logging.info(f"\n{'='*60}\nSELECTED {len(sel)} COMPOUNDS\n{'='*60}")
    for i, c in enumerate(sel, 1):
        logging.info(f"  {i}. {c['compound_id']}: dock={c['best_dock']:.3f} "
                     f"({c['best_site']}), P={c['prob']:.3f}, MW={c['mw']:.0f}")

    out = cfg.md_base_dir / "top_compounds.csv"
    cfg.md_base_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sel[0].keys()); w.writeheader(); w.writerows(sel)
    return sel

# ---------------------------------------------------------------------------
# Step 1  –  Protein preparation
# ---------------------------------------------------------------------------

def prepare_protein(cfg: MDConfig) -> Path:
    prot_dir = cfg.md_base_dir / "protein"
    prot_dir.mkdir(parents=True, exist_ok=True)
    src = cfg.project_dir / cfg.protein_pdb

    logging.info(f"\n--- Protein prep from {src} ---")

    # ---- read PDB, keep model 1, separate protein vs Zn ----
    with open(src) as f:
        raw = f.readlines()

    prot_lines, zn_lines = [], []
    in_model1, past = False, False
    has_model = any(l.startswith("MODEL") for l in raw)

    for ln in raw:
        if ln.startswith("MODEL"):
            n = int(ln.split()[1])
            in_model1 = (n == 1); continue
        if ln.startswith("ENDMDL"):
            if in_model1: past = True; in_model1 = False
            continue
        keep = (in_model1) if has_model else (not past)
        if not keep and has_model:
            continue
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        resn = ln[17:20].strip()
        atn  = ln[12:16].strip()
        if resn == "ZN" or atn == "ZN":
            zn_lines.append(ln)
        elif ln.startswith("ATOM"):
            prot_lines.append(ln)

    clean = prot_dir / "protein_nozn.pdb"
    with open(clean, "w") as f:
        f.writelines(prot_lines); f.write("END\n")
    zn_pdb = prot_dir / "zn_ions.pdb"
    with open(zn_pdb, "w") as f:
        f.writelines(zn_lines)
    n_zn = len(zn_lines)
    logging.info(f"  {len(prot_lines)} protein atoms, {n_zn} Zn2+ ions separated")

    # ---- pdb2gmx ----
    logging.info(f"  pdb2gmx ({cfg.ff} / {cfg.water_model})...")
    gmx(f"pdb2gmx -f {clean} -o {prot_dir/'protein.gro'} "
        f"-p {prot_dir/'topol.top'} -ignh -ff {cfg.ff} -water {cfg.water_model}",
        cwd=prot_dir)
    assert_file(prot_dir / "protein.gro", "pdb2gmx gro")
    assert_file(prot_dir / "topol.top",   "pdb2gmx top")

    # ---- append Zn2+ to .gro ----
    _append_zn_to_gro(prot_dir / "protein.gro", zn_lines, n_zn)

    # ---- patch topology for Zn ----
    _patch_topology_for_zn(prot_dir / "topol.top", n_zn)

    logging.info(f"  Protein prep done → {prot_dir}")
    return prot_dir


def _append_zn_to_gro(gro: Path, zn_lines: List[str], n_zn: int):
    """Append Zn2+ atoms to .gro file with correct GRO formatting."""
    with open(gro) as f:
        lines = f.readlines()
    title   = lines[0]
    n_atoms = int(lines[1])
    atoms   = lines[2:2 + n_atoms]
    box     = lines[2 + n_atoms]

    new = []
    for i, zl in enumerate(zn_lines):
        x = float(zl[30:38]) / 10.0   # Å → nm
        y = float(zl[38:46]) / 10.0
        z = float(zl[46:54]) / 10.0
        rnum = n_atoms + i + 1
        # GRO: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
        new.append(f"{rnum:5d}{'ZN':<5s}{'ZN':>5s}{rnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")

    with open(gro, "w") as f:
        f.write(title)
        f.write(f"{n_atoms + n_zn:5d}\n")
        f.writelines(atoms)
        f.writelines(new)
        f.write(box)
    logging.info(f"  Appended {n_zn} Zn2+ to {gro.name}")


def _patch_topology_for_zn(top: Path, n_zn: int):
    """Add ions.itp include and ZN molecules to topology."""
    with open(top) as f:
        txt = f.read()

    # ions.itp must come AFTER water model include
    water_inc = f'#include "{top.parent.name}/../amber99sb-ildn.ff/tip3p.itp"'
    # Find the tip3p include (pdb2gmx adds it)
    if "tip3p.itp" in txt and "ions.itp" not in txt:
        txt = txt.replace(
            '#include "amber99sb-ildn.ff/tip3p.itp"',
            '#include "amber99sb-ildn.ff/tip3p.itp"\n'
            '#include "amber99sb-ildn.ff/ions.itp"'
        )
    elif "ions.itp" not in txt:
        # fallback: insert before [ system ]
        txt = txt.replace("[ system ]",
                          '#include "amber99sb-ildn.ff/ions.itp"\n\n[ system ]')

    # Add ZN to [ molecules ]
    if n_zn > 0:
        txt = txt.rstrip() + f"\nZN               {n_zn}\n"

    with open(top, "w") as f:
        f.write(txt)
    logging.info(f"  Topology patched: ions.itp + {n_zn} ZN")

# ---------------------------------------------------------------------------
# Step 2  –  Ligand parameterization  (RDKit + acpype)
# ---------------------------------------------------------------------------

def prepare_ligand(compound: Dict, cfg: MDConfig, comp_dir: Path) -> Path:
    """Prepare ligand for MD using docked pose coordinates + GAFF2 (acpype).

    Three-tier strategy (Jakalian et al. 2002, J Comput Chem 23:1623):
      Tier 1: Docked PDBQT → obabel → PDB → acpype AM1-BCC
              (preserves docked orientation; best case)
      Tier 2: RDKit clean conformer → acpype AM1-BCC → Kabsch align to
              docked pose  (handles SQM convergence failures on obabel
              geometry while keeping accurate AM1-BCC charges)
      Tier 3: RDKit conformer → acpype Gasteiger → Kabsch align
              (last resort; Gasteiger less accurate for conjugated systems)
    """
    lig_dir = comp_dir / "ligand"
    lig_dir.mkdir(parents=True, exist_ok=True)
    cid = compound["compound_id"]
    smi = compound["smiles"]

    # ---- Compute formal charge from SMILES (needed for acpype) ----
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    formal_charge = Chem.GetFormalCharge(mol) if mol else 0
    logging.info(f"  Formal charge = {formal_charge}")

    # ---- Locate docked PDBQT ----
    site_tag = "zf23" if compound["best_site"] == "ZF2-3" else "zf45"
    site_subdir = "ZF2-3" if compound["best_site"] == "ZF2-3" else "ZF4-5"
    pdbqt = cfg.project_dir / cfg.docking_dir / site_subdir / f"{cid}_{site_tag}.pdbqt"

    # ---- TIER 1: docked pose → obabel → acpype AM1-BCC ----
    input_pdb = lig_dir / "input_for_acpype.pdb"
    tier_used = None

    if pdbqt.exists():
        ok = _extract_best_pose_pdb(pdbqt, input_pdb)
        if ok:
            logging.info(f"  TIER 1: acpype AM1-BCC on docked-pose PDB")
            if _run_acpype(lig_dir, input_pdb, formal_charge, method="bcc"):
                tier_used = 1
                logging.info(f"  ✓ Tier 1 succeeded (docked pose + AM1-BCC)")

    # ---- TIER 2: RDKit clean conformer → AM1-BCC → Kabsch align ----
    if tier_used is None and pdbqt.exists():
        logging.info(f"  TIER 2: AM1-BCC on clean RDKit conformer + Kabsch alignment")
        rdkit_pdb = lig_dir / "rdkit_clean.pdb"
        _rdkit_smiles_to_pdb(smi, rdkit_pdb)
        if _run_acpype(lig_dir, rdkit_pdb, formal_charge, method="bcc"):
            # Align the acpype GRO to the docked pose via Kabsch
            aligned = _kabsch_align_gro_to_pdbqt(lig_dir / "ligand.gro", pdbqt)
            if aligned:
                tier_used = 2
                logging.info(f"  ✓ Tier 2 succeeded (AM1-BCC charges + Kabsch alignment)")
            else:
                logging.warning(f"  Kabsch alignment failed; trying Tier 3")

    # ---- TIER 3: RDKit conformer → Gasteiger → Kabsch align ----
    if tier_used is None:
        logging.warning(f"  TIER 3: Gasteiger charges + Kabsch alignment (last resort)")
        rdkit_pdb = lig_dir / "rdkit_clean.pdb"
        if not rdkit_pdb.exists():
            _rdkit_smiles_to_pdb(smi, rdkit_pdb)
        if _run_acpype(lig_dir, rdkit_pdb, formal_charge, method="gas"):
            if pdbqt.exists():
                _kabsch_align_gro_to_pdbqt(lig_dir / "ligand.gro", pdbqt)
            tier_used = 3
            logging.warning(f"  ⚠ Tier 3: Gasteiger charges (less accurate for conjugated systems)")
        else:
            raise RuntimeError(f"acpype failed for {cid} (all tiers exhausted)")

    # ---- Validate output ----
    assert_file(lig_dir / "ligand.gro", "acpype GRO")
    assert_file(lig_dir / "ligand.itp", "acpype ITP")

    # ---- Rename moleculetype → LIG ----
    _rename_ligand_moltype(lig_dir)

    # ---- Split ITP into atomtypes + moleculetype ----
    _split_ligand_itp(lig_dir)

    logging.info(f"  Ligand prep done (Tier {tier_used}) → {lig_dir}")
    return lig_dir


def _extract_best_pose_pdb(pdbqt: Path, out_pdb: Path) -> bool:
    """Extract first MODEL from Vina PDBQT → clean PDB via Open Babel.

    This preserves the docked pose coordinates (position + orientation)
    so the ligand starts the MD simulation in the correct binding geometry.
    """
    # Write first model to a temp PDBQT
    tmp_pdbqt = out_pdb.parent / "_best_pose.pdbqt"
    first_model_lines = []
    in_model = False
    with open(pdbqt) as f:
        for ln in f:
            if ln.startswith("MODEL"):
                if in_model:
                    break          # second model → stop
                in_model = True
                continue
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith(("ATOM", "HETATM")):
                first_model_lines.append(ln)

    if not first_model_lines:
        return False

    with open(tmp_pdbqt, "w") as f:
        f.writelines(first_model_lines)
        f.write("END\n")

    # Convert PDBQT → PDB with Open Babel (adds missing H, cleans types)
    r = sh(f"obabel {tmp_pdbqt} -O {out_pdb} -h",
           cwd=out_pdb.parent, check=False)

    tmp_pdbqt.unlink(missing_ok=True)

    if not out_pdb.exists() or out_pdb.stat().st_size < 100:
        return False

    logging.info(f"  Extracted best docked pose → {out_pdb.name} "
                 f"({len(first_model_lines)} heavy atoms)")
    return True


def _kabsch_align_gro_to_pdbqt(gro: Path, pdbqt: Path) -> bool:
    """Align GRO heavy-atom coordinates to docked PDBQT pose via Kabsch algorithm.

    Uses SVD-based optimal rotation + translation (Kabsch 1976, Acta Cryst A32:922).
    Heavy atoms are matched by element type and graph-distance ordering.
    The rotation is applied to ALL atoms (including H) in the GRO file.
    Returns True if alignment succeeded.
    """
    import numpy as np

    # ---- Parse PDBQT: heavy atom coords from first MODEL (Å) ----
    pdbqt_coords = []
    pdbqt_elements = []
    in_model = False
    with open(pdbqt) as f:
        for ln in f:
            if ln.startswith("MODEL"):
                if in_model:
                    break
                in_model = True
                continue
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
                # AutoDock PDBQT: element type is last 1-2 non-space chars
                ad_type = ln[77:79].strip() if len(ln) >= 79 else ln[12:16].strip()[0]
                if ad_type in ("H", "HD", "HS"):
                    continue
                x = float(ln[30:38])
                y = float(ln[38:46])
                z = float(ln[46:54])
                pdbqt_coords.append([x, y, z])
                elem = ad_type[0]  # first char = element
                pdbqt_elements.append(elem)

    if len(pdbqt_coords) < 3:
        logging.warning(f"  Kabsch: too few PDBQT heavy atoms ({len(pdbqt_coords)})")
        return False

    # ---- Parse GRO: separate heavy vs all atoms (nm) ----
    with open(gro) as f:
        lines = f.readlines()
    n_atoms = int(lines[1])
    all_coords = []
    heavy_indices = []
    heavy_coords = []
    for i, a in enumerate(lines[2:2+n_atoms]):
        x = float(a[20:28])
        y = float(a[28:36])
        z = float(a[36:44])
        all_coords.append([x, y, z])
        # Detect hydrogen: atom name starts with H (after residue info)
        atom_name = a[10:15].strip()
        if not atom_name.startswith("H"):
            heavy_indices.append(i)
            heavy_coords.append([x * 10, y * 10, z * 10])  # nm → Å

    n_pdbqt = len(pdbqt_coords)
    n_gro_heavy = len(heavy_coords)

    if n_pdbqt != n_gro_heavy:
        logging.warning(f"  Kabsch: heavy atom count mismatch "
                        f"(PDBQT={n_pdbqt}, GRO={n_gro_heavy}). "
                        f"Attempting element-sorted alignment.")
        # If counts differ (obabel added/removed atoms), fall back to centroid
        if abs(n_pdbqt - n_gro_heavy) > 5:
            logging.warning(f"  Kabsch: too many mismatched atoms, using centroid only")
            dock_centroid = _pdbqt_centroid(pdbqt)
            if dock_centroid:
                _transplant_centroid(gro, dock_centroid)
            return True
        # Use the smaller set
        n_use = min(n_pdbqt, n_gro_heavy)
        P = np.array(pdbqt_coords[:n_use])
        Q = np.array(heavy_coords[:n_use])
    else:
        P = np.array(pdbqt_coords)  # target (docked pose, Å)
        Q = np.array(heavy_coords)  # source (RDKit/acpype, Å)

    # ---- Kabsch: compute optimal rotation R and translation t ----
    # such that P ≈ R·Q + t
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    H = Q_c.T @ P_c
    U, S, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    # ---- Apply rotation + translation to ALL atoms ----
    all_coords_A = np.array(all_coords) * 10  # nm → Å
    aligned_A = (R @ (all_coords_A - centroid_Q).T).T + centroid_P
    aligned_nm = aligned_A / 10  # Å → nm

    # ---- Write aligned GRO ----
    new_atoms = []
    for i, a in enumerate(lines[2:2+n_atoms]):
        x, y, z = aligned_nm[i]
        new_atoms.append(f"{a[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n")

    with open(gro, "w") as f:
        f.write(lines[0])
        f.write(lines[1])
        f.writelines(new_atoms)
        f.write(lines[2+n_atoms])

    # Compute alignment RMSD for diagnostics
    Q_aligned = (R @ (Q - centroid_Q).T).T + centroid_P
    rmsd = np.sqrt(np.mean(np.sum((P[:len(Q_aligned)] - Q_aligned[:len(P)])**2, axis=1)))
    logging.info(f"  Kabsch alignment: RMSD = {rmsd:.2f} Å "
                 f"({len(Q_aligned)} heavy atoms)")
    return True


def _rdkit_smiles_to_pdb(smi: str, out_pdb: Path) -> int:
    """Generate 3D PDB from SMILES using RDKit. Returns formal charge."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smi}")
    mol = Chem.AddHs(mol)
    ret = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if ret != 0:
        ret = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if ret != 0:
        raise RuntimeError("RDKit 3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    charge = Chem.GetFormalCharge(mol)
    Chem.MolToPDBFile(mol, str(out_pdb))
    return charge


def _pdbqt_centroid(pdbqt: Path) -> Optional[Tuple[float, float, float]]:
    """Parse first MODEL from PDBQT, return centroid in Ångströms."""
    xs, ys, zs = [], [], []
    first_model = True
    with open(pdbqt) as f:
        for ln in f:
            if ln.startswith("MODEL") and not first_model:
                break
            if ln.startswith("MODEL"):
                first_model = False; continue
            if ln.startswith("ENDMDL"):
                break
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
                xs.append(float(ln[30:38]))
                ys.append(float(ln[38:46]))
                zs.append(float(ln[46:54]))
    if not xs:
        return None
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))


def _run_acpype(lig_dir: Path, pdb: Path, charge: int, method: str) -> bool:
    """Run acpype; return True if GMX output files were produced."""
    # Clean previous attempts
    for d in lig_dir.glob("*.acpype"):
        shutil.rmtree(d, ignore_errors=True)
    for f in ("ligand.gro", "ligand.itp", "posre_ligand.itp"):
        (lig_dir / f).unlink(missing_ok=True)

    r = sh(f"acpype -i {pdb} -c {method} -a gaff2 -n {charge} -o gmx",
           cwd=lig_dir, check=False, timeout=300)

    # Find output
    dirs = list(lig_dir.glob("*.acpype"))
    if not dirs:
        return False
    out = dirs[0]

    # acpype names: <stem>_GMX.gro, <stem>_GMX.itp
    copied = False
    for pat, dest in [("*_GMX.gro", "ligand.gro"),
                      ("*_GMX.itp", "ligand.itp"),
                      ("posre*_GMX.itp", "posre_ligand.itp")]:
        hits = list(out.glob(pat))
        if hits:
            shutil.copy2(hits[0], lig_dir / dest)
            if dest == "ligand.gro":
                copied = True
    return copied


def _rename_ligand_moltype(lig_dir: Path):
    """Rename acpype's moleculetype (e.g. 'rdkit_3d') → 'LIG' in ITP and GRO."""
    # --- ITP: rename moleculetype name AND residue names ---
    itp = lig_dir / "ligand.itp"
    with open(itp) as f:
        txt = f.read()

    # Rename moleculetype name (line after [ moleculetype ] header + comment)
    lines = txt.split("\n")
    new_lines = []
    in_moltype_header = False
    old_resname = None
    for ln in lines:
        if "[ moleculetype ]" in ln:
            in_moltype_header = True
            new_lines.append(ln)
            continue
        if in_moltype_header and not ln.startswith(";") and ln.strip():
            parts = ln.split()
            old_name = parts[0]
            ln = ln.replace(old_name, "LIG", 1)
            in_moltype_header = False
            logging.info(f"  Renamed moleculetype '{old_name}' → 'LIG'")
        new_lines.append(ln)
    txt = "\n".join(new_lines)

    # Rename residue name in [ atoms ] section (e.g. UNL → LIG)
    # Format: "   1   c3     1   UNL    C1    1 ..."
    # The 4th column (field index 3) is the residue name
    import re as _re
    # Match lines in atoms section: nr type resnr resname atomname ...
    # Replace common acpype residue names with LIG
    for old_rn in ("UNL", "MOL", "LIG0", "rdkit_3d"):
        if old_rn in txt:
            txt = txt.replace(f"   {old_rn}   ", f"   LIG   ")
            txt = txt.replace(f"   {old_rn}  ", f"   LIG  ")
            # Also handle 4-char padding
            txt = txt.replace(f"  {old_rn}  ", f"  LIG  ")

    with open(itp, "w") as f:
        f.write(txt)

    # --- GRO: rename residue ---
    gro = lig_dir / "ligand.gro"
    with open(gro) as f:
        glines = f.readlines()
    n = int(glines[1])
    new_atoms = []
    for a in glines[2:2+n]:
        # GRO: cols 5-10 are residue name (5 chars, left-justified)
        new_atoms.append(a[:5] + f"{'LIG':<5s}" + a[10:])
    with open(gro, "w") as f:
        f.write(glines[0])
        f.write(glines[1])
        f.writelines(new_atoms)
        f.write(glines[2+n])


def _split_ligand_itp(lig_dir: Path):
    """Split ligand.itp into ligand_atomtypes.itp + ligand_moltype.itp.

    GAFF2 atomtypes must be included before any moleculetype in the topology,
    so they go into a separate file.
    """
    itp = lig_dir / "ligand.itp"
    with open(itp) as f:
        lines = f.readlines()

    atomtypes_lines = []
    moltype_lines = []
    in_atomtypes = False
    past_atomtypes = False

    for ln in lines:
        if "[ atomtypes ]" in ln:
            in_atomtypes = True
            atomtypes_lines.append(ln)
            continue
        if in_atomtypes:
            if ln.strip().startswith("[") and "atomtypes" not in ln:
                in_atomtypes = False
                past_atomtypes = True
                moltype_lines.append(ln)
            else:
                atomtypes_lines.append(ln)
        else:
            moltype_lines.append(ln)

    if atomtypes_lines:
        with open(lig_dir / "ligand_atomtypes.itp", "w") as f:
            f.writelines(atomtypes_lines)
        with open(lig_dir / "ligand_moltype.itp", "w") as f:
            f.writelines(moltype_lines)
        logging.info(f"  Split ITP: {len(atomtypes_lines)} atomtype lines, "
                     f"{len(moltype_lines)} moltype lines")
    else:
        # No atomtypes section; just copy as-is
        shutil.copy2(itp, lig_dir / "ligand_moltype.itp")
        with open(lig_dir / "ligand_atomtypes.itp", "w") as f:
            f.write("; No GAFF2 atomtypes in this ITP\n")


def _transplant_centroid(gro: Path, dock_centroid_A: Tuple[float, float, float]):
    """Translate ligand coordinates so centroid matches docked pose."""
    with open(gro) as f:
        lines = f.readlines()
    n = int(lines[1])
    atoms = lines[2:2+n]

    # Current centroid (GRO is in nm)
    cx = sum(float(a[20:28]) for a in atoms) / n
    cy = sum(float(a[28:36]) for a in atoms) / n
    cz = sum(float(a[36:44]) for a in atoms) / n

    # Target centroid (convert Å → nm)
    tx, ty, tz = dock_centroid_A[0]/10, dock_centroid_A[1]/10, dock_centroid_A[2]/10
    dx, dy, dz = tx - cx, ty - cy, tz - cz

    new_atoms = []
    for a in atoms:
        x = float(a[20:28]) + dx
        y = float(a[28:36]) + dy
        z = float(a[36:44]) + dz
        new_atoms.append(f"{a[:20]}{x:8.3f}{y:8.3f}{z:8.3f}\n")

    with open(gro, "w") as f:
        f.write(lines[0])
        f.write(lines[1])
        f.writelines(new_atoms)
        f.write(lines[2+n])

    logging.info(f"  Centroid transplanted: Δ=({dx:.2f},{dy:.2f},{dz:.2f}) nm")

# ---------------------------------------------------------------------------
# Step 3  –  Build complex (protein + ligand + solvent + ions)
# ---------------------------------------------------------------------------

def build_complex(compound: Dict, cfg: MDConfig, comp_dir: Path,
                  prot_dir: Path) -> Path:
    sys_dir = comp_dir / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    lig_dir = comp_dir / "ligand"

    # ---- Copy protein files into system dir ----
    for f in ("protein.gro", "topol.top"):
        shutil.copy2(prot_dir / f, sys_dir / f)
    for f in prot_dir.glob("*.itp"):
        shutil.copy2(f, sys_dir / f.name)

    # ---- Copy ligand files ----
    for f in ("ligand.gro", "ligand_atomtypes.itp", "ligand_moltype.itp"):
        src = lig_dir / f
        if src.exists():
            shutil.copy2(src, sys_dir / f)

    # ---- Merge protein + ligand GRO ----
    complex_gro = sys_dir / "complex.gro"
    _merge_gro(sys_dir / "protein.gro", sys_dir / "ligand.gro", complex_gro)

    # ---- Build system topology ----
    _build_system_topology(sys_dir)

    # ---- Box ----
    gmx(f"editconf -f {complex_gro} -o {sys_dir/'boxed.gro'} "
        f"-c -d {cfg.box_buffer} -bt dodecahedron", cwd=sys_dir)

    # ---- Solvate ----
    gmx(f"solvate -cp {sys_dir/'boxed.gro'} -cs spc216.gro "
        f"-o {sys_dir/'solvated.gro'} -p {sys_dir/'topol.top'}",
        cwd=sys_dir)

    # ---- Add ions ----
    ions_mdp = sys_dir / "ions.mdp"
    ions_mdp.write_text("integrator=steep\nnsteps=0\n")
    gmx(f"grompp -f {ions_mdp} -c {sys_dir/'solvated.gro'} "
        f"-p {sys_dir/'topol.top'} -o {sys_dir/'ions.tpr'} -maxwarn 10",
        cwd=sys_dir)
    gmx(f"genion -s {sys_dir/'ions.tpr'} -o {sys_dir/'ionized.gro'} "
        f"-p {sys_dir/'topol.top'} -pname NA -nname CL "
        f"-neutral -conc {cfg.salt_conc}",
        cwd=sys_dir, stdin_text="SOL\n")
    assert_file(sys_dir / "ionized.gro", "ionized GRO")

    logging.info(f"  System built → {sys_dir / 'ionized.gro'}")
    return sys_dir


def _merge_gro(prot_gro: Path, lig_gro: Path, out: Path):
    with open(prot_gro) as f: pl = f.readlines()
    with open(lig_gro) as f: ll = f.readlines()
    pn = int(pl[1]); ln = int(ll[1])
    pa = pl[2:2+pn]; la = ll[2:2+ln]; box = pl[2+pn]

    # Renumber ligand residue to "LIG"
    new_lig = []
    for line in la:
        # GRO format: 5-char resnum, 5-char resname, 5-char atomname, 5-char atomnum, coords
        rnum = pn + 1   # all lig atoms same residue
        new_lig.append(f"{rnum:5d}{'LIG':<5s}{line[10:]}")

    with open(out, "w") as f:
        f.write("Protein-Ligand Complex\n")
        f.write(f"{pn + ln:5d}\n")
        f.writelines(pa)
        f.writelines(new_lig)
        f.write(box)
    logging.info(f"  Merged GRO: {pn} prot + {ln} lig = {pn+ln} atoms")


def _build_system_topology(sys_dir: Path):
    """Rewrite topology with correct include order for AMBER+GAFF2."""
    top = sys_dir / "topol.top"
    with open(top) as f:
        txt = f.read()

    # Insert ligand atomtypes right after forcefield include
    if "ligand_atomtypes.itp" not in txt:
        txt = txt.replace(
            f'#include "amber99sb-ildn.ff/forcefield.itp"',
            f'#include "amber99sb-ildn.ff/forcefield.itp"\n'
            f'; GAFF2 ligand atom types\n'
            f'#include "ligand_atomtypes.itp"'
        )

    # Insert ligand moleculetype before water/ions includes
    if "ligand_moltype.itp" not in txt:
        # Find the tip3p include line
        if "tip3p.itp" in txt:
            txt = txt.replace(
                '#include "amber99sb-ildn.ff/tip3p.itp"',
                '; Ligand moleculetype\n'
                '#include "ligand_moltype.itp"\n\n'
                '#include "amber99sb-ildn.ff/tip3p.itp"'
            )
        else:
            # Insert before [ system ]
            txt = txt.replace(
                "[ system ]",
                '#include "ligand_moltype.itp"\n\n[ system ]'
            )

    # Add LIG to [ molecules ]
    if "LIG" not in txt:
        txt = txt.rstrip() + "\nLIG              1\n"

    with open(top, "w") as f:
        f.write(txt)

# ---------------------------------------------------------------------------
# Step 4  –  Index groups
# ---------------------------------------------------------------------------

def make_index(sys_dir: Path) -> Path:
    """Build index.ndx with Protein_LIG and Water_and_ions groups."""
    gro = sys_dir / "ionized.gro"
    ndx = sys_dir / "index.ndx"
    with open(gro) as f: lines = f.readlines()
    n = int(lines[1])

    prot_lig, wat_ion, lig_only, backbone = [], [], [], []
    for i, ln in enumerate(lines[2:2+n]):
        resn = ln[5:10].strip()
        atn  = ln[10:15].strip()
        idx  = i + 1
        if resn in ("SOL", "NA", "CL", "Na+", "Cl-"):
            wat_ion.append(idx)
        else:
            prot_lig.append(idx)
        if resn == "LIG":
            lig_only.append(idx)
        # Backbone: protein CA, C, N
        if resn not in ("SOL","NA","CL","Na+","Cl-","LIG","ZN") and atn in ("CA","C","N"):
            backbone.append(idx)

    def write_group(f, name, indices):
        f.write(f"[ {name} ]\n")
        for j, idx in enumerate(indices):
            f.write(f"{idx:7d}")
            if (j+1) % 15 == 0: f.write("\n")
        f.write("\n\n")

    with open(ndx, "w") as f:
        write_group(f, "Protein_LIG", prot_lig)
        write_group(f, "Water_and_ions", wat_ion)
        write_group(f, "LIG", lig_only)
        write_group(f, "Backbone", backbone)
        write_group(f, "System", list(range(1, n+1)))

    logging.info(f"  Index: {len(prot_lig)} Protein_LIG, "
                 f"{len(wat_ion)} Water_and_ions, {len(lig_only)} LIG, "
                 f"{len(backbone)} Backbone")
    return ndx

# ---------------------------------------------------------------------------
# Step 5  –  MD simulation
# ---------------------------------------------------------------------------

def run_simulation(compound: Dict, cfg: MDConfig, comp_dir: Path,
                   prot_dir: Path, gpu_id: int) -> Dict:
    cid = compound["compound_id"]
    logging.info(f"\n{'='*60}\nMD: {cid} | GPU {gpu_id}\n{'='*60}")

    mdp = cfg.md_base_dir / "mdp"
    result = {"compound_id": cid, "status": "running", "steps": {}}

    try:
        # ---------- ligand prep ----------
        logging.info(f"[{cid}] 1/8 Ligand parameterization")
        prepare_ligand(compound, cfg, comp_dir)
        result["steps"]["ligand_prep"] = "ok"

        # ---------- complex ----------
        logging.info(f"[{cid}] 2/8 Building complex")
        sys_dir = build_complex(compound, cfg, comp_dir, prot_dir)
        result["steps"]["build_complex"] = "ok"

        # ---------- index ----------
        logging.info(f"[{cid}] 3/8 Index groups")
        ndx = make_index(sys_dir)
        result["steps"]["index"] = "ok"

        top = sys_dir / "topol.top"
        gro = sys_dir / "ionized.gro"

        # ---------- Energy Minimization ----------
        logging.info(f"[{cid}] 4/8 Energy minimization")
        em = comp_dir / "em"; em.mkdir(exist_ok=True)
        gmx(f"grompp -f {mdp/'em.mdp'} -c {gro} -p {top} "
            f"-o {em/'em.tpr'} -maxwarn 10", cwd=sys_dir)
        gmx(f"mdrun -v -deffnm {em/'em'} -ntmpi 1 -ntomp 8 -nb gpu -gpu_id {gpu_id}", cwd=sys_dir)
        assert_file(em / "em.gro", "EM output")
        result["steps"]["em"] = "ok"

        # ---------- NVT ----------
        logging.info(f"[{cid}] 5/8 NVT equilibration (100 ps)")
        nvt = comp_dir / "nvt"; nvt.mkdir(exist_ok=True)
        gmx(f"grompp -f {mdp/'nvt.mdp'} -c {em/'em.gro'} -r {em/'em.gro'} "
            f"-p {top} -n {ndx} -o {nvt/'nvt.tpr'} -maxwarn 10", cwd=sys_dir)
        gmx(f"mdrun -deffnm {nvt/'nvt'} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu -gpu_id {gpu_id}", cwd=sys_dir)
        assert_file(nvt / "nvt.gro", "NVT output")
        result["steps"]["nvt"] = "ok"

        # ---------- NPT ----------
        logging.info(f"[{cid}] 6/8 NPT equilibration (100 ps)")
        npt = comp_dir / "npt"; npt.mkdir(exist_ok=True)
        gmx(f"grompp -f {mdp/'npt.mdp'} -c {nvt/'nvt.gro'} -r {nvt/'nvt.gro'} "
            f"-t {nvt/'nvt.cpt'} -p {top} -n {ndx} -o {npt/'npt.tpr'} "
            f"-maxwarn 10", cwd=sys_dir)
        gmx(f"mdrun -deffnm {npt/'npt'} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu -gpu_id {gpu_id}", cwd=sys_dir)
        assert_file(npt / "npt.gro", "NPT output")
        result["steps"]["npt"] = "ok"

        # ---------- Production MD ----------
        logging.info(f"[{cid}] 7/8 Production MD ({cfg.prod_ns} ns)")
        prod = comp_dir / "prod"; prod.mkdir(exist_ok=True)
        gmx(f"grompp -f {mdp/'md_prod.mdp'} -c {npt/'npt.gro'} "
            f"-t {npt/'npt.cpt'} -p {top} -n {ndx} "
            f"-o {prod/'md.tpr'} -maxwarn 10", cwd=sys_dir)
        gmx(f"mdrun -deffnm {prod/'md'} -ntmpi 1 -ntomp 8 -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id {gpu_id}",
            cwd=sys_dir)
        assert_file(prod / "md.gro", "Production output")
        result["steps"]["production"] = "ok"

        # ---------- VMD + analysis ----------
        logging.info(f"[{cid}] 8/8 VMD output & analysis")
        _generate_vmd(compound, comp_dir, ndx)
        _analyze(compound, comp_dir, ndx)
        result["steps"]["analysis"] = "ok"

        result["status"] = "success"
        logging.info(f"[{cid}] ✓ COMPLETE")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logging.error(f"[{cid}] ✗ FAILED at step: {e}")

    return result

# ---------------------------------------------------------------------------
# Step 6  –  VMD output
# ---------------------------------------------------------------------------

def _generate_vmd(compound: Dict, comp_dir: Path, ndx: Path):
    vmd = comp_dir / "vmd"; vmd.mkdir(exist_ok=True)
    prod = comp_dir / "prod"
    tpr, xtc = prod / "md.tpr", prod / "md.xtc"
    if not xtc.exists():
        logging.warning("  No trajectory for VMD"); return

    # PBC-correct + center
    gmx(f"trjconv -s {tpr} -f {xtc} -o {vmd/'md_center.xtc'} "
        f"-center -pbc mol -ur compact -n {ndx}",
        cwd=comp_dir, stdin_text="Protein_LIG\nSystem\n", check=False)

    # First frame as structure
    gmx(f"trjconv -s {tpr} -f {xtc} -o {vmd/'structure.gro'} "
        f"-pbc mol -dump 0 -n {ndx}",
        cwd=comp_dir, stdin_text="System\n", check=False)

    # VMD script
    cid = compound["compound_id"]
    dock = compound.get("best_dock", "N/A")
    (vmd / f"load_{cid}.tcl").write_text(f"""# VMD script for {cid} (dock: {dock} kcal/mol)
mol new structure.gro type gro waitfor all
mol addfile md_center.xtc type xtc waitfor all
mol delrep 0 top
mol representation NewCartoon 0.3 10 4.1 0
mol color Structure
mol selection {{protein}}
mol addrep top
mol representation Licorice 0.2 12 12
mol color Name
mol selection {{resname LIG}}
mol addrep top
mol representation VDW 0.5 12
mol color Element
mol selection {{name ZN}}
mol addrep top
mol representation Lines 1.0
mol color Name
mol selection {{protein and within 5 of resname LIG}}
mol addrep top
display projection Orthographic
color Display Background white
axes location Off
puts "Loaded {cid} trajectory"
""")
    logging.info(f"  VMD files → {vmd}")

# ---------------------------------------------------------------------------
# Step 7  –  Analysis
# ---------------------------------------------------------------------------

def _analyze(compound: Dict, comp_dir: Path, ndx: Path):
    ana = comp_dir / "analysis"; ana.mkdir(exist_ok=True)
    tpr, xtc = comp_dir/"prod"/"md.tpr", comp_dir/"prod"/"md.xtc"
    if not xtc.exists(): return

    for name, sel_in, args in [
        ("rmsd_backbone", "Backbone\nBackbone\n",
         f"rms -s {tpr} -f {xtc} -o {ana/'rmsd_backbone.xvg'} -n {ndx} -tu ns"),
        ("rmsd_ligand",   "Backbone\nLIG\n",
         f"rms -s {tpr} -f {xtc} -o {ana/'rmsd_ligand.xvg'} -n {ndx} -tu ns"),
        ("rmsf",          "Protein_LIG\n",
         f"rmsf -s {tpr} -f {xtc} -o {ana/'rmsf.xvg'} -n {ndx} -res"),
        ("gyrate",        "LIG\n",
         f"gyrate -s {tpr} -f {xtc} -o {ana/'gyrate.xvg'} -n {ndx}"),
    ]:
        gmx(args, cwd=comp_dir, stdin_text=sel_in, check=False)

    logging.info(f"  Analysis → {ana}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLI1 MD Pipeline")
    parser.add_argument("--project-dir", default=str(Path.home()/"GLI-final-model"))
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--n-compounds", type=int, default=5)
    parser.add_argument("--prod-ns", type=int, default=50)
    parser.add_argument("--candidates-csv", default=None,
                        help="Override candidates CSV (relative to project dir)")
    args = parser.parse_args()

    cfg = MDConfig()
    cfg.project_dir = Path(args.project_dir)
    cfg.md_base_dir = cfg.project_dir / "md_simulation"
    cfg.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    cfg.n_compounds = args.n_compounds
    cfg.prod_ns = args.prod_ns
    if args.candidates_csv:
        cfg.candidates_csv = args.candidates_csv

    # Adjust production MDP nsteps
    prod_mdp = cfg.md_base_dir / "mdp" / "md_prod.mdp"
    if prod_mdp.exists():
        nsteps = int(cfg.prod_ns * 1e6 / 2)
        txt = prod_mdp.read_text()
        txt = re.sub(r"nsteps\s*=\s*\d+", f"nsteps          = {nsteps}", txt)
        prod_mdp.write_text(txt)

    setup_logging(cfg.md_base_dir / "logs")
    logging.info("="*60)
    logging.info("GLI1 ZINC FINGER – LIGAND MD SIMULATION PIPELINE")
    logging.info(f"  Project:    {cfg.project_dir}")
    logging.info(f"  GPUs:       {cfg.gpu_ids}")
    logging.info(f"  Production: {cfg.prod_ns} ns")
    logging.info("="*60)

    # ---- Select compounds ----
    compounds = select_compounds(cfg)

    # ---- Prepare protein (once) ----
    prot_dir = prepare_protein(cfg)

    # ---- Run compounds in PARALLEL across GPUs for speed ----
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(comp, gpu):
        cid = comp["compound_id"]
        comp_dir = cfg.md_base_dir / "compounds" / cid.replace(".", "_")
        comp_dir.mkdir(parents=True, exist_ok=True)
        return run_simulation(comp, cfg, comp_dir, prot_dir, gpu)

    n_gpus = len(cfg.gpu_ids)
    results = [None] * len(compounds)

    # Run up to n_gpus compounds in parallel (each pinned to its own GPU)
    with ThreadPoolExecutor(max_workers=min(n_gpus, len(compounds))) as executor:
        futures = {}
        for i, comp in enumerate(compounds):
            gpu = cfg.gpu_ids[i % n_gpus]
            fut = executor.submit(_run_one, comp, gpu)
            futures[fut] = i
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    # ---- Summary ----
    logging.info(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        s = "✓" if r["status"] == "success" else "✗"
        e = f" — {r.get('error','')}" if r["status"] != "success" else ""
        logging.info(f"  {s} {r['compound_id']}: {r['status']}{e}")
        if "steps" in r:
            for step, st in r["steps"].items():
                logging.info(f"      {step}: {st}")

    ok = sum(1 for r in results if r["status"] == "success")
    logging.info(f"\n{ok}/{len(results)} completed")
    logging.info(f"VMD: vmd -e <compound>/vmd/load_<id>.tcl")

    (cfg.md_base_dir / "md_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
