#!/usr/bin/env python3
"""
GLI1-DNA Ternary Complex MD Pipeline
=====================================
Runs 200 ns MD simulations of the GLI1-DNA complex (PDB 2GLI) with and
without small-molecule inhibitors bound.

Force fields:
  - Protein:  AMBER99SB-ILDN (Lindorff-Larsen et al., 2010)
  - DNA:      bsc0 corrections (included in amber99sb-ildn)
  - Ligand:   GAFF2 + AM1-BCC via acpype (Wang et al., 2004)
  - Zinc:     ZAFF bonded model (Peters et al., 2010, JCTC 6:2935)
  - Water:    TIP3P, 0.15 M NaCl

Systems:
  1. Control:  GLI1-DNA (no inhibitor)
  2. Ternary:  GLI1-DNA + CNP0592286 (ZF4-5, best v3 binder)
  3. Ternary:  GLI1-DNA + CNP0544084 (ZF2-3, #2 v3 binder)
  4. Ternary:  GLI1-DNA + CNP0214725 (ZF2-3, BBB-permeable lead)
"""

import os
import sys
import csv
import json
import shutil
import logging
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ternary_pipeline.log"),
        logging.StreamHandler()
    ]
)

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TernaryConfig:
    """Configuration for ternary complex MD simulations."""
    project_dir: Path = Path("/home/sanjanp/GLI-final-model")
    ternary_base: Path = Path("/home/sanjanp/GLI-final-model/ternary_md")
    pdb_file: str = "2GLI.pdb"

    # Force field
    ff: str = "amber99sb-ildn"
    water: str = "tip3p"

    # Simulation parameters
    box_buffer: float = 1.2       # nm
    ion_conc: float = 0.15        # M NaCl
    em_steps: int = 50000
    nvt_ps: int = 200             # ps
    npt_ps: int = 200             # ps
    prod_ns: int = 200            # ns
    dt_fs: int = 2                # fs

    # Docking
    vina_exhaustiveness: int = 128
    vina_n_modes: int = 20

    # Docking box centers (approximate, from 2GLI structure)
    # ZF2-3 interface (phosphate backbone contacts)
    zf23_center: Tuple[float, float, float] = (-18.0, 5.0, 0.0)
    zf23_size: Tuple[float, float, float] = (24.0, 24.0, 24.0)
    # ZF4-5 interface (base-specific major groove contacts)
    zf45_center: Tuple[float, float, float] = (-8.0, 15.0, 15.0)
    zf45_size: Tuple[float, float, float] = (24.0, 24.0, 24.0)


# Compounds to simulate
COMPOUNDS = {
    "CNP0592286": {
        "smiles": "COc1ccc(Cc2cc(=O)c3c(CC(C)=O)cc(O)cc3o2)c2c(=O)cc(C(C)C)oc12",
        "site": "ZF4-5",
        "label": "Novel #1 (100% bound v3)",
        "dock_score_v3": -7.705,
    },
    "CNP0544084": {
        "smiles": "COc1cc(-c2c(O)cc(OC)c3c2ccc2cc(O)ccc23)cc2ccc3cc(O)ccc3c12",
        "site": "ZF2-3",
        "label": "Novel #2 (99% bound v3)",
        "dock_score_v3": -7.833,
    },
    "CNP0214725": {
        "smiles": "COc1cc(OC)c2c(c1)C(=O)c1cc(C)c(C)c(O)c1C2=O",
        "site": "ZF2-3",
        "label": "BBB Lead (MPO 5.76)",
        "dock_score_v3": -6.21,
    },
}

# =============================================================================
# Utility functions
# =============================================================================
def sh(cmd: str, cwd: Path = None, check: bool = True,
       stdin_text: str = None, timeout: int = None) -> subprocess.CompletedProcess:
    """Execute shell command with logging."""
    logging.info(f"  $ {cmd}")
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd,
                           input=stdin_text, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.error(f"  TIMEOUT ({timeout}s): {cmd}")
        raise
    if r.returncode != 0:
        logging.error(f"  STDOUT:\n{r.stdout[-3000:]}")
        logging.error(f"  STDERR:\n{r.stderr[-3000:]}")
        if check:
            raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd}")
    return r


def assert_file(path: Path, label: str = ""):
    """Verify file exists and is non-empty."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty: {path} ({label})")


# =============================================================================
# Step 1: Prepare PDB 2GLI
# =============================================================================
def prepare_pdb(cfg: TernaryConfig) -> Path:
    """Clean PDB 2GLI: replace Co→Zn, remove alternate conformations, etc."""
    raw_pdb = cfg.ternary_base / cfg.pdb_file
    clean_pdb = cfg.ternary_base / "2GLI_clean.pdb"

    logging.info("="*60)
    logging.info("STEP 1: Preparing PDB 2GLI")
    logging.info("="*60)

    assert_file(raw_pdb, "Raw PDB 2GLI")

    lines_out = []
    with open(raw_pdb) as f:
        for line in f:
            # Skip crystallographic waters (tleap will add explicit solvent)
            if line.startswith("HETATM") and "HOH" in line:
                continue
            # Replace cobalt with zinc (same coordination geometry)
            if line.startswith("HETATM") and " CO " in line and "CO  CO" in line:
                # Replace element symbol and residue name
                line = line.replace(" CO  CO ", " ZN  ZN ")
                line = line[:76] + "ZN  \n" if len(line) > 76 else line
                logging.info(f"  Co→Zn: {line.strip()}")
            # Remove alternate conformations (keep A only)
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                altloc = line[16]
                if altloc not in (' ', 'A', ''):
                    continue
                if altloc == 'A':
                    line = line[:16] + ' ' + line[17:]
            lines_out.append(line)

    with open(clean_pdb, 'w') as f:
        f.writelines(lines_out)

    logging.info(f"  Clean PDB written: {clean_pdb}")
    logging.info(f"  Total lines: {len(lines_out)}")

    # Verify zinc ions present
    zn_count = sum(1 for l in lines_out if 'ZN' in l and l.startswith('HETATM'))
    logging.info(f"  Zinc ions: {zn_count}")
    if zn_count != 5:
        logging.warning(f"  Expected 5 Zn ions for GLI1 ZF1-5, found {zn_count}")

    return clean_pdb


# =============================================================================
# Step 2: Generate GROMACS topology for protein-DNA complex
# =============================================================================
def build_protein_dna_topology(cfg: TernaryConfig, clean_pdb: Path) -> Tuple[Path, Path]:
    """Use pdb2gmx to generate topology for protein-DNA-Zn system."""
    logging.info("="*60)
    logging.info("STEP 2: Building protein-DNA topology (pdb2gmx)")
    logging.info("="*60)

    work_dir = cfg.ternary_base / "system_prep"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy clean PDB
    shutil.copy2(clean_pdb, work_dir / "input.pdb")

    # Run pdb2gmx with amber99sb-ildn
    # -ignh: ignore H atoms in input (let GROMACS add them)
    # -merge all: merge all chains into one molecule (protein + DNA)
    # Actually, we want separate chains. Let's NOT use -merge.
    gro_out = work_dir / "complex.gro"
    top_out = work_dir / "topol.top"

    sh(f"gmx pdb2gmx -f input.pdb -o complex.gro -p topol.top "
       f"-ff {cfg.ff} -water {cfg.water} -ignh -chainsep ter",
       cwd=work_dir, check=False, stdin_text="1\n")

    # Check if pdb2gmx succeeded
    if not gro_out.exists():
        logging.warning("pdb2gmx failed with -chainsep ter, trying interactive...")
        # Try with explicit chain handling
        sh(f"gmx pdb2gmx -f input.pdb -o complex.gro -p topol.top "
           f"-ff {cfg.ff} -water {cfg.water} -ignh",
           cwd=work_dir, check=True, stdin_text="1\n")

    assert_file(gro_out, "Complex GRO")
    assert_file(top_out, "Complex topology")

    logging.info(f"  Topology generated: {top_out}")
    return gro_out, top_out


# =============================================================================
# Step 3: Add ZAFF bonded zinc parameters
# =============================================================================
def add_zaff_zinc(cfg: TernaryConfig, top_file: Path):
    """Add ZAFF bonded zinc parameters to the GROMACS topology.

    ZAFF (Peters et al., 2010, JCTC 6:2935) provides bonded parameters
    for Zn²⁺ in Cys₂His₂ zinc fingers, preventing zinc escape during MD.

    For each zinc finger:
      Zn²⁺ is bonded to 2× Cys(Sγ) + 2× His(Nε2 or Nδ1)
      with bond lengths ~2.05 Å (Zn-S) and ~2.10 Å (Zn-N)
    """
    logging.info("="*60)
    logging.info("STEP 3: Adding ZAFF bonded zinc parameters")
    logging.info("="*60)

    # ZAFF parameters for Cys2His2 zinc fingers
    # Bond parameters (from Peters et al., 2010)
    zaff_params = """
; ZAFF bonded zinc parameters (Peters et al., JCTC 2010, 6:2935)
; Cys2His2 zinc finger coordination
[ bondtypes ]
; i    j    func   b0(nm)   kb(kJ/mol/nm2)
  ZN   SG   1      0.2180   83680.0    ; Zn-Cys(SG)
  ZN   NE2  1      0.2040   71128.0    ; Zn-His(NE2)
  ZN   ND1  1      0.2040   71128.0    ; Zn-His(ND1)

[ angletypes ]
; i    j    k    func   th0(deg)  kth(kJ/mol/rad2)
  SG   ZN   SG   1     115.0     418.4     ; S-Zn-S
  SG   ZN   NE2  1     109.5     418.4     ; S-Zn-N
  SG   ZN   ND1  1     109.5     418.4     ; S-Zn-N
  NE2  ZN   NE2  1     109.5     418.4     ; N-Zn-N
  NE2  ZN   ND1  1     109.5     418.4     ; N-Zn-N
  ND1  ZN   ND1  1     109.5     418.4     ; N-Zn-N
  CB   SG   ZN   1     109.5     418.4     ; CB-S-Zn
  CE1  NE2  ZN   1     126.0     418.4     ; CE1-NE2-Zn
  CD2  NE2  ZN   1     126.0     418.4     ; CD2-NE2-Zn
  CG   ND1  ZN   1     126.0     418.4     ; CG-ND1-Zn
  CE1  ND1  ZN   1     126.0     418.4     ; CE1-ND1-Zn
"""

    logging.info("  ZAFF parameters prepared for Cys2His2 coordination")
    logging.info("  Note: Explicit Zn-ligand bonds must be added to [ bonds ] section")
    logging.info("  in the topology after identifying coordinating residues from structure")

    # Write ZAFF parameter file
    zaff_file = top_file.parent / "zaff_zinc.itp"
    with open(zaff_file, 'w') as f:
        f.write(zaff_params)

    logging.info(f"  ZAFF parameters written: {zaff_file}")
    return zaff_file


# =============================================================================
# Step 4: Identify zinc coordination from structure
# =============================================================================
def identify_zinc_coordination(cfg: TernaryConfig, gro_file: Path) -> List[Dict]:
    """Parse GRO to identify Zn²⁺ coordinating residues (Cys SG, His NE2/ND1).

    Returns list of dicts with zinc index and coordinating atom indices.
    """
    logging.info("="*60)
    logging.info("STEP 4: Identifying zinc coordination sites")
    logging.info("="*60)

    # Read GRO file
    with open(gro_file) as f:
        lines = f.readlines()

    natoms = int(lines[1].strip())

    # Parse atoms
    atoms = []
    for i in range(2, 2 + natoms):
        line = lines[i]
        resnum = int(line[0:5].strip())
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomidx = int(line[15:20].strip())
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        atoms.append({
            'resnum': resnum, 'resname': resname, 'atomname': atomname,
            'idx': atomidx, 'x': x, 'y': y, 'z': z
        })

    # Find zinc atoms
    zn_atoms = [a for a in atoms if a['resname'] == 'ZN' or a['atomname'] == 'ZN']
    logging.info(f"  Found {len(zn_atoms)} zinc atoms")

    # For each zinc, find coordinating atoms within 2.5 Å (0.25 nm)
    coordinations = []
    for zn in zn_atoms:
        zn_pos = np.array([zn['x'], zn['y'], zn['z']])
        coord_atoms = []

        for a in atoms:
            if a['idx'] == zn['idx']:
                continue
            # Only look at CYS SG and HIS NE2/ND1
            if not ((a['resname'] in ('CYS', 'CYM') and a['atomname'] == 'SG') or
                    (a['resname'] in ('HIS', 'HID', 'HIE', 'HIP') and
                     a['atomname'] in ('NE2', 'ND1'))):
                continue

            dist = np.linalg.norm(np.array([a['x'], a['y'], a['z']]) - zn_pos)
            if dist < 0.30:  # 3.0 Å cutoff in nm
                coord_atoms.append({
                    'resnum': a['resnum'], 'resname': a['resname'],
                    'atomname': a['atomname'], 'idx': a['idx'],
                    'dist_nm': dist
                })
                logging.info(f"    Zn({zn['idx']}) ← {a['resname']}{a['resnum']}:{a['atomname']} "
                           f"dist={dist*10:.2f} Å")

        coordinations.append({
            'zn_idx': zn['idx'],
            'zn_resnum': zn['resnum'],
            'coord_atoms': coord_atoms,
            'n_coord': len(coord_atoms)
        })

        if len(coord_atoms) != 4:
            logging.warning(f"  Zn({zn['idx']}): expected 4 coordinating atoms, "
                          f"found {len(coord_atoms)}")

    return coordinations


# =============================================================================
# Step 5: Ligand preparation (GAFF2 + AM1-BCC)
# =============================================================================
def prepare_ligand(compound_id: str, smiles: str, cfg: TernaryConfig) -> Path:
    """Prepare ligand with GAFF2 + AM1-BCC charges via acpype.

    Uses RDKit for 3D conformer generation, then acpype for parameterization.
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"LIGAND PREP: {compound_id}")
    logging.info(f"{'='*60}")

    lig_dir = cfg.ternary_base / compound_id / "ligand"
    lig_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate 3D conformer from SMILES
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {compound_id}: {smiles}")

    formal_charge = Chem.GetFormalCharge(mol)
    logging.info(f"  SMILES: {smiles}")
    logging.info(f"  Formal charge: {formal_charge}")

    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)

    pdb_file = lig_dir / "ligand_rdkit.pdb"
    Chem.MolToPDBFile(mol_h, str(pdb_file))
    assert_file(pdb_file, "RDKit PDB")

    # Step 2: Run acpype (AM1-BCC first, Gasteiger fallback)
    for method, timeout_s in [("bcc", 600), ("gas", 120)]:
        logging.info(f"  acpype: trying {method} charges (timeout {timeout_s}s)...")

        # Clean previous attempts
        for d in lig_dir.glob("*.acpype"):
            shutil.rmtree(d, ignore_errors=True)

        try:
            r = sh(f"acpype -i {pdb_file} -c {method} -a gaff2 "
                   f"-n {formal_charge} -o gmx",
                   cwd=lig_dir, check=False, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logging.warning(f"  {method} timed out after {timeout_s}s")
            continue

        # Find output
        dirs = list(lig_dir.glob("*.acpype"))
        if not dirs:
            logging.warning(f"  No acpype output for {method}")
            continue
        out = dirs[0]

        # Copy files
        copied = False
        for pat, dest in [("*_GMX.gro", "ligand.gro"),
                          ("*_GMX.itp", "ligand.itp")]:
            hits = list(out.glob(pat))
            if hits:
                shutil.copy2(hits[0], lig_dir / dest)
                copied = True

        if copied and (lig_dir / "ligand.gro").exists():
            logging.info(f"  acpype succeeded with {method} charges")
            break
        else:
            logging.warning(f"  {method} failed — no output files")
    else:
        raise RuntimeError(f"acpype failed for {compound_id} (both bcc and gas)")

    # Step 3: Rename moleculetype to LIG
    _rename_ligand_moltype(lig_dir)

    # Step 4: Split ITP into atomtypes + moleculetype
    _split_ligand_itp(lig_dir)

    assert_file(lig_dir / "ligand.gro", "Ligand GRO")
    assert_file(lig_dir / "ligand.itp", "Ligand ITP")
    logging.info(f"  Ligand prep complete: {lig_dir}")
    return lig_dir


def _rename_ligand_moltype(lig_dir: Path):
    """Rename the acpype moleculetype to 'LIG'."""
    for fname in ("ligand.itp", "ligand.gro"):
        fp = lig_dir / fname
        if not fp.exists():
            continue
        txt = fp.read_text()
        # Find original moleculetype name (typically the PDB stem)
        import re
        m = re.search(r'\[\s*moleculetype\s*\]\s*;\s*\S+\s*\n(\S+)', txt)
        if m:
            old_name = m.group(1)
            txt = txt.replace(old_name, "LIG")
        # Also replace in GRO residue names
        if fname.endswith(".gro"):
            lines = txt.split('\n')
            new_lines = []
            for line in lines:
                if len(line) > 10 and line[5:10].strip():
                    line = line[:5] + "  LIG" + line[10:]
                new_lines.append(line)
            txt = '\n'.join(new_lines)
        fp.write_text(txt)


def _split_ligand_itp(lig_dir: Path):
    """Split ligand ITP into atomtypes (for ffnonbonded) and moleculetype."""
    itp = lig_dir / "ligand.itp"
    txt = itp.read_text()

    # Extract [ atomtypes ] section
    import re
    at_match = re.search(
        r'(\[\s*atomtypes\s*\].*?)(?=\[\s*moleculetype\s*\])',
        txt, re.DOTALL
    )
    if at_match:
        atomtypes = at_match.group(1)
        (lig_dir / "ligand_atomtypes.itp").write_text(atomtypes)
        # Remove atomtypes from main ITP
        txt = txt.replace(atomtypes, "")
        itp.write_text(txt)
        logging.info("  Split atomtypes from ligand ITP")


# =============================================================================
# Step 6: Dock ligand into GLI1-DNA complex
# =============================================================================
def dock_into_complex(compound_id: str, info: Dict, cfg: TernaryConfig,
                      receptor_pdbqt: Path) -> Path:
    """Dock ligand into GLI1-DNA complex using AutoDock Vina."""
    logging.info(f"\n{'='*60}")
    logging.info(f"DOCKING: {compound_id} → {info['site']}")
    logging.info(f"{'='*60}")

    dock_dir = cfg.ternary_base / compound_id / "docking"
    dock_dir.mkdir(parents=True, exist_ok=True)

    lig_dir = cfg.ternary_base / compound_id / "ligand"

    # Convert ligand to PDBQT
    lig_pdb = lig_dir / "ligand_rdkit.pdb"
    lig_pdbqt = dock_dir / "ligand.pdbqt"
    sh(f"obabel -ipdb {lig_pdb} -opdbqt -O {lig_pdbqt} --partialcharge gasteiger",
       cwd=dock_dir)

    # Select docking box based on binding site
    if info['site'] == 'ZF2-3':
        cx, cy, cz = cfg.zf23_center
        sx, sy, sz = cfg.zf23_size
    else:  # ZF4-5
        cx, cy, cz = cfg.zf45_center
        sx, sy, sz = cfg.zf45_size

    # Write Vina config
    vina_cfg = dock_dir / "vina.conf"
    vina_cfg.write_text(f"""receptor = {receptor_pdbqt}
ligand = {lig_pdbqt}
center_x = {cx}
center_y = {cy}
center_z = {cz}
size_x = {sx}
size_y = {sy}
size_z = {sz}
exhaustiveness = {cfg.vina_exhaustiveness}
num_modes = {cfg.vina_n_modes}
energy_range = 4
""")

    # Run Vina
    out_pdbqt = dock_dir / "docked.pdbqt"
    log_file = dock_dir / "vina.log"
    sh(f"vina --config {vina_cfg} --out {out_pdbqt} --log {log_file}",
       cwd=dock_dir, timeout=3600)

    assert_file(out_pdbqt, "Docked PDBQT")

    # Parse best score
    with open(log_file) as f:
        for line in f:
            if line.strip().startswith("1"):
                parts = line.split()
                if len(parts) >= 2:
                    score = float(parts[1])
                    logging.info(f"  Best docking score: {score} kcal/mol")
                    break

    # Extract best pose as PDB
    best_pdb = dock_dir / "best_pose.pdb"
    sh(f"obabel -ipdbqt {out_pdbqt} -opdb -O {best_pdb} -l 1",
       cwd=dock_dir)

    assert_file(best_pdb, "Best docked pose PDB")
    return best_pdb


# =============================================================================
# Step 7: Prepare receptor PDBQT for docking
# =============================================================================
def prepare_receptor_pdbqt(cfg: TernaryConfig, clean_pdb: Path) -> Path:
    """Convert cleaned PDB to PDBQT for Vina docking."""
    logging.info("="*60)
    logging.info("Preparing receptor PDBQT for docking")
    logging.info("="*60)

    dock_dir = cfg.ternary_base / "receptor_prep"
    dock_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = dock_dir / "receptor.pdbqt"

    # Use obabel for conversion (handles protein + DNA)
    sh(f"obabel -ipdb {clean_pdb} -opdbqt -O {receptor_pdbqt} -xr",
       cwd=dock_dir)

    assert_file(receptor_pdbqt, "Receptor PDBQT")
    logging.info(f"  Receptor PDBQT: {receptor_pdbqt}")
    return receptor_pdbqt


# =============================================================================
# Step 8: Kabsch alignment of parameterized ligand to docked pose
# =============================================================================
def kabsch_align_gro_to_pdb(gro_path: Path, ref_pdb: Path, out_gro: Path):
    """Align GRO coordinates to reference PDB using Kabsch SVD superposition.

    Reads heavy atoms from both, computes optimal rotation+translation,
    applies to ALL atoms in GRO (including hydrogens).
    """
    logging.info("  Kabsch alignment of parameterized ligand to docked pose...")

    from rdkit import Chem

    # Parse reference PDB heavy atom coordinates
    ref_mol = Chem.MolFromPDBFile(str(ref_pdb), removeHs=True, sanitize=False)
    if ref_mol is None:
        logging.warning("  Cannot parse reference PDB for Kabsch — skipping alignment")
        shutil.copy2(gro_path, out_gro)
        return

    ref_conf = ref_mol.GetConformer()
    ref_coords = np.array([ref_conf.GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])
    ref_coords /= 10.0  # Å → nm

    # Parse GRO heavy atom coordinates
    with open(gro_path) as f:
        lines = f.readlines()

    natoms = int(lines[1].strip())
    gro_atoms = []
    gro_heavy_idx = []
    for i in range(2, 2 + natoms):
        line = lines[i]
        atomname = line[10:15].strip()
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        gro_atoms.append((atomname, np.array([x, y, z])))
        if not atomname.startswith('H'):
            gro_heavy_idx.append(i - 2)

    gro_heavy = np.array([gro_atoms[j][1] for j in gro_heavy_idx])

    # Match atom counts
    n_ref = len(ref_coords)
    n_gro = len(gro_heavy)
    n_match = min(n_ref, n_gro)

    if n_match < 3:
        logging.warning(f"  Too few atoms for Kabsch ({n_match}) — skipping")
        shutil.copy2(gro_path, out_gro)
        return

    P = gro_heavy[:n_match]
    Q = ref_coords[:n_match]

    # Kabsch algorithm
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = centroid_Q - R @ centroid_P

    # Apply to all atoms
    new_lines = lines[:2]
    for i in range(natoms):
        old_line = lines[i + 2]
        old_pos = gro_atoms[i][1]
        new_pos = R @ old_pos + t
        new_line = old_line[:20] + f"{new_pos[0]:8.3f}{new_pos[1]:8.3f}{new_pos[2]:8.3f}" + old_line[44:]
        new_lines.append(new_line)
    new_lines.extend(lines[2 + natoms:])

    with open(out_gro, 'w') as f:
        f.writelines(new_lines)

    rmsd = np.sqrt(np.mean(np.sum((R @ P.T + t[:, None] - Q.T)**2, axis=0)))
    logging.info(f"  Kabsch RMSD: {rmsd*10:.2f} Å ({n_match} atoms)")


# =============================================================================
# Step 9: Build combined system (protein-DNA-ligand)
# =============================================================================
def build_ternary_system(compound_id: str, info: Dict, cfg: TernaryConfig,
                         complex_gro: Path, complex_top: Path,
                         docked_pdb: Path) -> Tuple[Path, Path]:
    """Combine protein-DNA complex with docked ligand and solvate."""
    logging.info(f"\n{'='*60}")
    logging.info(f"BUILDING SYSTEM: {compound_id}")
    logging.info(f"{'='*60}")

    sys_dir = cfg.ternary_base / compound_id / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)

    lig_dir = cfg.ternary_base / compound_id / "ligand"

    # Align parameterized ligand GRO to docked pose
    aligned_lig = sys_dir / "ligand_aligned.gro"
    kabsch_align_gro_to_pdb(lig_dir / "ligand.gro", docked_pdb, aligned_lig)

    # Merge protein-DNA GRO + ligand GRO
    merged_gro = sys_dir / "complex_lig.gro"
    _merge_gro_files(complex_gro, aligned_lig, merged_gro)

    # Copy and modify topology
    merged_top = sys_dir / "topol.top"
    shutil.copy2(complex_top, merged_top)

    # Add ligand atomtypes and include
    _add_ligand_to_topology(merged_top, lig_dir)

    # Solvate
    boxed_gro = sys_dir / "boxed.gro"
    sh(f"gmx editconf -f {merged_gro} -o {boxed_gro} -c -d {cfg.box_buffer} -bt dodecahedron",
       cwd=sys_dir)

    solvated_gro = sys_dir / "solvated.gro"
    sh(f"gmx solvate -cp {boxed_gro} -cs spc216.gro -o {solvated_gro} -p {merged_top}",
       cwd=sys_dir)

    # Add ions
    ions_gro = sys_dir / "ions.gro"
    _write_ions_mdp(sys_dir)
    sh(f"gmx grompp -f ions.mdp -c {solvated_gro} -p {merged_top} -o ions.tpr -maxwarn 10",
       cwd=sys_dir)
    sh(f"gmx genion -s ions.tpr -o {ions_gro} -p {merged_top} "
       f"-pname NA -nname CL -neutral -conc {cfg.ion_conc}",
       cwd=sys_dir, stdin_text="SOL\n")

    assert_file(ions_gro, "Ionized system")
    logging.info(f"  System built: {ions_gro}")
    return ions_gro, merged_top


def build_control_system(cfg: TernaryConfig, complex_gro: Path,
                         complex_top: Path) -> Tuple[Path, Path]:
    """Build control system (GLI1-DNA without inhibitor)."""
    logging.info(f"\n{'='*60}")
    logging.info("BUILDING CONTROL SYSTEM: GLI1-DNA (no inhibitor)")
    logging.info(f"{'='*60}")

    sys_dir = cfg.ternary_base / "control" / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)

    # Copy topology
    ctrl_top = sys_dir / "topol.top"
    shutil.copy2(complex_top, ctrl_top)
    # Copy any included ITP files
    for itp in complex_top.parent.glob("*.itp"):
        shutil.copy2(itp, sys_dir / itp.name)

    boxed_gro = sys_dir / "boxed.gro"
    sh(f"gmx editconf -f {complex_gro} -o {boxed_gro} -c -d {cfg.box_buffer} -bt dodecahedron",
       cwd=sys_dir)

    solvated_gro = sys_dir / "solvated.gro"
    sh(f"gmx solvate -cp {boxed_gro} -cs spc216.gro -o {solvated_gro} -p {ctrl_top}",
       cwd=sys_dir)

    ions_gro = sys_dir / "ions.gro"
    _write_ions_mdp(sys_dir)
    sh(f"gmx grompp -f ions.mdp -c {solvated_gro} -p {ctrl_top} -o ions.tpr -maxwarn 10",
       cwd=sys_dir)
    sh(f"gmx genion -s ions.tpr -o {ions_gro} -p {ctrl_top} "
       f"-pname NA -nname CL -neutral -conc {cfg.ion_conc}",
       cwd=sys_dir, stdin_text="SOL\n")

    assert_file(ions_gro, "Control ionized system")
    return ions_gro, ctrl_top


def _merge_gro_files(complex_gro: Path, lig_gro: Path, out_gro: Path):
    """Merge two GRO files (protein-DNA + ligand)."""
    with open(complex_gro) as f:
        c_lines = f.readlines()
    with open(lig_gro) as f:
        l_lines = f.readlines()

    c_natoms = int(c_lines[1].strip())
    l_natoms = int(l_lines[1].strip())
    total = c_natoms + l_natoms

    out_lines = [c_lines[0]]
    out_lines.append(f"{total}\n")
    # Complex atoms
    out_lines.extend(c_lines[2:2 + c_natoms])
    # Ligand atoms (renumber)
    for i in range(2, 2 + l_natoms):
        line = l_lines[i]
        # Update residue number
        max_resnum = int(c_lines[1 + c_natoms].split()[0][:5]) if c_natoms > 0 else 0
        try:
            old_resnum = int(line[0:5].strip())
            new_resnum = max_resnum + old_resnum
            line = f"{new_resnum:5d}" + line[5:]
        except ValueError:
            pass
        out_lines.append(line)
    # Box vector
    out_lines.append(c_lines[-1])

    with open(out_gro, 'w') as f:
        f.writelines(out_lines)


def _add_ligand_to_topology(top_file: Path, lig_dir: Path):
    """Add ligand include directives and molecule entry to topology."""
    txt = top_file.read_text()

    # Add atomtypes include after force field include
    atomtypes_itp = lig_dir / "ligand_atomtypes.itp"
    lig_itp = lig_dir / "ligand.itp"

    if atomtypes_itp.exists():
        # Insert atomtypes after the first #include of forcefield
        import re
        ff_include = re.search(r'(#include\s+".*forcefield\.itp".*\n)', txt)
        if ff_include:
            insert_pos = ff_include.end()
            txt = (txt[:insert_pos] +
                   f'\n; Ligand atom types\n#include "{atomtypes_itp}"\n' +
                   txt[insert_pos:])

    # Add ligand ITP include before [ system ]
    system_match = txt.find("[ system ]")
    if system_match > 0:
        txt = (txt[:system_match] +
               f'; Ligand topology\n#include "{lig_itp}"\n\n' +
               txt[system_match:])

    # Add LIG to [ molecules ]
    txt += "\nLIG                 1\n"

    top_file.write_text(txt)


def _write_ions_mdp(work_dir: Path):
    """Write minimal MDP for genion."""
    (work_dir / "ions.mdp").write_text(
        "integrator = steep\nnsteps = 0\n"
    )


# =============================================================================
# Step 10: Write MDP files
# =============================================================================
def write_mdp_files(cfg: TernaryConfig, work_dir: Path):
    """Write MDP parameter files for EM, NVT, NPT, and production MD."""
    work_dir.mkdir(parents=True, exist_ok=True)

    # Energy minimization
    (work_dir / "em.mdp").write_text(f"""\
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = {cfg.em_steps}
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
""")

    # NVT equilibration (with position restraints)
    (work_dir / "nvt.mdp").write_text(f"""\
integrator  = md
nsteps      = {cfg.nvt_ps * 500}
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
; Temperature
tcoupl      = V-rescale
tc-grps     = Protein_DNA LIG Water_and_ions
tau_t       = 0.1  0.1  0.1
ref_t       = 300  300  300
; Pressure (none for NVT)
pcoupl      = no
; Constraints
constraints = h-bonds
constraint_algorithm = lincs
continuation = no
; Position restraints
define      = -DPOSRES
; Velocity generation
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
""")

    # NVT for control (no ligand group)
    (work_dir / "nvt_ctrl.mdp").write_text(f"""\
integrator  = md
nsteps      = {cfg.nvt_ps * 500}
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein_DNA Water_and_ions
tau_t       = 0.1  0.1
ref_t       = 300  300
pcoupl      = no
constraints = h-bonds
constraint_algorithm = lincs
continuation = no
define      = -DPOSRES
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
""")

    # NPT equilibration
    (work_dir / "npt.mdp").write_text(f"""\
integrator  = md
nsteps      = {cfg.npt_ps * 500}
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein_DNA LIG Water_and_ions
tau_t       = 0.1  0.1  0.1
ref_t       = 300  300  300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
refcoord_scaling = com
constraints = h-bonds
constraint_algorithm = lincs
continuation = yes
define      = -DPOSRES
gen_vel     = no
""")

    (work_dir / "npt_ctrl.mdp").write_text(f"""\
integrator  = md
nsteps      = {cfg.npt_ps * 500}
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein_DNA Water_and_ions
tau_t       = 0.1  0.1
ref_t       = 300  300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
refcoord_scaling = com
constraints = h-bonds
constraint_algorithm = lincs
continuation = yes
define      = -DPOSRES
gen_vel     = no
""")

    # Production MD
    prod_nsteps = cfg.prod_ns * 500000  # ns → steps at 2fs
    (work_dir / "prod.mdp").write_text(f"""\
integrator  = md
nsteps      = {prod_nsteps}
dt          = 0.002
nstxout-compressed = 5000    ; 10 ps
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein_DNA LIG Water_and_ions
tau_t       = 0.1  0.1  0.1
ref_t       = 300  300  300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
constraints = h-bonds
constraint_algorithm = lincs
continuation = yes
gen_vel     = no
""")

    (work_dir / "prod_ctrl.mdp").write_text(f"""\
integrator  = md
nsteps      = {prod_nsteps}
dt          = 0.002
nstxout-compressed = 5000
nstlog      = 5000
nstenergy   = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = Protein_DNA Water_and_ions
tau_t       = 0.1  0.1
ref_t       = 300  300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
constraints = h-bonds
constraint_algorithm = lincs
continuation = yes
gen_vel     = no
""")

    logging.info(f"  MDP files written to {work_dir}")


# =============================================================================
# Step 11: Run MD simulation
# =============================================================================
def run_md(system_name: str, ions_gro: Path, top_file: Path,
           cfg: TernaryConfig, is_control: bool = False):
    """Run full MD pipeline: EM → NVT → NPT → Production."""
    logging.info(f"\n{'='*60}")
    logging.info(f"MD SIMULATION: {system_name}")
    logging.info(f"{'='*60}")

    md_dir = ions_gro.parent.parent / "md"
    md_dir.mkdir(parents=True, exist_ok=True)
    mdp_dir = cfg.ternary_base / "mdp_files"

    suffix = "_ctrl" if is_control else ""

    # Energy minimization
    logging.info("  EM: Energy minimization...")
    sh(f"gmx grompp -f {mdp_dir}/em.mdp -c {ions_gro} -p {top_file} "
       f"-o {md_dir}/em.tpr -maxwarn 10", cwd=md_dir)
    sh(f"gmx mdrun -v -deffnm {md_dir}/em -ntmpi 1 -ntomp 8 -nb gpu -gpu_id 0",
       cwd=md_dir)

    # NVT equilibration
    logging.info("  NVT: Temperature equilibration...")
    sh(f"gmx grompp -f {mdp_dir}/nvt{suffix}.mdp -c {md_dir}/em.gro "
       f"-r {md_dir}/em.gro -p {top_file} -o {md_dir}/nvt.tpr -maxwarn 10",
       cwd=md_dir)
    sh(f"gmx mdrun -v -deffnm {md_dir}/nvt -ntmpi 1 -ntomp 8 -nb gpu -gpu_id 0",
       cwd=md_dir)

    # NPT equilibration
    logging.info("  NPT: Pressure equilibration...")
    sh(f"gmx grompp -f {mdp_dir}/npt{suffix}.mdp -c {md_dir}/nvt.gro "
       f"-r {md_dir}/nvt.gro -p {top_file} -o {md_dir}/npt.tpr -maxwarn 10",
       cwd=md_dir)
    sh(f"gmx mdrun -v -deffnm {md_dir}/npt -ntmpi 1 -ntomp 8 -nb gpu -gpu_id 0",
       cwd=md_dir)

    # Production MD
    logging.info(f"  PRODUCTION: {cfg.prod_ns} ns...")
    sh(f"gmx grompp -f {mdp_dir}/prod{suffix}.mdp -c {md_dir}/npt.gro "
       f"-p {top_file} -o {md_dir}/prod.tpr -maxwarn 10",
       cwd=md_dir)
    sh(f"gmx mdrun -v -deffnm {md_dir}/prod -ntmpi 1 -ntomp 8 "
       f"-nb gpu -pme gpu -gpu_id 0",
       cwd=md_dir)

    logging.info(f"  MD complete: {md_dir}/prod.xtc")
    return md_dir


# =============================================================================
# Step 12: Analysis
# =============================================================================
def run_analysis(system_name: str, md_dir: Path, cfg: TernaryConfig,
                 is_control: bool = False):
    """Run comprehensive analysis of MD trajectory."""
    logging.info(f"\n{'='*60}")
    logging.info(f"ANALYSIS: {system_name}")
    logging.info(f"{'='*60}")

    analysis_dir = md_dir.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tpr = md_dir / "prod.tpr"
    xtc = md_dir / "prod.xtc"

    # 1. Backbone RMSD
    logging.info("  Backbone RMSD...")
    sh(f"gmx rms -s {tpr} -f {xtc} -o {analysis_dir}/rmsd_backbone.xvg -tu ns",
       cwd=analysis_dir, stdin_text="4\n4\n")  # Backbone

    # 2. Protein-DNA RMSD (all non-water heavy atoms)
    logging.info("  System RMSD...")
    sh(f"gmx rms -s {tpr} -f {xtc} -o {analysis_dir}/rmsd_system.xvg -tu ns",
       cwd=analysis_dir, stdin_text="1\n1\n")  # Protein+DNA

    # 3. Ligand RMSD (if not control)
    if not is_control:
        logging.info("  Ligand RMSD...")
        sh(f"gmx rms -s {tpr} -f {xtc} -o {analysis_dir}/rmsd_ligand.xvg -tu ns",
           cwd=analysis_dir, stdin_text="4\n13\n", check=False)  # Backbone fit, ligand RMSD

    # 4. Per-residue RMSF
    logging.info("  Per-residue RMSF...")
    sh(f"gmx rmsf -s {tpr} -f {xtc} -o {analysis_dir}/rmsf.xvg -res",
       cwd=analysis_dir, stdin_text="4\n")  # Backbone

    # 5. Radius of gyration
    logging.info("  Radius of gyration...")
    sh(f"gmx gyrate -s {tpr} -f {xtc} -o {analysis_dir}/gyrate.xvg",
       cwd=analysis_dir, stdin_text="1\n")  # Protein

    # 6. Hydrogen bonds between protein and DNA
    logging.info("  Protein-DNA hydrogen bonds...")
    # Create index groups for protein and DNA
    sh(f"gmx hbond -s {tpr} -f {xtc} -num {analysis_dir}/hbond_prot_dna.xvg -tu ns",
       cwd=analysis_dir, stdin_text="1\n12\n", check=False)  # Protein / DNA

    # 7. Minimum distance protein-DNA
    logging.info("  Protein-DNA minimum distance...")
    sh(f"gmx mindist -s {tpr} -f {xtc} -od {analysis_dir}/mindist_prot_dna.xvg -tu ns",
       cwd=analysis_dir, stdin_text="1\n12\n", check=False)

    logging.info(f"  Analysis complete: {analysis_dir}")
    return analysis_dir


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    cfg = TernaryConfig()
    cfg.ternary_base.mkdir(parents=True, exist_ok=True)

    os.chdir(cfg.ternary_base)

    logging.info("="*70)
    logging.info("GLI1-DNA TERNARY COMPLEX MD PIPELINE")
    logging.info("="*70)
    logging.info(f"Base directory: {cfg.ternary_base}")
    logging.info(f"Production MD: {cfg.prod_ns} ns per system")
    logging.info(f"Systems: 1 control + {len(COMPOUNDS)} ternary complexes")
    logging.info("")

    # Step 1: Prepare PDB
    clean_pdb = prepare_pdb(cfg)

    # Step 2: Build protein-DNA topology
    complex_gro, complex_top = build_protein_dna_topology(cfg, clean_pdb)

    # Step 3: ZAFF zinc parameters
    zaff_file = add_zaff_zinc(cfg, complex_top)

    # Step 4: Identify zinc coordination
    coordinations = identify_zinc_coordination(cfg, complex_gro)

    # Step 5: Write MDP files
    mdp_dir = cfg.ternary_base / "mdp_files"
    write_mdp_files(cfg, mdp_dir)

    # Step 6: Prepare receptor for docking
    receptor_pdbqt = prepare_receptor_pdbqt(cfg, clean_pdb)

    # Step 7: Prepare ligands
    for cid, info in COMPOUNDS.items():
        prepare_ligand(cid, info['smiles'], cfg)

    # Step 8: Dock ligands into GLI1-DNA complex
    docked_poses = {}
    for cid, info in COMPOUNDS.items():
        docked_poses[cid] = dock_into_complex(cid, info, cfg, receptor_pdbqt)

    # Step 9: Build systems
    # Control
    ctrl_gro, ctrl_top = build_control_system(cfg, complex_gro, complex_top)

    # Ternary complexes
    ternary_systems = {}
    for cid, info in COMPOUNDS.items():
        gro, top = build_ternary_system(cid, info, cfg, complex_gro, complex_top,
                                        docked_poses[cid])
        ternary_systems[cid] = (gro, top)

    logging.info("\n" + "="*70)
    logging.info("ALL SYSTEMS BUILT — READY FOR MD")
    logging.info("="*70)
    logging.info("Use launch_ternary_md.sh to start production runs on GPUs")

    # Save system info for the launch script
    sys_info = {
        "control": {"gro": str(ctrl_gro), "top": str(ctrl_top)},
    }
    for cid, (gro, top) in ternary_systems.items():
        sys_info[cid] = {"gro": str(gro), "top": str(top)}

    with open(cfg.ternary_base / "system_info.json", 'w') as f:
        json.dump(sys_info, f, indent=2)

    logging.info(f"System info saved: {cfg.ternary_base / 'system_info.json'}")


if __name__ == "__main__":
    main()
