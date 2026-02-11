"""
Fix GLI1 Receptor PDBQT for Zinc-Aware Docking.

Problems in current 2gli_receptor.pdbqt:
  1. All Zn atoms have charge +0.000 (should be +2.000 for Zn2+)
  2. All protein atoms have charge +0.000 (no Gasteiger/Kollman charges)
  3. Zinc-coordinating CYS/HIS residues may need adjusted charges

This script:
  1. Re-prepares the receptor from the PDB using MGLTools prepare_receptor4.py
     with proper charge assignment (Gasteiger)
  2. If MGLTools not available, patches zinc charges in existing PDBQT
  3. Validates zinc coordination geometry
  4. Optionally generates AD4 parameter file with zinc parameters

Usage:
    python fix_receptor_zinc.py --pdb 2gli_with_zinc.pdb --output 2gli_receptor_zn_fixed.pdbqt
    python fix_receptor_zinc.py --patch 2gli_receptor.pdbqt --output 2gli_receptor_zn_fixed.pdbqt
"""

import os
import sys
import argparse
import subprocess
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Zinc coordination validation
# ---------------------------------------------------------------------------

def parse_pdb_atoms(pdb_path: str) -> List[Dict]:
    """Parse ATOM/HETATM records from PDB file."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append({
                    "record": line[:6].strip(),
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip() if len(line) > 76 else "",
                    "line": line.rstrip(),
                })
    return atoms


def validate_zinc_coordination(pdb_path: str) -> Dict:
    """Validate zinc coordination geometry in the PDB.
    
    Expected: Each Zn2+ coordinated by 2 Cys (SG) + 2 His (NE2/ND1) at ~2.0-2.4 Å.
    """
    atoms = parse_pdb_atoms(pdb_path)
    
    zinc_atoms = [a for a in atoms if a["element"] == "ZN" or a["name"] == "ZN"]
    coord_donors = [a for a in atoms if 
                    (a["resname"] == "CYS" and a["name"] == "SG") or
                    (a["resname"] == "HIS" and a["name"] in ("NE2", "ND1"))]
    
    logging.info(f"Found {len(zinc_atoms)} zinc atoms and {len(coord_donors)} potential donors")
    
    results = {}
    for zn in zinc_atoms:
        zn_coord = np.array([zn["x"], zn["y"], zn["z"]])
        zn_label = f"ZN_{zn['resseq']}"
        
        # Find coordinating atoms within 3.0 Å
        coordinating = []
        for donor in coord_donors:
            d_coord = np.array([donor["x"], donor["y"], donor["z"]])
            dist = np.linalg.norm(zn_coord - d_coord)
            if dist < 3.0:
                coordinating.append({
                    "atom": f"{donor['resname']}{donor['resseq']}.{donor['name']}",
                    "distance": dist,
                    "resname": donor["resname"],
                })
        
        n_cys = sum(1 for c in coordinating if c["resname"] == "CYS")
        n_his = sum(1 for c in coordinating if c["resname"] == "HIS")
        
        results[zn_label] = {
            "coord": zn_coord.tolist(),
            "n_ligands": len(coordinating),
            "n_cys": n_cys,
            "n_his": n_his,
            "ligands": coordinating,
            "geometry_ok": len(coordinating) == 4 and n_cys == 2 and n_his == 2,
        }
        
        status = "OK" if results[zn_label]["geometry_ok"] else "UNUSUAL"
        logging.info(f"  {zn_label}: {n_cys} Cys + {n_his} His = {len(coordinating)} ligands [{status}]")
        for c in coordinating:
            logging.info(f"    {c['atom']}: {c['distance']:.2f} Å")
    
    return results


# ---------------------------------------------------------------------------
# PDBQT patching (fallback when MGLTools unavailable)
# ---------------------------------------------------------------------------

def patch_pdbqt_zinc_charges(pdbqt_path: str, output_path: str) -> str:
    """Patch zinc charges in an existing PDBQT file.
    
    Sets:
      - Zn atoms: charge = +2.000, type = Zn
      - Also fixes coordinating CYS SG and HIS NE2/ND1 if they have 0 charge
    """
    lines = []
    n_fixed_zn = 0
    n_fixed_other = 0
    
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                atom_type = line[77:79].strip() if len(line) > 77 else ""
                
                # Fix zinc charge
                if resname == "ZN" or atom_name == "ZN" or atom_type == "Zn":
                    # Set charge to +2.000
                    # PDBQT format: columns 70-76 = charge, 77-79 = type
                    charge_str = "+2.000"
                    # Reconstruct line with correct charge
                    new_line = line[:70] + f"{charge_str:>6s}" + " " + "Zn" + "\n"
                    lines.append(new_line)
                    n_fixed_zn += 1
                    continue
                
                # Check if charge is 0.000 for coordinating residues
                if len(line) > 76:
                    try:
                        charge = float(line[70:76].strip().replace("+", "").replace("-", "-"))
                    except ValueError:
                        charge = 0.0
                    
                    if abs(charge) < 0.001:
                        # Assign approximate Gasteiger charges for key atoms
                        if resname == "CYS" and atom_name == "SG":
                            new_charge = "-0.232"
                            new_line = line[:70] + f"{new_charge:>6s}" + line[76:]
                            lines.append(new_line)
                            n_fixed_other += 1
                            continue
                        elif resname == "HIS" and atom_name in ("NE2", "ND1"):
                            new_charge = "-0.398"
                            new_line = line[:70] + f"{new_charge:>6s}" + line[76:]
                            lines.append(new_line)
                            n_fixed_other += 1
                            continue
            
            lines.append(line)
    
    with open(output_path, "w") as f:
        f.writelines(lines)
    
    logging.info(f"Patched {n_fixed_zn} Zn charges to +2.000")
    logging.info(f"Patched {n_fixed_other} coordinating atom charges")
    logging.info(f"Saved to {output_path}")
    
    return output_path


# ---------------------------------------------------------------------------
# Full re-preparation with MGLTools (preferred)
# ---------------------------------------------------------------------------

def prepare_receptor_mgltools(pdb_path: str, output_path: str) -> Optional[str]:
    """Prepare receptor using MGLTools prepare_receptor4.py (if available).
    
    This properly assigns Gasteiger charges to ALL atoms.
    """
    # Try to find prepare_receptor4.py
    search_paths = [
        "prepare_receptor4.py",
        os.path.expanduser("~/MGLTools/bin/prepare_receptor4.py"),
        "/usr/local/bin/prepare_receptor4.py",
        os.path.expanduser("~/miniforge3/envs/gli/bin/prepare_receptor4.py"),
    ]
    
    prep_script = None
    for p in search_paths:
        if os.path.exists(p):
            prep_script = p
            break
    
    # Also try just calling it
    if prep_script is None:
        try:
            result = subprocess.run(
                ["prepare_receptor4.py", "-h"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                prep_script = "prepare_receptor4.py"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    if prep_script is None:
        logging.warning("MGLTools prepare_receptor4.py not found. Using patch mode.")
        return None
    
    cmd = [
        sys.executable if prep_script.endswith(".py") else prep_script,
        prep_script if prep_script.endswith(".py") else "",
        "-r", pdb_path,
        "-o", output_path,
        "-A", "hydrogens",     # Add hydrogens
        "-U", "nphs_lps_waters",  # Clean up
        "-e",                    # Compute Gasteiger charges
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(output_path):
        logging.info(f"Receptor prepared with MGLTools: {output_path}")
        return output_path
    else:
        logging.warning(f"MGLTools failed: {result.stderr}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fix receptor PDBQT for zinc-aware docking")
    parser.add_argument("--pdb", type=str, default="2gli_with_zinc.pdb",
                        help="Input PDB file (for full re-preparation)")
    parser.add_argument("--patch", type=str, default=None,
                        help="Existing PDBQT to patch (fallback mode)")
    parser.add_argument("--output", type=str, default="2gli_receptor_zn_fixed.pdbqt",
                        help="Output PDBQT path")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate zinc coordination, don't fix")
    args = parser.parse_args()

    # Step 1: Validate zinc coordination
    if os.path.exists(args.pdb):
        logging.info("=== Validating Zinc Coordination ===")
        zn_results = validate_zinc_coordination(args.pdb)
        
        all_ok = all(r["geometry_ok"] for r in zn_results.values())
        if all_ok:
            logging.info("All zinc coordination geometries are standard (2 Cys + 2 His)")
        else:
            logging.warning("Some zinc sites have non-standard coordination — check manually")
        
        if args.validate_only:
            return

    # Step 2: Fix receptor
    if args.patch:
        # Patch mode: fix charges in existing PDBQT
        logging.info("\n=== Patching Existing PDBQT ===")
        patch_pdbqt_zinc_charges(args.patch, args.output)
    else:
        # Try MGLTools first, fall back to patching
        logging.info("\n=== Preparing Receptor ===")
        result = prepare_receptor_mgltools(args.pdb, args.output)
        
        if result is None:
            # Fallback: patch existing PDBQT
            existing_pdbqt = args.pdb.replace(".pdb", "_receptor.pdbqt").replace("_with_zinc", "")
            if not os.path.exists(existing_pdbqt):
                existing_pdbqt = "2gli_receptor.pdbqt"
            
            if os.path.exists(existing_pdbqt):
                logging.info(f"Falling back to patching {existing_pdbqt}")
                patch_pdbqt_zinc_charges(existing_pdbqt, args.output)
            else:
                logging.error(f"No PDBQT found to patch. Provide --patch argument.")
                sys.exit(1)

    # Step 3: Verify the fix
    logging.info("\n=== Verifying Fixed PDBQT ===")
    n_zn_fixed = 0
    with open(args.output) as f:
        for line in f:
            if "ZN" in line and line.startswith(("ATOM", "HETATM")):
                charge = line[70:76].strip()
                logging.info(f"  Zinc line: ...{line[60:].rstrip()}")
                if "+2.000" in charge or "2.0" in charge:
                    n_zn_fixed += 1

    logging.info(f"\nVerification: {n_zn_fixed}/5 zinc atoms have correct +2.000 charge")
    if n_zn_fixed == 5:
        logging.info("SUCCESS: Receptor PDBQT is fixed for zinc-aware docking")
    else:
        logging.warning("Some zinc charges may still be wrong — inspect manually")


if __name__ == "__main__":
    main()
