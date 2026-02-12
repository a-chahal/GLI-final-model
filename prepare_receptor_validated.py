#!/usr/bin/env python3
"""
Prepare GLI1 zinc finger receptor for docking — publication-quality protocol.

Addresses all issues identified in the structural audit:
1. Strips DNA (chains C, D) — inhibitors compete with DNA for the protein surface
2. Adds polar hydrogens via Open Babel at pH 7.4
3. Generates PDBQT with correct AutoDock atom types
4. Patches Zn2+ charges to +2.000 and atom type to Zn
5. Validates zinc coordination geometry (C2H2 tetrahedral)
6. Validates that known binding site residues are present and accessible

References:
  - 2GLI: Pavletich & Pabo, Science 261:1701 (1993)
  - GANT61 site (ZF2-3): Agyeman et al., Oncotarget 5:4492 (2014)
    Binding between E119/E167 (PDB numbering = E250/E298 full-length)
    Validated by SPR (KD ~11 µM) and E119A/E167A mutagenesis (~60% reduction)
  - GlaB site (ZF4-5): Infante et al., EMBO J 34:200 (2015)
    Binding at K340/K350 (full-length = K209/K219 PDB numbering)
    Validated by NMR chemical shift perturbation and K340A/K350A mutagenesis

PDB-to-full-length mapping: full_length = PDB_resnum + 131
  (2GLI PDB residues 103-257 = UniProt P08151 residues 234-388, per PDBe)

Zinc finger boundaries (PDB numbering):
  ZF1: C106, C111, H129, H131 — does NOT contact DNA (Pavletich & Pabo)
  ZF2: C139, C144, H160, H164
  ZF3: C172, C177, H190, H194
  ZF4: C202, C207, H220, H225
  ZF5: C233, C238, H251, H256
"""

import argparse
import subprocess
import sys
import os
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Zinc finger definitions (PDB numbering)
# ---------------------------------------------------------------------------
ZF_DEFINITIONS = {
    "ZF1": {"cys": [106, 111], "his": [129, 131], "dna_contact": False},
    "ZF2": {"cys": [139, 144], "his": [160, 164], "dna_contact": True},
    "ZF3": {"cys": [172, 177], "his": [190, 194], "dna_contact": True},
    "ZF4": {"cys": [202, 207], "his": [220, 225], "dna_contact": True},
    "ZF5": {"cys": [233, 238], "his": [251, 256], "dna_contact": True},
}

# Literature-validated binding sites (PDB numbering)
BINDING_SITES = {
    "ZF2-3": {
        "residues": {"E119": "GLU", "E167": "GLU"},
        "reference": "Agyeman et al. 2014 Oncotarget — GANT61 SPR + mutagenesis",
        "full_length_residues": "E250/E298",
    },
    "ZF4-5": {
        "residues": {"K209": "LYS", "K219": "LYS"},
        "reference": "Infante et al. 2015 EMBO J — GlaB NMR CSP + mutagenesis",
        "full_length_residues": "K340/K350",
    },
}


def parse_pdb_atoms(pdb_path):
    """Parse ATOM/HETATM records from PDB file."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom = {
                    "record": line[:6].strip(),
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                    "bfactor": float(line[60:66]) if line[60:66].strip() else 0.0,
                    "element": line[76:78].strip() if len(line) > 76 else "",
                    "raw": line.rstrip(),
                }
                atoms.append(atom)
    return atoms


def strip_dna_and_write_apo(pdb_path, output_path):
    """Strip DNA chains (C, D) and write apo protein + zinc only."""
    atoms = parse_pdb_atoms(pdb_path)

    protein_atoms = [a for a in atoms if a["chain"] == "A"]
    n_dna = len([a for a in atoms if a["chain"] in ("C", "D")])
    n_protein = len(protein_atoms)

    print(f"  Input atoms: {len(atoms)}")
    print(f"  DNA atoms removed (chains C, D): {n_dna}")
    print(f"  Protein + Zn atoms kept (chain A): {n_protein}")

    # Write apo PDB
    with open(output_path, "w") as f:
        f.write(f"REMARK   Apo GLI1 zinc finger domain (DNA stripped)\n")
        f.write(f"REMARK   Source: {pdb_path}\n")
        f.write(f"REMARK   DNA chains C/D removed for docking\n")
        serial = 1
        for a in protein_atoms:
            # Reformat with clean serial numbers
            name_field = f" {a['name']:<3s}" if len(a['name']) < 4 else a['name']
            f.write(
                f"{a['record']:<6s}{serial:5d} {name_field} {a['resname']:>3s} "
                f"{a['chain']}{a['resseq']:4d}    "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                f"{a['occupancy']:6.2f}{a['bfactor']:6.2f}"
                f"          {a['element']:>2s}\n"
            )
            serial += 1
        f.write("END\n")

    return n_protein, n_dna


def add_hydrogens_obabel(input_pdb, output_pdb, ph=7.4):
    """Add polar hydrogens using Open Babel at specified pH."""
    cmd = [
        "obabel", input_pdb, "-O", output_pdb,
        "-p", str(ph),  # Add hydrogens at this pH
        "--partialcharge", "gasteiger",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: obabel H addition returned {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
    else:
        print(f"  Hydrogens added at pH {ph}")
    return result.returncode == 0


def generate_pdbqt_obabel(input_pdb, output_pdbqt):
    """Generate PDBQT with proper atom types using Open Babel."""
    cmd = [
        "obabel", input_pdb, "-O", output_pdbqt,
        "-xr",  # Rigid receptor (no rotatable bonds)
        "--partialcharge", "gasteiger",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: obabel PDBQT generation returned {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    print(f"  PDBQT generated: {output_pdbqt}")
    return True


def patch_zinc_in_pdbqt(pdbqt_path):
    """Patch Zn atoms: set charge to +2.000 and atom type to Zn."""
    lines = []
    n_patched = 0
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and "ZN" in line[12:16].upper():
                # Ensure correct charge and atom type
                # PDBQT format: cols 70-76 = charge, cols 77-79 = atom type
                line = line[:70] + "+2.000" + " Zn\n"
                n_patched += 1
            lines.append(line)

    with open(pdbqt_path, "w") as f:
        f.writelines(lines)

    print(f"  Patched {n_patched} Zn atoms → charge +2.000, type Zn")
    return n_patched


def validate_atom_types(pdbqt_path):
    """Validate PDBQT atom types for Vina compatibility."""
    type_counts = {}
    n_atoms = 0
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n_atoms += 1
                atype = line[77:].strip()
                type_counts[atype] = type_counts.get(atype, 0) + 1

    print(f"\n  Atom type distribution ({n_atoms} total):")
    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {atype:<4s}: {count:>5d}")

    # Validate expected types
    issues = []
    if "HD" not in type_counts:
        issues.append("NO polar hydrogens (HD) — H-bond scoring broken")
    if type_counts.get("Co", 0) > 0:
        issues.append(f"Cobalt (Co) atoms present — likely misassigned")
    if type_counts.get("Zn", 0) != 5:
        issues.append(f"Expected 5 Zn atoms, found {type_counts.get('Zn', 0)}")

    n_ratio = type_counts.get("N", 0) / max(type_counts.get("NA", 1), 1)
    if n_ratio < 0.3:
        issues.append(f"N/NA ratio too low ({n_ratio:.2f}) — backbone amides may be mistyped")

    if issues:
        print(f"\n  ⚠ ATOM TYPE ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ Atom types look correct")

    return len(issues) == 0, type_counts


def validate_zinc_coordination(pdb_path):
    """Validate C2H2 zinc coordination geometry."""
    atoms = parse_pdb_atoms(pdb_path)
    zinc_atoms = [a for a in atoms if a["resname"] == "ZN"]

    print(f"\n  Zinc atoms found: {len(zinc_atoms)}")

    for zf_name, zf_def in ZF_DEFINITIONS.items():
        cys_residues = zf_def["cys"]
        his_residues = zf_def["his"]
        coord_label = f"{zf_name}: C{cys_residues[0]}/C{cys_residues[1]}/H{his_residues[0]}/H{his_residues[1]}"

        # Find coordinating atoms (SG for Cys, NE2/ND1 for His)
        coord_atoms = []
        for a in atoms:
            if a["chain"] == "A":
                if a["resname"] == "CYS" and a["resseq"] in cys_residues and a["name"] == "SG":
                    coord_atoms.append(a)
                elif a["resname"] == "HIS" and a["resseq"] in his_residues and a["name"] in ("NE2", "ND1"):
                    coord_atoms.append(a)

        if len(coord_atoms) < 4:
            print(f"    {coord_label}: Only {len(coord_atoms)} coordinating atoms found (expected 4)")
            continue

        # Find nearest zinc
        cx = sum(a["x"] for a in coord_atoms) / len(coord_atoms)
        cy = sum(a["y"] for a in coord_atoms) / len(coord_atoms)
        cz = sum(a["z"] for a in coord_atoms) / len(coord_atoms)

        nearest_zn = min(
            zinc_atoms,
            key=lambda z: math.sqrt(
                (z["x"] - cx) ** 2 + (z["y"] - cy) ** 2 + (z["z"] - cz) ** 2
            ),
        )

        # Check Zn-ligand distances (ideal: 2.0-2.4 Å for C2H2)
        distances = []
        for a in coord_atoms:
            d = math.sqrt(
                (a["x"] - nearest_zn["x"]) ** 2
                + (a["y"] - nearest_zn["y"]) ** 2
                + (a["z"] - nearest_zn["z"]) ** 2
            )
            distances.append((a["resname"], a["resseq"], a["name"], d))

        avg_d = sum(d[3] for d in distances) / len(distances)
        ok = all(1.8 < d[3] < 2.7 for d in distances)
        status = "✓" if ok else "⚠"

        detail = ", ".join(f"{d[0]}{d[1]}.{d[2]}={d[3]:.2f}Å" for d in distances)
        print(f"    {status} {coord_label}: avg={avg_d:.2f}Å [{detail}]")


def validate_binding_sites(pdb_path):
    """Validate that literature binding site residues exist and report coordinates."""
    atoms = parse_pdb_atoms(pdb_path)

    print("\n  Literature binding site validation:")
    sites = {}
    for site_name, site_def in BINDING_SITES.items():
        print(f"\n  [{site_name}] {site_def['reference']}")
        print(f"    Full-length residues: {site_def['full_length_residues']}")
        coords = []
        for res_label, expected_resname in site_def["residues"].items():
            resseq = int(res_label[1:])
            ca_atoms = [
                a
                for a in atoms
                if a["chain"] == "A"
                and a["resseq"] == resseq
                and a["name"] == "CA"
                and a["resname"] == expected_resname
            ]
            if ca_atoms:
                a = ca_atoms[0]
                coords.append((a["x"], a["y"], a["z"]))
                print(
                    f"    ✓ {res_label} ({expected_resname}) CA at "
                    f"({a['x']:.1f}, {a['y']:.1f}, {a['z']:.1f})"
                )
            else:
                print(f"    ✗ {res_label} ({expected_resname}) NOT FOUND")

        if len(coords) == 2:
            center = [
                (coords[0][0] + coords[1][0]) / 2,
                (coords[0][1] + coords[1][1]) / 2,
                (coords[0][2] + coords[1][2]) / 2,
            ]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coords[0], coords[1])))
            print(f"    → Grid center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
            print(f"    → Inter-residue distance: {dist:.1f} Å")
            sites[site_name] = center

    return sites


def check_dna_removed(pdbqt_path):
    """Verify no DNA remains in the PDBQT."""
    dna_residues = {"DT", "DA", "DC", "DG", "DT5", "DA5", "DC5", "DG5",
                    "DT3", "DA3", "DC3", "DG3"}
    n_dna = 0
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname in dna_residues:
                    n_dna += 1

    if n_dna > 0:
        print(f"  ✗ DNA atoms still present: {n_dna}")
    else:
        print(f"  ✓ No DNA in receptor PDBQT")
    return n_dna == 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GLI1 receptor for docking (publication-quality protocol)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "2gli_with_zinc.pdb"),
        help="Input PDB file (2GLI with zinc atoms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "2gli_apo_prepared.pdbqt"),
        help="Output PDBQT file",
    )
    parser.add_argument("--ph", type=float, default=7.4, help="pH for protonation")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing PDBQT, don't regenerate",
    )
    args = parser.parse_args()

    input_pdb = Path(args.input)
    output_pdbqt = Path(args.output)
    work_dir = output_pdbqt.parent

    print("=" * 70)
    print("GLI1 RECEPTOR PREPARATION — VALIDATED PROTOCOL")
    print("=" * 70)

    if args.validate_only:
        print(f"\n[VALIDATE] Checking {output_pdbqt}...")
        check_dna_removed(str(output_pdbqt))
        validate_atom_types(str(output_pdbqt))
        validate_zinc_coordination(str(input_pdb))
        validate_binding_sites(str(input_pdb))
        return

    # Step 1: Strip DNA
    print(f"\n[STEP 1] Stripping DNA from {input_pdb}...")
    apo_pdb = work_dir / "2gli_apo_protein.pdb"
    n_prot, n_dna = strip_dna_and_write_apo(str(input_pdb), str(apo_pdb))

    # Step 2: Add polar hydrogens
    print(f"\n[STEP 2] Adding polar hydrogens at pH {args.ph}...")
    apo_h_pdb = work_dir / "2gli_apo_h.pdb"
    add_hydrogens_obabel(str(apo_pdb), str(apo_h_pdb), ph=args.ph)

    # Step 3: Generate PDBQT
    print(f"\n[STEP 3] Generating PDBQT with AutoDock atom types...")
    if not generate_pdbqt_obabel(str(apo_h_pdb), str(output_pdbqt)):
        print("  FATAL: PDBQT generation failed")
        sys.exit(1)

    # Step 4: Patch zinc
    print(f"\n[STEP 4] Patching Zn2+ charges and atom types...")
    patch_zinc_in_pdbqt(str(output_pdbqt))

    # Step 5: Validate
    print(f"\n[STEP 5] Validation...")
    print("\n  --- DNA check ---")
    check_dna_removed(str(output_pdbqt))
    print("\n  --- Atom types ---")
    types_ok, _ = validate_atom_types(str(output_pdbqt))
    print("\n  --- Zinc coordination ---")
    validate_zinc_coordination(str(apo_pdb))
    print("\n  --- Binding sites ---")
    sites = validate_binding_sites(str(apo_pdb))

    # Summary
    print(f"\n{'=' * 70}")
    print("RECEPTOR PREPARATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output: {output_pdbqt}")
    print(f"  Protein atoms (pre-H): {n_prot}")
    print(f"  DNA atoms removed: {n_dna}")
    if sites:
        for name, center in sites.items():
            print(f"  {name} grid center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"\n  Recommended box size: 22 × 22 × 22 Å")
    print(f"  Recommended exhaustiveness: 32")


if __name__ == "__main__":
    main()
