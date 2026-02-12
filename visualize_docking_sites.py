"""
PyMOL visualization script for GLI1 zinc finger docking site validation.

Run in PyMOL:  run visualize_docking_sites.py

This script:
1. Loads the 2GLI structure (apo protein with zinc)
2. Colors zinc fingers individually (ZF1-ZF5)
3. Highlights literature-validated binding site residues:
   - ZF2-3 site: E119/E167 (GANT61 — Agyeman et al. 2014 Oncotarget)
   - ZF4-5 site: K209/K219 (GlaB — Infante et al. 2015 EMBO J)
4. Shows docking grid boxes as CGO wireframes
5. Displays zinc coordination geometry
6. Labels everything for publication-quality figures

PDB-to-full-length mapping (2GLI → UniProt P08151):
  full_length = PDB_resnum + 131
  PDBe confirms: 2GLI covers residues 234-388 of P08151

References:
  Pavletich & Pabo, Science 261:1701 (1993) — 2GLI crystal structure
  Agyeman et al., Oncotarget 5:4492 (2014) — GANT61 site E119/E167
  Infante et al., EMBO J 34:200 (2015) — GlaB site K340/K350 (=K209/K219 PDB)
"""

from pymol import cmd
from pymol.cgo import *
import os

# ─── Configuration ───────────────────────────────────────────────────────
PDB_FILE = os.path.join(os.path.dirname(__file__), "2gli_apo_protein.pdb")
# Fall back to original PDB if apo not available
if not os.path.exists(PDB_FILE):
    PDB_FILE = os.path.join(os.path.dirname(__file__), "2gli_with_zinc.pdb")

# Zinc finger residue ranges (PDB numbering)
ZF_RANGES = {
    "ZF1": (103, 131),
    "ZF2": (135, 164),
    "ZF3": (168, 194),
    "ZF4": (198, 225),
    "ZF5": (229, 257),
}

# Zinc coordinating residues (PDB numbering)
ZN_COORD = {
    "ZF1": {"cys": [106, 111], "his": [129, 131]},
    "ZF2": {"cys": [139, 144], "his": [160, 164]},
    "ZF3": {"cys": [172, 177], "his": [190, 194]},
    "ZF4": {"cys": [202, 207], "his": [220, 225]},
    "ZF5": {"cys": [233, 238], "his": [251, 256]},
}

# Literature binding sites
SITES = {
    "ZF2-3": {
        "residues": [119, 167],
        "resnames": ["GLU", "GLU"],
        "center": (-32.6, -5.7, -0.6),
        "box_size": 22.0,
        "color": [0.2, 0.6, 1.0],  # blue
        "label": "GANT61 site\n(E119/E167 PDB = E250/E298 FL)\nAgyeman et al. 2014",
    },
    "ZF4-5": {
        "residues": [209, 219],
        "resnames": ["LYS", "LYS"],
        "center": (-5.1, 9.9, 11.5),
        "box_size": 22.0,
        "color": [1.0, 0.4, 0.2],  # orange
        "label": "GlaB site\n(K209/K219 PDB = K340/K350 FL)\nInfante et al. 2015",
    },
}

# ZF colors (colorblind-friendly palette)
ZF_COLORS = {
    "ZF1": [0.7, 0.7, 0.7],   # gray (no DNA contact)
    "ZF2": [0.3, 0.7, 0.3],   # green
    "ZF3": [0.2, 0.5, 0.8],   # blue
    "ZF4": [0.9, 0.6, 0.1],   # orange
    "ZF5": [0.8, 0.2, 0.3],   # red
}


def make_box_cgo(name, center, size, color, linewidth=3.0):
    """Create a wireframe box as CGO object."""
    cx, cy, cz = center
    s = size / 2.0

    # 8 corners
    corners = [
        (cx - s, cy - s, cz - s), (cx + s, cy - s, cz - s),
        (cx + s, cy + s, cz - s), (cx - s, cy + s, cz - s),
        (cx - s, cy - s, cz + s), (cx + s, cy - s, cz + s),
        (cx + s, cy + s, cz + s), (cx - s, cy + s, cz + s),
    ]

    # 12 edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]

    obj = [LINEWIDTH, linewidth, BEGIN, LINES, COLOR] + color
    for i, j in edges:
        obj += [VERTEX] + list(corners[i]) + [VERTEX] + list(corners[j])
    obj += [END]

    cmd.load_cgo(obj, name)


def setup():
    """Main visualization setup."""

    # ── Load structure ──
    cmd.load(PDB_FILE, "GLI1")
    cmd.remove("resn HOH")  # remove waters

    # Check if DNA is present and remove for clarity
    cmd.remove("chain C or chain D")

    cmd.hide("everything", "GLI1")
    cmd.show("cartoon", "GLI1 and chain A")
    cmd.set("cartoon_transparency", 0.3)

    # ── Color zinc fingers individually ──
    for zf_name, (start, end) in ZF_RANGES.items():
        sel_name = zf_name
        cmd.select(sel_name, f"GLI1 and chain A and resi {start}-{end}")
        cmd.set_color(f"{zf_name}_color", ZF_COLORS[zf_name])
        cmd.color(f"{zf_name}_color", sel_name)

    # ── Show zinc atoms ──
    cmd.select("zinc_atoms", "GLI1 and name ZN")
    cmd.show("spheres", "zinc_atoms")
    cmd.set("sphere_scale", 0.6, "zinc_atoms")
    cmd.color("gray50", "zinc_atoms")

    # ── Show zinc coordination ──
    for zf_name, coord in ZN_COORD.items():
        for c in coord["cys"]:
            sel = f"GLI1 and chain A and resi {c} and name SG"
            cmd.show("sticks", sel)
            cmd.color("yellow", sel)
        for h in coord["his"]:
            sel = f"GLI1 and chain A and resi {h} and (name NE2 or name ND1 or name CE1 or name CD2 or name CG)"
            cmd.show("sticks", sel)
            cmd.color("marine", sel)

    # ── Show and label binding sites ──
    for site_name, site_def in SITES.items():
        color = site_def["color"]
        cmd.set_color(f"{site_name}_color", color)

        # Highlight binding site residues
        resi_list = "+".join(str(r) for r in site_def["residues"])
        sel = f"GLI1 and chain A and resi {resi_list}"
        cmd.show("sticks", sel)
        cmd.color(f"{site_name}_color", sel)
        cmd.label(f"{sel} and name CA", "'%s%s (PDB)' % (resn, resi)")

        # Draw grid box
        make_box_cgo(
            f"box_{site_name}",
            site_def["center"],
            site_def["box_size"],
            color,
            linewidth=2.5,
        )

        # Add pseudoatom at center for labeling
        cx, cy, cz = site_def["center"]
        cmd.pseudoatom(
            f"center_{site_name}",
            pos=[cx, cy, cz],
            label=f"{site_name}",
        )
        cmd.show("spheres", f"center_{site_name}")
        cmd.set("sphere_scale", 0.4, f"center_{site_name}")
        cmd.color(f"{site_name}_color", f"center_{site_name}")

    # ── Linker regions ──
    for start, end, label in [(132, 138, "L1-2"), (165, 171, "L2-3"),
                               (195, 201, "L3-4"), (226, 232, "L4-5")]:
        cmd.select(f"linker_{label}", f"GLI1 and chain A and resi {start}-{end}")
        cmd.color("white", f"linker_{label}")

    # ── Rendering settings ──
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("label_size", 14)
    cmd.set("label_color", "black")
    cmd.set("label_font_id", 7)
    cmd.set("stick_radius", 0.15)
    cmd.set("sphere_transparency", 0.0)

    # ── Camera: overview ──
    cmd.zoom("GLI1 and chain A", buffer=5)
    cmd.orient("GLI1 and chain A")

    # ── Clean up selections ──
    cmd.deselect()

    print("\n" + "=" * 60)
    print("GLI1 DOCKING SITE VISUALIZATION LOADED")
    print("=" * 60)
    print("\nColor key:")
    print("  ZF1 (gray)   — no DNA contact, disordered")
    print("  ZF2 (green)  — DNA contact, part of GANT61 site")
    print("  ZF3 (blue)   — DNA contact, part of GANT61 site")
    print("  ZF4 (orange) — DNA contact, part of GlaB site")
    print("  ZF5 (red)    — DNA contact, part of GlaB site")
    print("\nBinding sites:")
    print("  Blue box:   ZF2-3 (GANT61) — E119/E167 (PDB) = E250/E298 (full-length)")
    print("  Orange box: ZF4-5 (GlaB)   — K209/K219 (PDB) = K340/K350 (full-length)")
    print("\nZinc atoms: gray spheres")
    print("Cys coordination: yellow sticks")
    print("His coordination: marine sticks")
    print("\nUseful commands:")
    print("  zoom box_ZF2-3     # Focus on GANT61 site")
    print("  zoom box_ZF4-5     # Focus on GlaB site")
    print("  set cartoon_transparency, 0  # Opaque cartoon")
    print("  ray 2400, 1800     # High-res render")


# Run automatically when script is loaded
setup()
