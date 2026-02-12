"""
PyMOL visualization: GLI1 docking sites + top hit poses.

Run in PyMOL:  run visualize_top_hits.py

Loads:
  1. Apo GLI1 structure with zinc fingers colored individually
  2. Literature-validated grid boxes (ZF2-3, ZF4-5)
  3. Top docked ligand poses from the corrected docking run (v2)

Usage:
  After loading, try these views:
    zoom box_ZF2-3          # Focus on GANT61 site
    zoom box_ZF4-5          # Focus on GlaB site
    zoom CNP0140643_zf23    # Focus on top hit
    set cartoon_transparency, 0   # Opaque protein
    ray 2400, 1800          # Render hi-res image
"""

from pymol import cmd
from pymol.cgo import *
import os
import glob

PROJECT_DIR = "/Users/aaranchahal/onnat"
# Use full complex (with DNA) for visualization context
RECEPTOR_PDB = os.path.join(PROJECT_DIR, "2gli_with_zinc.pdb")
POSES_DIR = os.path.join(PROJECT_DIR, "top_poses")

# ── Zinc finger ranges (PDB numbering) ──
ZF_RANGES = {
    "ZF1": (103, 131), "ZF2": (135, 164), "ZF3": (168, 194),
    "ZF4": (198, 225), "ZF5": (229, 257),
}
ZF_COLORS = {
    "ZF1": [0.7, 0.7, 0.7], "ZF2": [0.3, 0.7, 0.3], "ZF3": [0.2, 0.5, 0.8],
    "ZF4": [0.9, 0.6, 0.1], "ZF5": [0.8, 0.2, 0.3],
}
ZN_COORD = {
    "ZF1": {"cys": [106, 111], "his": [129, 131]},
    "ZF2": {"cys": [139, 144], "his": [160, 164]},
    "ZF3": {"cys": [172, 177], "his": [190, 194]},
    "ZF4": {"cys": [202, 207], "his": [220, 225]},
    "ZF5": {"cys": [233, 238], "his": [251, 256]},
}

# ── Binding sites ──
SITES = {
    "ZF2-3": {
        "key_resi": [119, 167],
        "center": (-32.6, -5.7, -0.6),
        "box": 22.0,
        "color": [0.2, 0.6, 1.0],
        "desc": "GANT61 site (E119/E167 PDB = E250/E298 FL)",
    },
    "ZF4-5": {
        "key_resi": [209, 219],
        "center": (-5.1, 9.9, 11.5),
        "box": 22.0,
        "color": [1.0, 0.4, 0.2],
        "desc": "GlaB site (K209/K219 PDB = K340/K350 FL)",
    },
}

# ── Ligand colors (for up to 6 ligands) ──
LIG_COLORS = [
    ("hit1_c", [1.0, 0.2, 0.6]),   # magenta
    ("hit2_c", [0.0, 0.9, 0.9]),   # cyan
    ("hit3_c", [1.0, 0.85, 0.0]),  # gold
    ("hit4_c", [0.5, 1.0, 0.3]),   # lime
    ("hit5_c", [0.7, 0.3, 1.0]),   # purple
    ("hit6_c", [1.0, 0.5, 0.0]),   # orange
]


def make_box(name, center, size, color, lw=2.5):
    cx, cy, cz = center
    s = size / 2.0
    c = [
        (cx-s,cy-s,cz-s),(cx+s,cy-s,cz-s),(cx+s,cy+s,cz-s),(cx-s,cy+s,cz-s),
        (cx-s,cy-s,cz+s),(cx+s,cy-s,cz+s),(cx+s,cy+s,cz+s),(cx-s,cy+s,cz+s),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    obj = [LINEWIDTH, lw, BEGIN, LINES, COLOR] + color
    for i, j in edges:
        obj += [VERTEX]+list(c[i])+[VERTEX]+list(c[j])
    obj += [END]
    cmd.load_cgo(obj, name)


def setup():
    # ── Load receptor + DNA complex ──
    cmd.load(RECEPTOR_PDB, "GLI1")
    cmd.remove("resn HOH")
    cmd.hide("everything", "GLI1")
    cmd.show("cartoon", "GLI1 and chain A")
    cmd.set("cartoon_transparency", 0.35)

    # ── Show DNA for spatial reference ──
    cmd.show("cartoon", "GLI1 and (chain C or chain D)")
    cmd.color("lightorange", "GLI1 and (chain C or chain D)")
    cmd.set("cartoon_transparency", 0.45, "GLI1 and (chain C or chain D)")
    cmd.set("cartoon_tube_radius", 0.4, "GLI1 and (chain C or chain D)")

    # ── Color zinc fingers ──
    for zf, (s, e) in ZF_RANGES.items():
        cmd.select(zf, "GLI1 and chain A and resi %d-%d" % (s, e))
        cmd.set_color(zf+"_color", ZF_COLORS[zf])
        cmd.color(zf+"_color", zf)

    # ── Linkers white ──
    for s, e in [(132,138),(165,171),(195,201),(226,232)]:
        cmd.color("white", "GLI1 and chain A and resi %d-%d" % (s, e))

    # ── Zinc spheres ──
    cmd.select("zinc_atoms", "GLI1 and name ZN")
    cmd.show("spheres", "zinc_atoms")
    cmd.set("sphere_scale", 0.5, "zinc_atoms")
    cmd.color("gray50", "zinc_atoms")

    # ── Zinc coordination sticks ──
    for zf, coord in ZN_COORD.items():
        for c in coord["cys"]:
            cmd.show("sticks", "GLI1 and chain A and resi %d and name SG+CB" % c)
            cmd.color("yellow", "GLI1 and chain A and resi %d and name SG" % c)
        for h in coord["his"]:
            cmd.show("sticks", "GLI1 and chain A and resi %d and (name NE2+ND1+CE1+CD2+CG)" % h)

    # ── Binding site residues + grid boxes ──
    for site_name, sd in SITES.items():
        cmd.set_color(site_name+"_c", sd["color"])
        resi_str = "+".join(str(r) for r in sd["key_resi"])
        cmd.show("sticks", "GLI1 and chain A and resi %s" % resi_str)
        cmd.color(site_name+"_c", "GLI1 and chain A and resi %s" % resi_str)
        cmd.label("GLI1 and chain A and resi %s and name CA" % resi_str,
                  "'%s%s' % (resn, resi)")
        make_box("box_"+site_name, sd["center"], sd["box"], sd["color"])
        cmd.pseudoatom("ctr_"+site_name, pos=list(sd["center"]),
                       label=site_name)
        cmd.show("spheres", "ctr_"+site_name)
        cmd.set("sphere_scale", 0.35, "ctr_"+site_name)
        cmd.color(site_name+"_c", "ctr_"+site_name)

    # ── Load docked poses ──
    if os.path.isdir(POSES_DIR):
        pose_files = sorted(glob.glob(os.path.join(POSES_DIR, "*.pdbqt")))
        print("\nLoading %d docked poses from %s" % (len(pose_files), POSES_DIR))
        for idx, pf in enumerate(pose_files):
            basename = os.path.splitext(os.path.basename(pf))[0]
            obj_name = basename.replace(".", "_")
            cmd.load(pf, obj_name, state=1)  # load only best pose (model 1)
            cmd.hide("everything", obj_name)
            cmd.show("sticks", obj_name)
            cname, cval = LIG_COLORS[idx % len(LIG_COLORS)]
            cmd.set_color(cname + str(idx), cval)
            cmd.color(cname + str(idx), obj_name)
            cmd.set("stick_radius", 0.2, obj_name)

            # Read score from REMARK
            score = "?"
            with open(pf) as fh:
                for line in fh:
                    if "VINA RESULT" in line:
                        score = line.split()[3]
                        break

            # Determine which site
            site_tag = "ZF2-3" if "zf23" in basename else "ZF4-5"
            print("  %s  (%s: %s kcal/mol) -> %s" % (
                obj_name, site_tag, score, cname))
    else:
        print("\nNo top_poses/ directory found — run docking first")

    # ── Rendering ──
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("label_size", 14)
    cmd.set("label_color", "black")
    cmd.set("label_font_id", 7)
    cmd.set("stick_radius", 0.12)
    cmd.set("surface_quality", 1)

    cmd.zoom("GLI1 and chain A", buffer=5)
    cmd.orient("GLI1 and chain A")
    cmd.deselect()

    print("\n" + "="*60)
    print("VISUALIZATION LOADED")
    print("="*60)
    print("\nColor key:")
    print("  ZF1=gray  ZF2=green  ZF3=blue  ZF4=orange  ZF5=red")
    print("  Blue box = ZF2-3 (GANT61)   Orange box = ZF4-5 (GlaB)")
    print("  Ligands = magenta/cyan/gold/lime/purple sticks")
    print("\nTips:")
    print("  zoom box_ZF2-3              # focus GANT61 site")
    print("  zoom CNP0140643_0_zf23      # focus top hit")
    print("  show surface, GLI1          # show protein surface")
    print("  set transparency, 0.6, GLI1 # translucent surface")
    print("  ray 2400, 1800              # hi-res render")


setup()
