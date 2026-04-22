#!/usr/bin/env python3
"""
Plot MD simulation results: RMSD, RMSF, Radius of Gyration
Reads GROMACS .xvg files from analysis directories.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPOUND_LABELS = {
    "CNP0150834_0": "CNP0150834 (ZF2-3, novel)",
    "CNP0544084_0": "CNP0544084 (ZF2-3, novel)",
    "CNP0179919_0": "CNP0179919 (ZF4-5, novel)",
    "CNP0592286_0": "CNP0592286 (ZF4-5, novel)",
    "CNP0367428_0": "CNP0367428 (ZF4-5, novel)",
    "GANT61-D_0":   "GANT61-D (ZF2-3, reference)",
    "GlaB_0":       "GlaB (ZF4-5, reference)",
}

# Order: novel ZF2-3 (stable), novel ZF4-5, references
COMPOUND_ORDER = [
    "CNP0150834_0", "CNP0544084_0",
    "CNP0179919_0", "CNP0592286_0", "CNP0367428_0",
    "GANT61-D_0", "GlaB_0",
]

COLORS = {
    "CNP0150834_0": "#2196F3",  # blue - stable novel
    "CNP0544084_0": "#4CAF50",  # green - rebinding novel
    "CNP0179919_0": "#FF9800",  # orange - dissociating
    "CNP0592286_0": "#FF5722",  # deep orange - dissociating
    "CNP0367428_0": "#9C27B0",  # purple - dissociating
    "GANT61-D_0":   "#E91E63",  # pink - reference
    "GlaB_0":       "#795548",  # brown - reference
}

LINESTYLES = {
    "CNP0150834_0": "-", "CNP0544084_0": "-",
    "CNP0179919_0": "-", "CNP0592286_0": "-", "CNP0367428_0": "-",
    "GANT61-D_0": "--", "GlaB_0": "--",
}


def parse_xvg(filepath):
    """Parse GROMACS .xvg file, return numpy arrays of x, y columns."""
    x, y = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    return np.array(x), np.array(y)


def parse_xvg_multi(filepath):
    """Parse .xvg with multiple y columns (e.g., gyration)."""
    x, ys = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                ys.append([float(v) for v in parts[1:]])
    return np.array(x), np.array(ys)


def find_analysis_dirs():
    """Find all compound analysis directories, ordered by COMPOUND_ORDER."""
    found = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.endswith("_analysis"):
            cid = d.name.replace("_analysis", "")
            found[cid] = d
    # Return in canonical order
    ordered = {}
    for cid in COMPOUND_ORDER:
        if cid in found:
            ordered[cid] = found[cid]
    # Add any not in order list
    for cid in sorted(found):
        if cid not in ordered:
            ordered[cid] = found[cid]
    return ordered


def plot_rmsd_backbone(dirs):
    """RMSD of protein backbone over time for all compounds."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for cid, adir in dirs.items():
        f = adir / "rmsd_backbone.xvg"
        if not f.exists():
            continue
        t, rmsd = parse_xvg(f)
        label = COMPOUND_LABELS.get(cid, cid)
        ax.plot(t, rmsd, color=COLORS.get(cid, '#333'),
                linestyle=LINESTYLES.get(cid, '-'),
                alpha=0.8, linewidth=0.7, label=label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (nm)")
    ax.set_title("Protein Backbone RMSD — GLI1 Zinc Finger + Ligand Complexes")
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsd_backbone_all.png")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'rmsd_backbone_all.png'}")


def plot_rmsd_ligand(dirs):
    """RMSD of ligand over time for all compounds."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for cid, adir in dirs.items():
        f = adir / "rmsd_ligand.xvg"
        if not f.exists():
            continue
        t, rmsd = parse_xvg(f)
        label = COMPOUND_LABELS.get(cid, cid)
        ax.plot(t, rmsd, color=COLORS.get(cid, '#333'),
                linestyle=LINESTYLES.get(cid, '-'),
                alpha=0.8, linewidth=0.7, label=label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (nm)")
    ax.set_title("Ligand RMSD — GLI1 Zinc Finger Binding Pocket")
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsd_ligand_all.png")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'rmsd_ligand_all.png'}")


def plot_rmsf(dirs):
    """Per-residue RMSF for all compounds."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for cid, adir in dirs.items():
        f = adir / "rmsf.xvg"
        if not f.exists():
            continue
        res, rmsf = parse_xvg(f)
        label = COMPOUND_LABELS.get(cid, cid)
        ax.plot(res, rmsf, color=COLORS.get(cid, '#333'),
                linestyle=LINESTYLES.get(cid, '-'),
                alpha=0.7, linewidth=0.8, label=label)
    ax.set_xlabel("Residue Number")
    ax.set_ylabel("RMSF (nm)")
    ax.set_title("Per-Residue RMSF — GLI1 Zinc Finger + Ligand Complexes")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsf_all.png")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'rmsf_all.png'}")


def plot_gyration(dirs):
    """Radius of gyration for ligands over time."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for cid, adir in dirs.items():
        f = adir / "gyrate.xvg"
        if not f.exists():
            continue
        t, ys = parse_xvg_multi(f)
        rg = ys[:, 0]  # total Rg is first column
        label = COMPOUND_LABELS.get(cid, cid)
        ax.plot(t, rg, color=COLORS.get(cid, '#333'),
                linestyle=LINESTYLES.get(cid, '-'),
                alpha=0.8, linewidth=0.6, label=label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Radius of Gyration (nm)")
    ax.set_title("Ligand Radius of Gyration — GLI1 Zinc Finger Complexes")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gyration_all.png")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'gyration_all.png'}")


def plot_rmsd_individual(dirs):
    """Individual RMSD panels (backbone + ligand) per compound."""
    for cid, adir in dirs.items():
        bb = adir / "rmsd_backbone.xvg"
        lig = adir / "rmsd_ligand.xvg"
        if not bb.exists() or not lig.exists():
            continue

        color = COLORS.get(cid, '#333')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        label = COMPOUND_LABELS.get(cid, cid)

        t_bb, rmsd_bb = parse_xvg(bb)
        ax1.plot(t_bb, rmsd_bb, color=color, linewidth=0.6)
        ax1.set_ylabel("Backbone RMSD (nm)")
        ax1.set_title(f"{label} — 50 ns MD Simulation")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        mean_bb = np.mean(rmsd_bb[len(rmsd_bb)//5:])
        ax1.axhline(mean_bb, color='gray', linestyle='--', alpha=0.6,
                     label=f'Mean (equil.): {mean_bb:.3f} nm')
        ax1.legend(fontsize=9)

        t_lig, rmsd_lig = parse_xvg(lig)
        ax2.plot(t_lig, rmsd_lig, color=color, linewidth=0.6)
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Ligand RMSD (nm)")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=0)
        mean_lig = np.mean(rmsd_lig[len(rmsd_lig)//5:])
        ax2.axhline(mean_lig, color='gray', linestyle='--', alpha=0.6,
                     label=f'Mean (equil.): {mean_lig:.3f} nm')
        ax2.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"rmsd_{cid}.png")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / f'rmsd_{cid}.png'}")


def plot_binding_comparison(dirs):
    """Side-by-side comparison: novel ZF2-3 binders vs references."""
    compare = {
        "CNP0150834_0": "CNP0150834 (novel)",
        "CNP0544084_0": "CNP0544084 (novel)",
        "GANT61-D_0": "GANT61-D (reference)",
        "GlaB_0": "GlaB (reference)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax_i, (cid, label) in enumerate(compare.items()):
        adir = dirs.get(cid)
        if not adir:
            continue
        r, c = divmod(ax_i, 2)
        ax = axes[r][c]
        lig_f = adir / "rmsd_ligand.xvg"
        bb_f = adir / "rmsd_backbone.xvg"
        if lig_f.exists():
            t, rmsd = parse_xvg(lig_f)
            ax.plot(t, rmsd, color=COLORS.get(cid, '#333'), linewidth=0.6,
                    label='Ligand RMSD')
        if bb_f.exists():
            t, rmsd = parse_xvg(bb_f)
            ax.plot(t, rmsd, color='gray', linewidth=0.5, alpha=0.6,
                    label='Backbone RMSD')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('RMSD (nm)')
        ax.set_xlim(0, 50)
        ax.set_ylim(0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Binding Stability Comparison — Novel Hits vs Known Inhibitors',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binding_comparison_panel.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'binding_comparison_panel.png'}")


def print_summary(dirs):
    """Print summary statistics."""
    print("\n" + "="*65)
    print("MD SIMULATION RESULTS SUMMARY")
    print("="*65)
    for cid, adir in dirs.items():
        label = COMPOUND_LABELS.get(cid, cid)
        print(f"\n  {label}:")

        bb = adir / "rmsd_backbone.xvg"
        if bb.exists():
            t, rmsd = parse_xvg(bb)
            eq = rmsd[len(rmsd)//5:]  # skip first 20%
            print(f"    Backbone RMSD: {np.mean(eq):.3f} ± {np.std(eq):.3f} nm "
                  f"(equil. mean ± SD)")

        lig = adir / "rmsd_ligand.xvg"
        if lig.exists():
            t, rmsd = parse_xvg(lig)
            eq = rmsd[len(rmsd)//5:]
            print(f"    Ligand RMSD:   {np.mean(eq):.3f} ± {np.std(eq):.3f} nm")

        rmsf = adir / "rmsf.xvg"
        if rmsf.exists():
            res, vals = parse_xvg(rmsf)
            print(f"    RMSF range:    {np.min(vals):.3f} – {np.max(vals):.3f} nm")

        gy = adir / "gyrate.xvg"
        if gy.exists():
            t, ys = parse_xvg_multi(gy)
            rg = ys[:, 0]
            eq = rg[len(rg)//5:]
            print(f"    Rg (ligand):   {np.mean(eq):.3f} ± {np.std(eq):.3f} nm")


if __name__ == "__main__":
    print("Plotting MD simulation results...")
    dirs = find_analysis_dirs()
    if not dirs:
        print(f"No analysis directories found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(dirs)} compounds: {', '.join(dirs.keys())}")

    plot_rmsd_backbone(dirs)
    plot_rmsd_ligand(dirs)
    plot_rmsf(dirs)
    plot_gyration(dirs)
    plot_rmsd_individual(dirs)
    plot_binding_comparison(dirs)
    print_summary(dirs)
    print(f"\nAll figures saved to: {OUT_DIR}")
