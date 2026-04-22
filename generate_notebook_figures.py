#!/usr/bin/env python3
"""Generate all 10 GSDSEF Notebook figures — publication quality.

Output: outputs/notebook_figures/fig{01..10}_*.png (300 DPI)
Matches the 10 [PLACEHOLDER] slots in GSDSEF_NOTEBOOK.md.
"""

import csv, os, sys, warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════
# Global Style — unified across all figures
# ═══════════════════════════════════════════════════════════════════
C = {  # Okabe-Ito colorblind-safe + custom accents
    "blue":   "#0173B2", "orange": "#DE8F05", "green":  "#009E73",
    "vermil": "#D55E00", "sky":    "#56B4E9", "purple": "#CC78BC",
    "yellow": "#ECE133", "grey":   "#999999", "dark":   "#333333",
    "zf23":   "#0173B2", "zf45":   "#DE8F05", "zf13":   "#009E73",
    "miss":   "#D55E00", "hit":    "#0173B2",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": C["dark"], "axes.grid": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
})

OUT = Path("outputs/notebook_figures")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════
def load_loocv(seed, encoder="esm2"):
    path = f"outputs/logs/{encoder}_seed{seed}_stage3_loocv_folds.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["compound"] == "GANT61":
                continue
            rows.append(r)
    return rows

def load_inhibitors():
    sites = {}
    with open("gli_inhibitors.csv") as f:
        for r in csv.DictReader(f):
            s = r["binding_site"]
            if "2-3" in s or "1-3" in s:
                sites[r["compound_name"]] = "ZF2-3"
            else:
                sites[r["compound_name"]] = "ZF4-5"
    return sites

def load_candidates():
    rows = []
    with open("outputs/final_candidates_v3.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Model Architecture Diagram
# ═══════════════════════════════════════════════════════════════════
def fig01_architecture():
    print("Fig 01: Architecture diagram")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 10)
    ax.axis("off")

    def card(x, y, w, h, title, subtitle, color, frozen=False):
        # Solid colored header strip + white body
        header_h = h * 0.38
        body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor="white", edgecolor=color, linewidth=2.5)
        ax.add_patch(body)
        header = FancyBboxPatch((x + 0.05, y + h - header_h - 0.02), w - 0.1, header_h,
                                boxstyle="round,pad=0.08", facecolor=color, edgecolor="none")
        ax.add_patch(header)
        ax.text(x + w/2, y + h - header_h/2, title,
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(x + w/2, y + (h - header_h) * 0.45, subtitle,
                ha="center", va="center", fontsize=9.5, color=C["dark"], linespacing=1.4)
        if frozen:
            ax.text(x + w - 0.15, y + h + 0.12, "FROZEN",
                    ha="right", va="bottom", fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="white", linewidth=1.5))

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=2.5,
                                    connectionstyle="arc3,rad=0"))

    # Title
    ax.text(7, 9.6, "GLI-NT Model Architecture", ha="center", fontsize=22, fontweight="bold", color=C["dark"])

    # Input labels
    ax.text(1.8, 9.1, "GLI1 Protein Sequence\n(1,106 amino acids)", ha="center", fontsize=10, color=C["dark"])
    ax.text(7.0, 9.1, "Ligand SMILES String", ha="center", fontsize=10, color=C["dark"])
    ax.text(12.2, 9.1, "Molecular Structure\n(2D graph)", ha="center", fontsize=10, color=C["dark"])
    arrow(1.8, 8.75, 1.8, 8.35); arrow(7.0, 8.85, 7.0, 8.35); arrow(12.2, 8.75, 12.2, 8.35)

    # Encoder layer
    card(0.3, 6.9, 3.0, 1.4, "ESM-2", "650M parameters\nProtein language model", C["blue"], frozen=True)
    card(5.5, 6.9, 3.0, 1.4, "ChemBERTa", "84M parameters\nSMILES language model", C["green"], frozen=True)
    card(10.7, 6.9, 3.0, 1.4, "Morgan FP", "ECFP4, radius=2\n2,048-bit vector", C["orange"])

    arrow(1.8, 6.9, 1.8, 6.45); arrow(7.0, 6.9, 7.0, 6.45); arrow(12.2, 6.9, 12.2, 6.45)

    # Embedding layer
    card(0.3, 5.1, 3.0, 1.3, "1,280-dim", "Protein embedding", C["blue"])
    card(5.5, 5.1, 3.0, 1.3, "768-dim", "Ligand embedding", C["green"])
    card(10.7, 5.1, 3.0, 1.3, "2,048-bit", "Binary fingerprint", C["orange"])

    arrow(1.8, 5.1, 1.8, 4.65); arrow(7.0, 5.1, 7.0, 4.65); arrow(12.2, 5.1, 12.2, 4.65)

    # Branch projections
    card(0.3, 3.3, 3.0, 1.3, "Protein Branch", "Linear(1280, 256)\nReLU + Dropout(0.3)", C["blue"])
    card(5.5, 3.3, 3.0, 1.3, "Ligand Branch", "Linear(768, 256)\nReLU + Dropout(0.3)", C["green"])
    card(10.7, 3.3, 3.0, 1.3, "Morgan Branch", "Linear(2048, 128)\nReLU + Dropout(0.3)", C["orange"])

    arrow(1.8, 3.3, 4.8, 2.75); arrow(7.0, 3.3, 7.0, 2.75); arrow(12.2, 3.3, 9.2, 2.75)

    # Fusion
    card(3.0, 1.5, 8.0, 1.2, "Hadamard Fusion",
         "concat(P[256], L[256], P\u2299L[256], M[128]) = 896-dim", C["purple"])

    arrow(7.0, 1.5, 7.0, 1.05)

    # MLP output
    card(3.5, -0.3, 7.0, 1.3, "Prediction MLP",
         "896 \u2192 256 \u2192 128 \u2192 1 (sigmoid)\n1,049,729 trainable params | MC Dropout \u00D7 50", C["vermil"])

    save(fig, "fig01_architecture")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Three-Stage Transfer Learning Pipeline
# ═══════════════════════════════════════════════════════════════════
def fig02_pipeline():
    print("Fig 02: Transfer learning pipeline")
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(-0.2, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.2, "Cascaded Transfer Learning Pipeline", ha="center",
            fontsize=20, fontweight="bold", color=C["dark"])

    stages = [
        ("Stage 1", "General Binding",
         "~200,000 BindingDB pairs\nBCE Loss  |  LR = 1e-3  |  BS = 256\n20 epochs  |  Patience 5",
         C["blue"], 0.3),
        ("Stage 2", "ZF Domain Adaptation",
         "~1,500 zinc finger pairs\nBCE Loss  |  LR = 1e-4  |  BS = 32\n50 epochs  |  Patience 10",
         C["green"], 5.15),
        ("Stage 3", "GLI-Specific LOOCV",
         "28 GLI binders (27+1 per fold)\nFocal Loss  |  LR = 5e-5  |  BS = 16\n30 epochs  |  MC Dropout \u00D7 50",
         C["vermil"], 10.0),
    ]

    bw, bh = 4.2, 4.2
    for stage_num, stage_name, desc, color, x in stages:
        # Card body
        body = FancyBboxPatch((x, 0.8), bw, bh, boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color, linewidth=3)
        ax.add_patch(body)
        # Colored header strip
        hh = 1.2
        hdr = FancyBboxPatch((x + 0.08, 0.8 + bh - hh - 0.05), bw - 0.16, hh,
                             boxstyle="round,pad=0.1", facecolor=color, edgecolor="none")
        ax.add_patch(hdr)
        ax.text(x + bw/2, 0.8 + bh - hh/2, f"{stage_num}\n{stage_name}",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white",
                linespacing=1.3)
        ax.text(x + bw/2, 2.4, desc, ha="center", va="center",
                fontsize=10, color=C["dark"], linespacing=1.6)

    # Arrows between stages with "checkpoint" label
    for x in [4.5, 9.35]:
        ax.annotate("", xy=(x + 0.65, 3.0), xytext=(x, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color=C["dark"], lw=3.5))
        pass  # clean arrow, no label needed

    # Bottom annotation
    ax.annotate("", xy=(12.8, 0.35), xytext=(2.5, 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=2, linestyle="--"))
    ax.text(7.5, 0.0, "Data volume decreases     \u2192     Task specificity increases",
            ha="center", fontsize=11, color=C["dark"], fontweight="bold")

    save(fig, "fig02_pipeline")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Screening Funnel
# ═══════════════════════════════════════════════════════════════════
def fig03_funnel():
    print("Fig 03: Screening funnel")
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    ax.text(6, 9.8, "Virtual Screening Pipeline", ha="center",
            fontsize=22, fontweight="bold", color=C["dark"])

    steps = [
        ("Enamine + COCONUT Libraries", "1,176,000", C["blue"], 5.0),
        ("Prescreened (MW, charge, SMILES)", "1,070,832", "#2196F3", 4.4),
        ("ML Ensemble (P \u2265 0.5, 28-fold)", "9,262", C["green"], 3.6),
        ("PAINS-Free + Diverse (Butina)", "500", C["purple"], 2.6),
        ("Docked (Vina, 2 ZF sites)", "499", C["vermil"], 2.0),
    ]
    annotations = [
        "",
        "91.1% pass rate",
        "0.86% selected by model",
        "84.6% novel scaffolds",
        "78 compounds at \u2264 \u22127.0 kcal/mol",
    ]
    bar_h = 1.1
    gap = 0.35
    start_y = 8.3

    for i, (label, count, color, width) in enumerate(steps):
        y = start_y - i * (bar_h + gap)
        cx = 6.0
        left = cx - width / 2

        # Draw trapezoid connecting to next level
        if i < len(steps) - 1:
            next_w = steps[i + 1][3]
            trap_x = [cx - width/2, cx + width/2, cx + next_w/2, cx - next_w/2]
            trap_y = [y, y, y - gap, y - gap]
            ax.fill(trap_x, trap_y, color=color, alpha=0.08)

        # Main bar with solid fill
        bar = FancyBboxPatch((left, y), width, bar_h, boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor="white", linewidth=2, alpha=0.92)
        ax.add_patch(bar)

        # Count (white bold text on colored bar)
        ax.text(cx, y + bar_h * 0.58, count, ha="center", va="center",
                fontsize=20, fontweight="bold", color="white")
        # Label (white text on colored bar, below count)
        ax.text(cx, y + bar_h * 0.22, label, ha="center", va="center",
                fontsize=10, color="white", alpha=0.95)

        # Right-side annotation
        if annotations[i]:
            ax.text(cx + width/2 + 0.25, y + bar_h/2, annotations[i],
                    ha="left", va="center", fontsize=10.5, color=C["dark"], fontweight="bold")

    # Down arrows between bars
    for i in range(len(steps) - 1):
        y_top = start_y - i * (bar_h + gap)
        ax.annotate("", xy=(6, y_top - gap + 0.02), xytext=(6, y_top - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=C["dark"], lw=2))

    save(fig, "fig03_screening_funnel")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-Compound LOOCV (multi-seed, color by binding site)
# ═══════════════════════════════════════════════════════════════════
def fig04_loocv():
    print("Fig 04: Per-compound LOOCV (5 seeds)")
    sites = load_inhibitors()
    all_seeds = {s: load_loocv(s) for s in SEEDS}

    # Aggregate per compound: mean and std across seeds
    cpd_data = defaultdict(list)
    for s, rows in all_seeds.items():
        for r in rows:
            cpd_data[r["compound"]].append(float(r["held_out_prob"]))

    compounds = sorted(cpd_data.keys(), key=lambda c: np.mean(cpd_data[c]))
    means = [np.mean(cpd_data[c]) for c in compounds]
    stds = [np.std(cpd_data[c]) for c in compounds]

    fig, ax = plt.subplots(figsize=(10, 10))
    y_pos = np.arange(len(compounds))
    threshold = 0.5

    colors = []
    for c in compounds:
        m = np.mean(cpd_data[c])
        site = sites.get(c, "ZF4-5")
        if m < threshold:
            colors.append(C["miss"])
        elif site == "ZF2-3":
            colors.append(C["zf23"])
        else:
            colors.append(C["zf45"])

    bars = ax.barh(y_pos, means, xerr=stds, height=0.7,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 1.2, "color": "#555555"})

    ax.axvline(x=threshold, color=C["dark"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(threshold + 0.01, -0.8, "Threshold (P = 0.5)",
            fontsize=9, color=C["dark"], va="top")

    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.02, i, f"{m:.2f}", va="center", fontsize=8.5, color="#444444")

    # Count hits/misses
    n_hits = sum(1 for m in means if m >= threshold)
    n_miss = len(means) - n_hits

    ax.set_yticks(y_pos)
    ax.set_yticklabels(compounds, fontsize=9.5)
    ax.set_xlabel("Held-Out P(binder), mean \u00B1 SD across 5 seeds")
    ax.set_title(f"Per-Compound LOOCV Binding Probability\n(ESM-2, 5 seeds, n = {len(compounds)} GLI binders)")
    ax.set_xlim(0, 1.08)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=C["zf23"], lw=8, label=f"HIT \u2014 ZF2-3 site"),
        Line2D([0], [0], color=C["zf45"], lw=8, label=f"HIT \u2014 ZF4-5 site"),
        Line2D([0], [0], color=C["miss"], lw=8, label=f"MISS (P < 0.5)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=10, framealpha=0.9)

    ax.text(0.98, 0.98, f"Hits: {n_hits}/{len(compounds)}  |  Misses: {n_miss}/{len(compounds)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "fig04_per_compound_loocv")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Multi-Seed Performance Summary
# ═══════════════════════════════════════════════════════════════════
def fig05_multiseed():
    print("Fig 05: Multi-seed summary")
    all_seeds = {s: load_loocv(s) for s in SEEDS}

    hit_rates, auc_rocs, fprs = [], [], []
    for s in SEEDS:
        rows = all_seeds[s]
        n_hits = sum(1 for r in rows if float(r["held_out_prob"]) >= 0.5)
        hr = n_hits / len(rows) * 100
        hit_rates.append(hr)
        auc = np.mean([float(r["val_auroc"]) for r in rows]) * 100
        auc_rocs.append(auc)
        fpr = np.mean([float(r["fold_fpr_default"]) for r in rows]) * 100
        fprs.append(fpr)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Multi-Seed LOOCV Stability (ESM-2, 5 seeds, n = 28 GLI binders)",
                 fontsize=16, fontweight="bold", y=1.02)

    panels = [
        (axes[0], "LOOCV Hit Rate", hit_rates, "%", C["blue"], 60),
        (axes[1], "Per-Fold AUC-ROC", auc_rocs, "%", C["green"], 98),
        (axes[2], "Mean FPR @ P=0.5", fprs, "%", C["vermil"], 0),
    ]

    for ax, title, vals, unit, color, ymin in panels:
        x = np.arange(len(SEEDS))
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white", width=0.6)
        mean_val = np.mean(vals)
        ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.5, alpha=0.6)

        for i, v in enumerate(vals):
            ax.text(i, v + (max(vals) - min(vals)) * 0.05, f"{v:.1f}%",
                    ha="center", fontsize=9.5, fontweight="bold")

        ax.text(0.98, 0.97, f"Mean: {mean_val:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9, rotation=30)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(unit)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if title == "Per-Fold AUC-ROC":
            ax.set_ylim(99.0, 100.05)
        elif ymin > 0:
            ax.set_ylim(ymin, max(vals) * 1.08)

    plt.tight_layout()
    save(fig, "fig05_multiseed_summary")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6 — Ablation Study
# ═══════════════════════════════════════════════════════════════════
def fig06_ablation():
    print("Fig 06: Ablation study")

    # Data from actual ablation run (seed 42)
    conditions = [
        "Full Model",
        "No BindingDB\nPretraining",
        "No Focal Loss\n(BCE instead)",
        "No ZF Domain\nAdaptation",
        "No Morgan\nFingerprints",
        "No SMILES\nAugmentation",
    ]
    hit_rates = [75.0, 67.9, 67.9, 71.4, 71.4, 78.6]
    fprs =      [1.72, 0.38, 0.87, 1.88, 11.50, 9.52]
    hr_deltas = ["", "-7.1 pp", "-7.1 pp", "-3.6 pp", "-3.6 pp", "+3.6 pp"]
    fpr_deltas = ["", "-1.3 pp", "-0.9 pp", "+0.2 pp", "+9.8 pp", "+7.8 pp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Ablation Study: Component Contribution to Model Performance\n(ESM-2, Seed 42, n = 28 GLI binders)",
                 fontsize=15, fontweight="bold", y=1.04)

    y = np.arange(len(conditions))
    colors_hr = [C["blue"]] + [C["orange"]] * 4 + [C["sky"]]
    colors_fpr = [C["green"]] + [C["orange"]] * 4 + [C["vermil"]]

    # Hit rate panel
    bars1 = ax1.barh(y, hit_rates, color=colors_hr, edgecolor="white", height=0.65)
    ax1.axvline(hit_rates[0], color=C["dark"], linestyle="--", linewidth=1, alpha=0.4)
    for i, (v, d) in enumerate(zip(hit_rates, hr_deltas)):
        label = f"{v:.1f}%"
        if d:
            label += f" ({d})"
        ax1.text(v + 0.5, i, label, va="center", fontsize=9.5, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(conditions, fontsize=10)
    ax1.set_xlabel("LOOCV Hit Rate (%)")
    ax1.set_title("Sensitivity (Hit Rate)", fontsize=13)
    ax1.set_xlim(55, 88)
    ax1.axvspan(55, hit_rates[0], alpha=0.06, color=C["vermil"])
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # FPR panel
    bars2 = ax2.barh(y, fprs, color=colors_fpr, edgecolor="white", height=0.65)
    ax2.axvline(fprs[0], color=C["dark"], linestyle="--", linewidth=1, alpha=0.4)
    for i, (v, d) in enumerate(zip(fprs, fpr_deltas)):
        label = f"{v:.2f}%"
        if d:
            label += f" ({d})"
        ax2.text(v + 0.15, i, label, va="center", fontsize=9.5, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("False Positive Rate (%)")
    ax2.set_title("Specificity (FPR)", fontsize=13)
    ax2.set_xlim(0, 16)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    save(fig, "fig06_ablation_study")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7 — 28-Compound x 5-Seed Detection Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig07_heatmap():
    print("Fig 07: Detection heatmap")
    sites = load_inhibitors()
    all_seeds = {s: load_loocv(s) for s in SEEDS}

    # Build matrix
    cpd_data = defaultdict(dict)
    for s in SEEDS:
        for r in all_seeds[s]:
            cpd_data[r["compound"]][s] = float(r["held_out_prob"])

    compounds = sorted(cpd_data.keys(),
                       key=lambda c: np.mean(list(cpd_data[c].values())), reverse=True)
    matrix = np.array([[cpd_data[c].get(s, 0) for s in SEEDS] for c in compounds])
    hits_count = [(matrix[i] >= 0.5).sum() for i in range(len(compounds))]

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(matrix, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")

    for i in range(len(compounds)):
        for j in range(len(SEEDS)):
            val = matrix[i, j]
            hit = "HIT" if val >= 0.5 else "X"
            txtcolor = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}\n{hit}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txtcolor)

    # Hit count column (just right of heatmap)
    for i, hc in enumerate(hits_count):
        ax.text(len(SEEDS) - 0.08, i, f"{hc}/5", va="center", fontsize=9,
                fontweight="bold", color=C["green"] if hc >= 3 else C["miss"])

    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=10)
    ax.set_yticks(range(len(compounds)))
    ax.set_yticklabels(compounds, fontsize=9)
    ax.set_title("Per-Compound LOOCV Detection Across 5 Random Seeds\n(ESM-2, n = 28, HIT = P \u2265 0.5, X = MISS)",
                 fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label("Held-Out P(binder)", fontsize=11)

    plt.tight_layout()
    save(fig, "fig07_detection_heatmap")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8 — Tanimoto Similarity Heatmap (all 28 compounds)
# ═══════════════════════════════════════════════════════════════════
def fig08_tanimoto():
    print("Fig 08: Tanimoto heatmap (28 compounds)")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        print("  SKIPPED (rdkit not available)")
        return

    names, smiles_list = [], []
    with open("gli_inhibitors.csv") as f:
        for r in csv.DictReader(f):
            names.append(r["compound_name"])
            smiles_list.append(r["smiles"])

    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_idx.append(i)

    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

    valid_names = [names[i] for i in valid_idx]

    # Shorten names for display
    short = []
    for nm in valid_names:
        if len(nm) > 15:
            nm = nm[:13] + ".."
        short.append(nm)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim, cmap="YlGnBu", vmin=0, vmax=1)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j:
                val = sim[i, j]
                txtcolor = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=txtcolor)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title("Tanimoto Similarity Between 28 GLI Inhibitors\n(Morgan FP, r=2, 2048-bit)",
                 fontsize=15, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Tanimoto Coefficient", fontsize=11)

    plt.tight_layout()
    save(fig, "fig08_tanimoto_heatmap")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 9 — ML Probability vs Docking Score Scatter
# ═══════════════════════════════════════════════════════════════════
def fig09_scatter():
    print("Fig 09: ML vs docking scatter")
    rows = load_candidates()

    probs, docks, site_colors = [], [], []
    for r in rows:
        try:
            p = float(r["ensemble_prob"])
            d = float(r["best_dock_score"])
        except (ValueError, KeyError):
            continue
        probs.append(p)
        docks.append(d)
        site = r.get("best_site", "")
        site_colors.append(C["zf23"] if "2-3" in site else C["zf45"])

    probs = np.array(probs)
    docks = np.array(docks)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(probs, docks, c=site_colors, s=25, alpha=0.5, edgecolors="white", linewidth=0.3)

    # Tier boundaries
    ax.axhline(-7.0, color=C["dark"], linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(0.85, color=C["dark"], linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(0.80, color=C["dark"], linestyle=":", linewidth=1, alpha=0.3)

    # Shade top-priority region
    ax.fill_between([0.85, probs.max() * 1.01], -10, -7.0,
                    alpha=0.08, color=C["green"], zorder=0)
    ax.text(0.92, -8.5, "Top Priority\nRegion", ha="center", fontsize=10,
            color=C["green"], fontweight="bold", alpha=0.8)

    # Annotations
    best_idx = np.argmin(docks)
    ax.annotate(f"Best: {docks[best_idx]:.2f} kcal/mol",
                xy=(probs[best_idx], docks[best_idx]),
                xytext=(probs[best_idx] - 0.03, docks[best_idx] - 0.3),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["dark"]))

    ax.set_xlabel("Ensemble P(binder)")
    ax.set_ylabel("Best Docking Score (kcal/mol)")
    ax.set_title("ML Ensemble Score vs. Molecular Docking Affinity\n"
                 f"(n = {len(probs)} candidates, AutoDock Vina \u2192 GLI1 ZF sites)",
                 fontsize=14, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["zf23"],
               markersize=8, label="ZF2-3 preferred"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["zf45"],
               markersize=8, label="ZF4-5 preferred"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=10)

    # Stats box
    n_priority = sum(1 for p, d in zip(probs, docks) if p >= 0.85 and d <= -7.0)
    n_novel = sum(1 for r in rows if r.get("novel", "").lower() == "true")
    stats = f"Mean docking: {np.mean(docks):.2f} kcal/mol\nPriority (P\u22650.85 & \u2264-7.0): {n_priority}\nNovel scaffolds: {n_novel}/{len(probs)} ({n_novel/len(probs)*100:.1f}%)"
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                                  edgecolor="#cccccc"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "fig09_ml_vs_docking")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10 — MD Time Series Composite
# ═══════════════════════════════════════════════════════════════════
def fig10_md():
    print("Fig 10: MD composite")

    # Load summary stats for ternary MD
    md_stats = {}
    with open("ternary_md/analysis/summary_stats.csv") as f:
        for r in csv.DictReader(f):
            md_stats[r["system"]] = r

    systems = ["control", "CNP0544084", "CNP0592286", "CNP0214725"]
    labels = ["GLI1-DNA\n(control)", "CNP0544084\n(ZF2-3)", "CNP0592286\n(ZF4-5)", "CNP0214725\n(ZF2-3, PAINS)"]
    sys_colors = [C["grey"], C["blue"], C["orange"], C["vermil"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Ternary Molecular Dynamics Validation (~110 ns)\nGLI1 Zinc Finger + DNA + Ligand",
                 fontsize=15, fontweight="bold", y=1.02)

    # Panel A: Backbone RMSD
    ax = axes[0, 0]
    vals = [float(md_stats[s]["bb_rmsd_mean_nm"]) * 10 for s in systems]
    bars = ax.bar(range(4), vals, color=sys_colors, edgecolor="white", width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f} \u00C5", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Backbone RMSD (\u00C5)")
    ax.set_title("A) Protein Stability", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Ligand RMSD (only 3 compounds, not control)
    ax = axes[0, 1]
    lig_systems = ["CNP0544084", "CNP0592286", "CNP0214725"]
    lig_labels = ["CNP0544084\n(ZF2-3)", "CNP0592286\n(ZF4-5)", "CNP0214725\n(PAINS)"]
    lig_colors = [C["blue"], C["orange"], C["vermil"]]
    vals = [float(md_stats[s]["lig_rmsd_mean_nm"]) * 10 for s in lig_systems]
    bars = ax.bar(range(3), vals, color=lig_colors, edgecolor="white", width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.1f} \u00C5", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(lig_labels, fontsize=8)
    ax.set_ylabel("Ligand RMSD (\u00C5)")
    ax.set_title("B) Ligand Positional Stability", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Radius of Gyration
    ax = axes[1, 0]
    vals = [float(md_stats[s]["rg_mean_nm"]) * 10 for s in systems]
    bars = ax.bar(range(4), vals, color=sys_colors, edgecolor="white", width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f} \u00C5", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Radius of Gyration (\u00C5)")
    ax.set_title("C) Domain Compactness", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel D: Ligand Min Distance to Protein (binding retention)
    ax = axes[1, 1]
    vals = [float(md_stats[s]["lig_mindist_mean_nm"]) * 10 for s in lig_systems]
    bars = ax.bar(range(3), vals, color=lig_colors, edgecolor="white", width=0.55)
    ax.axhline(3.5, color=C["dark"], linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.4, 4.2, "Contact threshold (3.5 \u00C5)", fontsize=9, color=C["dark"])
    max_v = max(vals)
    for i, v in enumerate(vals):
        bl = "BOUND" if v < 5.0 else "DISSOCIATED"
        lbl_color = C["green"] if bl == "BOUND" else C["vermil"]
        # Place label inside bar for tall bars, above for short bars
        if v > 10:
            ax.text(i, v * 0.55, f"{bl}\n{v:.1f} \u00C5", ha="center", fontsize=10,
                    fontweight="bold", color="white")
        else:
            ax.text(i, v + 0.3, f"{v:.1f} \u00C5  {bl}", ha="center", fontsize=8.5,
                    fontweight="bold", color=lbl_color)
    ax.set_xticks(range(3))
    ax.set_xticklabels(lig_labels, fontsize=8)
    ax.set_ylabel("Min Ligand-Protein Distance (\u00C5)")
    ax.set_title("D) Binding Retention", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save(fig, "fig10_md_validation")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating 10 GSDSEF notebook figures -> {OUT}/")
    print("=" * 60)
    fig01_architecture()
    fig02_pipeline()
    fig03_funnel()
    fig04_loocv()
    fig05_multiseed()
    fig06_ablation()
    fig07_heatmap()
    fig08_tanimoto()
    fig09_scatter()
    fig10_md()
    print("=" * 60)
    print(f"Done! All figures saved to {OUT}/")
    print("Figures are ready for upload to the GSDSEF notebook document.")
