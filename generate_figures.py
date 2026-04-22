#!/usr/bin/env python3
"""Generate all GSDSEF presentation figures for GLI-MultiNet.

Produces 6 publication-quality figures:
  1. Per-compound LOOCV held-out probability (Slide 8)
  2. Multi-seed LOOCV performance summary (Slide 8)
  3. Ablation study component contribution (Slide 9)
  4. Per-compound detection consistency heatmap (Slide 9)
  5. ML ensemble score vs. docking score scatter (Slide 10)
  6. Screening pipeline funnel (Slide 10)

All figures: 300 DPI, white background, black text, Okabe-Ito colorblind-safe palette.
"""

import csv
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
OKABE = {
    "blue":    "#0173B2",
    "orange":  "#DE8F05",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#ECE133",
    "vermil":  "#D55E00",
    "purple":  "#CC78BC",
    "grey":    "#999999",
}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          14,
    "axes.titlesize":     18,
    "axes.titleweight":   "bold",
    "axes.labelsize":     16,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#333333",
    "axes.grid":          False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_loocv(seed: int, encoder: str = "esm2"):
    """Load one LOOCV fold CSV, excluding GANT61 prodrug."""
    path = f"outputs/logs/{encoder}_seed{seed}_stage3_loocv_folds.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["compound"] == "GANT61":
                continue
            rows.append(row)
    return rows


def load_all_seeds():
    """Load all ESM-2 LOOCV seeds into {seed: [rows]}."""
    return {s: load_loocv(s) for s in SEEDS}


def load_candidates():
    """Load final_candidates_v2.csv for docking scatter."""
    path = "outputs/final_candidates_v2.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure 1: Per-Compound LOOCV Bar Chart (Slide 8)
# ---------------------------------------------------------------------------
def fig1_per_compound_loocv():
    rows = load_loocv(42)
    rows.sort(key=lambda r: float(r["held_out_prob"]))

    compounds = [r["compound"] for r in rows]
    probs = [float(r["held_out_prob"]) for r in rows]
    uncs = [float(r["held_out_uncertainty"]) for r in rows]
    hits = [int(r["held_out_correct"]) for r in rows]

    colors = [OKABE["blue"] if h else OKABE["orange"] for h in hits]

    fig, ax = plt.subplots(figsize=(10, 9))

    y_pos = np.arange(len(compounds))
    bars = ax.barh(y_pos, probs, xerr=uncs, height=0.72,
                   color=colors, edgecolor="none",
                   error_kw=dict(ecolor="#555555", capsize=3, linewidth=1.0))

    ax.axvline(x=0.5, color="#333333", linestyle="--", linewidth=1.2,
               label="Classification threshold (P = 0.5)")

    ax.set_yticks(y_pos)
    # Clean up compound names for display
    display_names = []
    for c in compounds:
        c = c.replace("_Barardozi", " (Bar.)").replace("_Lospinoso", " (Los.)")
        c = c.replace("_Manetti", " (Man.)").replace("_Quinoline", " (Quin.)")
        c = c.replace("Compound_", "Cmpd ")
        display_names.append(c)
    ax.set_yticklabels(display_names, fontsize=11)

    ax.set_xlabel("Held-Out P(binder)")
    ax.set_xlim(0, 1.08)
    ax.set_title("Per-Compound LOOCV Held-Out Binding Probability\n(ESM-2, Seed 42, n = 28 GLI Binders)")

    # Value labels on bars
    for i, (p, h) in enumerate(zip(probs, hits)):
        label = f"{p:.3f}"
        ax.text(p + uncs[i] + 0.02, i, label, va="center", fontsize=9,
                color="#333333")

    n_hits = sum(hits)
    hit_patch = mpatches.Patch(color=OKABE["blue"], label=f"HIT (P ≥ 0.5): {n_hits}")
    miss_patch = mpatches.Patch(color=OKABE["orange"], label=f"MISS (P < 0.5): {len(hits) - n_hits}")
    thresh_line = plt.Line2D([0], [0], color="#333333", linestyle="--",
                             linewidth=1.2, label="Threshold (P = 0.5)")
    ax.legend(handles=[hit_patch, miss_patch, thresh_line],
              loc="lower right", fontsize=11, framealpha=0.95)

    # Annotation box
    ax.text(0.97, 0.03,
            f"Hit Rate: {n_hits}/28 = {n_hits/28*100:.1f}%\n"
            f"Mean P(binder): {np.mean(probs):.3f}\n"
            f"MC Dropout: T = 50 passes",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                      edgecolor="#999999", alpha=0.9))

    ax.invert_yaxis()
    plt.tight_layout()
    out = OUT_DIR / "fig1_per_compound_loocv.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Multi-Seed LOOCV Performance (Slide 8)
# ---------------------------------------------------------------------------
def fig2_multiseed_performance():
    all_data = load_all_seeds()

    seed_labels = [str(s) for s in SEEDS]
    hit_rates = []
    aurocs = []
    fprs = []

    for s in SEEDS:
        rows = all_data[s]
        hr = sum(int(r["held_out_correct"]) for r in rows) / len(rows)
        hit_rates.append(hr * 100)
        aurocs.append(np.mean([float(r["val_auroc"]) for r in rows]) * 100)
        fprs.append(np.mean([float(r["fold_fpr_default"]) for r in rows]) * 100)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Hit Rate
    ax = axes[0]
    bars = ax.bar(seed_labels, hit_rates, color=OKABE["blue"],
                  edgecolor="white", width=0.6)
    mean_hr = np.mean(hit_rates)
    ax.axhline(y=mean_hr, color=OKABE["vermil"], linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_hr:.1f}%")
    ax.fill_between([-0.5, len(SEEDS) - 0.5],
                     mean_hr - np.std(hit_rates), mean_hr + np.std(hit_rates),
                     color=OKABE["vermil"], alpha=0.10)
    for bar, val in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_xlabel("Random Seed")
    ax.set_title("LOOCV Hit Rate")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper left")

    # Panel B: AUROC
    ax = axes[1]
    bars = ax.bar(seed_labels, aurocs, color=OKABE["green"],
                  edgecolor="white", width=0.6)
    mean_auc = np.mean(aurocs)
    ax.axhline(y=mean_auc, color=OKABE["vermil"], linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_auc:.2f}%")
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#333333")
    ax.set_ylabel("Mean Per-Fold AUC-ROC (%)")
    ax.set_xlabel("Random Seed")
    ax.set_title("Per-Fold AUC-ROC")
    ax.set_ylim(98.0, 100.3)
    ax.legend(fontsize=10, loc="lower left")

    # Panel C: FPR
    ax = axes[2]
    bars = ax.bar(seed_labels, fprs, color=OKABE["purple"],
                  edgecolor="white", width=0.6)
    mean_fpr = np.mean(fprs)
    ax.axhline(y=mean_fpr, color=OKABE["vermil"], linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_fpr:.2f}%")
    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_xlabel("Random Seed")
    ax.set_title("Mean FPR @ P = 0.5")
    ax.set_ylim(0, max(fprs) * 1.5)
    ax.legend(fontsize=10, loc="upper right")

    fig.suptitle("Multi-Seed LOOCV Stability (ESM-2, n = 28 GLI Binders, 5 Seeds)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig2_multiseed_performance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Ablation Study (Slide 9)
# ---------------------------------------------------------------------------
def fig3_ablation_study():
    # Ablation data (from Rosenbluth, seed 42, 28 compounds excl. GANT61)
    conditions = [
        ("Full Model", 75.0, 1.72),
        ("No BindingDB\nPretraining", 67.9, 0.38),
        ("No Focal Loss\n(BCE instead)", 67.9, 0.87),
        ("No ZF Domain\nAdaptation", 71.4, 1.88),
        ("No Morgan\nFingerprints", 71.4, 11.5),
        ("No SMILES\nAugmentation", 78.6, 9.52),
    ]
    labels = [c[0] for c in conditions]
    hit_rates = [c[1] for c in conditions]
    fprs = [c[2] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---- Panel A: Hit Rate ----
    y_pos = np.arange(len(labels))
    colors_hr = []
    for i, hr in enumerate(hit_rates):
        if i == 0:
            colors_hr.append(OKABE["blue"])  # baseline
        elif hr > hit_rates[0]:
            colors_hr.append(OKABE["sky"])  # improved (but overfitting)
        elif hr < hit_rates[0]:
            colors_hr.append(OKABE["orange"])  # degraded
        else:
            colors_hr.append(OKABE["grey"])

    bars1 = ax1.barh(y_pos, hit_rates, height=0.6, color=colors_hr,
                     edgecolor="none")
    ax1.axvline(x=75.0, color="#333333", linestyle="--", linewidth=1.0,
                alpha=0.6, label="Full model baseline")

    for i, (hr, label) in enumerate(zip(hit_rates, labels)):
        delta = hr - 75.0
        delta_str = f"  {hr:.1f}%" if i == 0 else f"  {hr:.1f}% ({delta:+.1f}%)"
        ax1.text(hr + 0.5, i, delta_str, va="center", fontsize=11,
                 fontweight="bold" if i == 0 else "normal")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.set_xlabel("LOOCV Hit Rate (%)")
    ax1.set_xlim(0, 95)
    ax1.set_title("Hit Rate by Condition")
    ax1.invert_yaxis()
    ax1.legend(fontsize=10, loc="lower right")

    # ---- Panel B: FPR ----
    colors_fpr = []
    for i, fpr in enumerate(fprs):
        if i == 0:
            colors_fpr.append(OKABE["blue"])
        elif fpr > 3.0:
            colors_fpr.append(OKABE["vermil"])  # bad FPR
        else:
            colors_fpr.append(OKABE["green"])  # good FPR

    bars2 = ax2.barh(y_pos, fprs, height=0.6, color=colors_fpr,
                     edgecolor="none")
    ax2.axvline(x=1.72, color="#333333", linestyle="--", linewidth=1.0,
                alpha=0.6, label="Full model baseline")

    for i, (fpr, label) in enumerate(zip(fprs, labels)):
        ax2.text(fpr + 0.3, i, f"  {fpr:.2f}%", va="center", fontsize=11,
                 fontweight="bold" if i == 0 else "normal")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in labels])
    ax2.set_xlabel("False Positive Rate @ P = 0.5 (%)")
    ax2.set_xlim(0, 16)
    ax2.set_title("Specificity (FPR)")
    ax2.invert_yaxis()
    ax2.legend(fontsize=10, loc="lower right")

    fig.suptitle("Ablation Study: Component Contribution to Model Performance\n"
                 "(ESM-2, Seed 42, n = 28 GLI Binders)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig3_ablation_study.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Per-Compound Detection Heatmap (Slide 9)
# ---------------------------------------------------------------------------
def fig4_compound_heatmap():
    all_data = load_all_seeds()

    # Get compound order (sorted by mean prob across seeds, descending)
    compound_names = [r["compound"] for r in all_data[42]]
    mean_probs = {}
    for name in compound_names:
        probs = []
        for s in SEEDS:
            for r in all_data[s]:
                if r["compound"] == name:
                    probs.append(float(r["held_out_prob"]))
        mean_probs[name] = np.mean(probs)

    compound_names.sort(key=lambda c: mean_probs[c], reverse=True)

    # Build matrix: rows = compounds, cols = seeds
    matrix = np.zeros((len(compound_names), len(SEEDS)))
    for j, s in enumerate(SEEDS):
        for r in all_data[s]:
            if r["compound"] in compound_names:
                i = compound_names.index(r["compound"])
                matrix[i, j] = float(r["held_out_prob"])

    fig, ax = plt.subplots(figsize=(8, 10))

    # Custom colormap: orange for low → white at 0.5 → blue for high
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list("hit_miss", [
        (0.0,  OKABE["orange"]),
        (0.35, "#FADDB5"),
        (0.5,  "#F5F5F5"),
        (0.65, "#B3D7F0"),
        (1.0,  OKABE["blue"]),
    ])

    im = ax.imshow(matrix, cmap=cmap_custom, vmin=0, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(len(compound_names)):
        for j in range(len(SEEDS)):
            val = matrix[i, j]
            text_color = "white" if val > 0.8 or val < 0.2 else "#333333"
            marker = "HIT" if val >= 0.5 else "X"
            ax.text(j, i, f"{val:.2f}\n{marker}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Display names
    display_names = []
    for c in compound_names:
        c = c.replace("_Barardozi", " (Bar.)").replace("_Lospinoso", " (Los.)")
        c = c.replace("_Manetti", " (Man.)").replace("_Quinoline", " (Quin.)")
        c = c.replace("Compound_", "Cmpd ")
        display_names.append(c)

    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=12)
    ax.set_yticks(range(len(compound_names)))
    ax.set_yticklabels(display_names, fontsize=10)

    ax.set_title("Per-Compound LOOCV Detection Across 5 Random Seeds\n"
                 "(ESM-2, n = 28 GLI Binders, HIT = P >= 0.5, X = MISS)",
                 fontsize=15, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Held-Out P(binder)", fontsize=13)
    cbar.ax.axhline(y=0.5, color="#333333", linewidth=1.5, linestyle="--")

    # Summary row counts at right
    for i, name in enumerate(compound_names):
        n_detected = sum(1 for j in range(len(SEEDS)) if matrix[i, j] >= 0.5)
        ax.text(len(SEEDS) + 0.1, i, f"{n_detected}/5",
                ha="left", va="center", fontsize=10,
                fontweight="bold" if n_detected >= 3 else "normal",
                color=OKABE["blue"] if n_detected >= 3 else OKABE["orange"])

    plt.tight_layout()
    out = OUT_DIR / "fig4_compound_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: ML Score vs. Docking Score (Slide 10)
# ---------------------------------------------------------------------------
def fig5_ml_vs_docking():
    candidates = load_candidates()

    probs = []
    docks = []
    sites = []
    novel_flags = []

    for r in candidates:
        try:
            p = float(r["ensemble_prob"])
            d = float(r["best_dock_score"])
            site = r.get("best_site", "ZF4-5")
            novel = r.get("novel", "").lower() == "true"
        except (ValueError, KeyError):
            continue
        probs.append(p)
        docks.append(d)
        sites.append(site)
        novel_flags.append(novel)

    probs = np.array(probs)
    docks = np.array(docks)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Separate by site
    zf23_mask = np.array([s == "ZF2-3" for s in sites])
    zf45_mask = ~zf23_mask

    ax.scatter(probs[zf45_mask], docks[zf45_mask],
               c=OKABE["blue"], alpha=0.45, s=35,
               label=f"ZF4-5 preferred (n={zf45_mask.sum()})",
               marker="o", edgecolors="none")
    ax.scatter(probs[zf23_mask], docks[zf23_mask],
               c=OKABE["orange"], alpha=0.55, s=45,
               label=f"ZF2-3 preferred (n={zf23_mask.sum()})",
               marker="^", edgecolors="none")

    # Priority quadrant shading (top-left = high ML + strong docking)
    ax.axhspan(ymin=min(docks) - 0.5, ymax=-7.0,
               xmin=0, xmax=1, alpha=0.04, color=OKABE["green"])
    ax.axvline(x=0.85, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.axhline(y=-7.0, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.text(0.92, min(docks) + 0.15,
            "Priority\ncandidates",
            fontsize=11, fontweight="bold", color=OKABE["green"],
            ha="center", va="bottom", alpha=0.8)

    # Annotate best candidate
    best_idx = np.argmin(docks)
    ax.annotate(f"Best: {docks[best_idx]:.2f} kcal/mol",
                xy=(probs[best_idx], docks[best_idx]),
                xytext=(probs[best_idx] - 0.08, docks[best_idx] - 0.25),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333333",
                                lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                          edgecolor="#999999"))

    ax.set_xlabel("Ensemble P(binder)")
    ax.set_ylabel("Best Docking Score (kcal/mol)")
    ax.set_title("Multi-Modal Candidate Prioritization:\n"
                 "ML Ensemble Score vs. Molecular Docking Affinity\n"
                 f"(n = {len(probs)} candidates, AutoDock Vina → GLI1 ZF sites)")

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # Stats annotation
    n_priority = sum(1 for p, d in zip(probs, docks) if p >= 0.85 and d <= -7.0)
    ax.text(0.97, 0.03,
            f"Mean docking: {np.mean(docks):.2f} kcal/mol\n"
            f"Priority (P ≥ 0.85 & ≤ −7.0): {n_priority}\n"
            f"Novel scaffolds: {sum(novel_flags)}/{len(novel_flags)} "
            f"({sum(novel_flags)/len(novel_flags)*100:.1f}%)",
            transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                      edgecolor="#999999", alpha=0.9))

    ax.invert_yaxis()
    plt.tight_layout()
    out = OUT_DIR / "fig5_ml_vs_docking.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 6: Screening Pipeline Funnel (Slide 10)
# ---------------------------------------------------------------------------
def fig6_screening_funnel():
    stages = [
        ("Raw Library\n(Enamine + COCONUT)", 1176000, ""),
        ("Prescreened\n(MW, PAINS, Lipinski)", 1070832, "91.1% pass"),
        ("Very High Confidence\n(P > 0.85)", 9262, "0.86% of screened"),
        ("Diverse Hits\n(Butina clustering)", 500, "84.6% novel"),
        ("Docked Candidates\n(AutoDock Vina)", 499, "2 ZF sites"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    n_stages = len(stages)
    max_val = stages[0][1]

    # Use log-scaled widths for visual clarity (raw numbers span 3 orders of magnitude)
    log_widths = [np.log10(s[1]) / np.log10(max_val) for s in stages]

    bar_height = 0.7
    y_positions = list(range(n_stages))[::-1]  # top to bottom

    colors = [OKABE["blue"], OKABE["sky"], OKABE["green"],
              OKABE["purple"], OKABE["orange"]]

    for i, (label, count, note) in enumerate(stages):
        y = y_positions[i]
        w = log_widths[i]
        # Center the bar
        left = (1 - w) / 2
        ax.barh(y, w, left=left, height=bar_height,
                color=colors[i], edgecolor="white", linewidth=2)

        # Count label (inside bar)
        count_str = f"{count:,}"
        ax.text(0.5, y + 0.02, count_str, ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if w > 0.5 else "#333333")

        # Stage label (left side)
        ax.text(left - 0.02, y, label, ha="right", va="center",
                fontsize=11, color="#333333")

        # Note (right side)
        if note:
            ax.text(left + w + 0.02, y, note, ha="left", va="center",
                    fontsize=10, color="#666666", style="italic")

        # Connecting arrow between stages
        if i < n_stages - 1:
            next_y = y_positions[i + 1]
            ax.annotate("", xy=(0.5, next_y + bar_height / 2 + 0.05),
                        xytext=(0.5, y - bar_height / 2 - 0.05),
                        arrowprops=dict(arrowstyle="->", color="#AAAAAA",
                                        lw=1.5))

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.8, n_stages - 0.2)
    ax.axis("off")

    ax.set_title("Virtual Screening Pipeline:\nFrom 1.18M Compounds to 499 Docked Candidates",
                 fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    out = OUT_DIR / "fig6_screening_funnel.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating GSDSEF presentation figures...\n")

    print("[1/6] Per-compound LOOCV bar chart...")
    fig1_per_compound_loocv()

    print("[2/6] Multi-seed LOOCV performance...")
    fig2_multiseed_performance()

    print("[3/6] Ablation study comparison...")
    fig3_ablation_study()

    print("[4/6] Per-compound detection heatmap...")
    fig4_compound_heatmap()

    print("[5/6] ML score vs. docking score scatter...")
    fig5_ml_vs_docking()

    print("[6/6] Screening pipeline funnel...")
    fig6_screening_funnel()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
