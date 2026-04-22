#!/usr/bin/env python3
"""
GLI-PLAPT Results Breakdown — Per-compound, per-encoder, per-seed visualizations.

Generates 4 publication-quality figures from saved results:
  1. Per-compound LOOCV probabilities (hit vs miss, both encoders, seed 42)
  2. Multi-seed hit consistency heatmap (5 seeds × 2 encoders × 11 compounds)
  3. Tanimoto structural similarity heatmap
  4. Cross-encoder scatter (ProtBERT vs ESM-2, all seeds)

Data source: outputs/phase1_results.json, outputs/results_summary.json
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open("outputs/phase1_results.json") as f:
    phase1 = json.load(f)

with open("outputs/results_summary.json") as f:
    summary = json.load(f)

COMPOUNDS = phase1["tanimoto_analysis"]["compound_names"]
SEEDS = [str(s) for s in phase1["seeds"]]
THRESHOLD = 0.5

# Short display names
SHORT = {
    "GANT61": "GANT61",
    "GANT61-D": "GANT61-D",
    "Compound_1": "Cpd_1",
    "GlaB": "GlaB",
    "JC19": "JC19",
    "BAS07019774": "BAS079",
    "Compound_39": "Cpd_39",
    "Compound_48": "Cpd_48",
    "Compound_49": "Cpd_49",
    "Compound_50": "Cpd_50",
    "Compound_52": "Cpd_52",
}

# Scaffold groups for coloring
SCAFFOLD_GROUP = {}
for c in ["GANT61", "GANT61-D"]:
    SCAFFOLD_GROUP[c] = "GANT61 family"
for c in ["Compound_39", "Compound_48", "Compound_49", "Compound_50", "Compound_52"]:
    SCAFFOLD_GROUP[c] = "Wen2023 series"
for c in ["GlaB"]:
    SCAFFOLD_GROUP[c] = "GlaB (isoflavone)"
for c in ["JC19"]:
    SCAFFOLD_GROUP[c] = "JC19"
for c in ["BAS07019774"]:
    SCAFFOLD_GROUP[c] = "BAS079"
for c in ["Compound_1"]:
    SCAFFOLD_GROUP[c] = "Compound_1"

GROUP_COLORS = {
    "GANT61 family": "#E07B39",
    "Wen2023 series": "#4C9A2A",
    "GlaB (isoflavone)": "#8B4513",
    "JC19": "#6A5ACD",
    "BAS079": "#DC143C",
    "Compound_1": "#2E86C1",
}


def get_seed42_data():
    """Extract per-compound probs from seed 42 consensus data."""
    seed_data = phase1["consensus"]["per_seed"]["42"]["per_compound"]
    esm2_probs, prot_probs, ensemble_probs = [], [], []
    for c in COMPOUNDS:
        d = seed_data[c]
        esm2_probs.append(d["esm2_prob"])
        prot_probs.append(d["protbert_prob"])
        ensemble_probs.append(d["ensemble_prob"])
    return np.array(esm2_probs), np.array(prot_probs), np.array(ensemble_probs)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-compound LOOCV probabilities — seed 42, both encoders
# ══════════════════════════════════════════════════════════════════════════════
def fig1_per_compound_probs():
    esm2, prot, ens = get_seed42_data()
    names = [SHORT[c] for c in COMPOUNDS]

    # Sort by ensemble probability descending
    order = np.argsort(ens)  # ascending → bottom to top in barh
    names_sorted = [names[i] for i in order]
    esm2_sorted = esm2[order]
    prot_sorted = prot[order]
    ens_sorted = ens[order]
    compounds_sorted = [COMPOUNDS[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names_sorted))
    bar_h = 0.28

    # Bars
    bars_prot = ax.barh(y + bar_h, prot_sorted, bar_h, label="ProtBERT",
                        color="#3498DB", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_esm2 = ax.barh(y, esm2_sorted, bar_h, label="ESM-2",
                        color="#E74C3C", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_ens = ax.barh(y - bar_h, ens_sorted, bar_h, label="Ensemble",
                       color="#2ECC71", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Threshold line
    ax.axvline(x=THRESHOLD, color="#333333", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax.text(THRESHOLD + 0.01, len(names_sorted) - 0.3, "P = 0.5\nthreshold",
            fontsize=8, color="#333333", va="top")

    # Hit/Miss labels on right side
    for i, c in enumerate(compounds_sorted):
        hit = ens_sorted[i] >= THRESHOLD
        label = "HIT" if hit else "MISS"
        color = "#27AE60" if hit else "#C0392B"
        ax.text(1.02, y[i], label, fontsize=8, fontweight="bold",
                color=color, va="center", transform=ax.get_yaxis_transform())

    # Scaffold group color strips on left
    for i, c in enumerate(compounds_sorted):
        grp = SCAFFOLD_GROUP[c]
        ax.barh(y[i], 0.015, 0.85, left=-0.025, color=GROUP_COLORS[grp],
                clip_on=False, zorder=10)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xlabel("LOOCV Held-Out Probability P(binder)", fontsize=11)
    ax.set_title("Per-Compound LOOCV Predictions — Seed 42", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.03, 1.05)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Shade miss region
    ax.axvspan(0, THRESHOLD, alpha=0.04, color="red", zorder=0)
    ax.axvspan(THRESHOLD, 1.05, alpha=0.04, color="green", zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_per_compound_probs.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig1_per_compound_probs.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 1: Per-compound LOOCV probabilities")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Multi-seed hit consistency heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig2_multiseed_heatmap():
    # Build matrix: rows = compounds, cols = seeds, cells = ensemble probability
    # Two panels: ESM-2 and ProtBERT
    esm2_data = phase1["esm2_multi_seed"]["per_compound"]
    prot_data = phase1["protbert_multi_seed"]["per_compound"]

    # Get per-seed probabilities from consensus data
    esm2_matrix = np.zeros((len(COMPOUNDS), len(SEEDS)))
    prot_matrix = np.zeros((len(COMPOUNDS), len(SEEDS)))

    for j, seed in enumerate(SEEDS):
        seed_consensus = phase1["consensus"]["per_seed"][seed]["per_compound"]
        for i, c in enumerate(COMPOUNDS):
            esm2_matrix[i, j] = seed_consensus[c]["esm2_prob"]
            prot_matrix[i, j] = seed_consensus[c]["protbert_prob"]

    # Sort compounds by mean ensemble probability
    mean_probs = (esm2_matrix.mean(axis=1) + prot_matrix.mean(axis=1)) / 2
    order = np.argsort(mean_probs)[::-1]

    names = [SHORT[COMPOUNDS[i]] for i in order]
    esm2_sorted = esm2_matrix[order]
    prot_sorted = prot_matrix[order]

    # Custom colormap: red (miss) → yellow (borderline) → green (hit)
    cmap = LinearSegmentedColormap.from_list("hit_miss",
        ["#C0392B", "#E74C3C", "#F5B041", "#F9E79F", "#82E0AA", "#27AE60", "#1E8449"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, matrix, title in [(ax1, esm2_sorted, "ESM-2"), (ax2, prot_sorted, "ProtBERT")]:
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Annotate cells with probability and hit/miss symbol
        for i in range(len(names)):
            for j in range(len(SEEDS)):
                val = matrix[i, j]
                hit = val >= THRESHOLD
                symbol = "●" if hit else "✗"
                text_color = "white" if val > 0.7 or val < 0.25 else "black"
                ax.text(j, i, f"{val:.2f}\n{symbol}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold" if hit else "normal")

        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=9, rotation=30, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Hit rate per seed at bottom
        for j in range(len(SEEDS)):
            hits = (matrix[:, j] >= THRESHOLD).sum()
            ax.text(j, len(names) + 0.1, f"{hits}/{len(names)}",
                    ha="center", va="top", fontsize=8, color="#555")

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)

    fig.suptitle("LOOCV Hit Consistency Across Seeds & Encoders", fontsize=14, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=[ax1, ax2], label="P(binder)", shrink=0.8, pad=0.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_multiseed_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig2_multiseed_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 2: Multi-seed hit consistency heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Tanimoto similarity heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig3_tanimoto_heatmap():
    sim_matrix = np.array(phase1["tanimoto_analysis"]["sim_matrix"])
    names = [SHORT[c] for c in COMPOUNDS]

    # Cluster order: Wen2023 together, GANT61 together, then singletons
    cluster_order = [6, 7, 8, 9, 10,  # Wen2023 (Cpd_39..52)
                     0, 1,              # GANT61 family
                     2,                 # Compound_1
                     4,                 # JC19
                     5,                 # BAS079
                     3]                 # GlaB
    names_ordered = [names[i] for i in cluster_order]
    sim_ordered = sim_matrix[np.ix_(cluster_order, cluster_order)]

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = LinearSegmentedColormap.from_list("tc",
        ["#FFFFFF", "#AED6F1", "#3498DB", "#1A5276", "#0B2545"])
    im = ax.imshow(sim_ordered, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Annotate
    for i in range(len(names_ordered)):
        for j in range(len(names_ordered)):
            val = sim_ordered[i, j]
            if i == j:
                continue
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_xticks(range(len(names_ordered)))
    ax.set_xticklabels(names_ordered, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(names_ordered)))
    ax.set_yticklabels(names_ordered, fontsize=9)

    # Draw cluster boxes
    # Wen2023: indices 0-4
    rect1 = plt.Rectangle((-0.5, -0.5), 5, 5, fill=False, edgecolor="#27AE60", linewidth=2.5, linestyle="-")
    ax.add_patch(rect1)
    ax.text(2, 5.2, "Wen2023 cluster\nTc = 0.72 ± 0.05", ha="center", va="top",
            fontsize=8, color="#27AE60", fontweight="bold")

    # GANT61: indices 5-6
    rect2 = plt.Rectangle((4.5, 4.5), 2, 2, fill=False, edgecolor="#E07B39", linewidth=2.5, linestyle="-")
    ax.add_patch(rect2)

    ax.set_title("Tanimoto Similarity Between GLI Inhibitors\n(Morgan FP, r=2, 2048-bit)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Tanimoto Coefficient", shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_tanimoto_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig3_tanimoto_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 3: Tanimoto similarity heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cross-encoder scatter — ESM-2 vs ProtBERT across all seeds
# ══════════════════════════════════════════════════════════════════════════════
def fig4_cross_encoder_scatter():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Collect all (esm2, protbert) pairs across seeds
    for seed in SEEDS:
        seed_data = phase1["consensus"]["per_seed"][seed]["per_compound"]
        for c in COMPOUNDS:
            d = seed_data[c]
            e2 = d["esm2_prob"]
            pb = d["protbert_prob"]
            grp = SCAFFOLD_GROUP[c]
            color = GROUP_COLORS[grp]
            marker = "o"
            size = 60
            alpha = 0.7
            ax.scatter(pb, e2, c=color, s=size, alpha=alpha, edgecolors="white",
                       linewidths=0.5, zorder=5)

    # Threshold quadrant lines
    ax.axhline(y=THRESHOLD, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=THRESHOLD, color="#888", linestyle="--", linewidth=1, alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.95, "Both HIT", transform=ax.transAxes, fontsize=10,
            color="#27AE60", fontweight="bold", ha="center", va="top", alpha=0.6)
    ax.text(0.25, 0.05, "Both MISS", transform=ax.transAxes, fontsize=10,
            color="#C0392B", fontweight="bold", ha="center", va="bottom", alpha=0.6)
    ax.text(0.25, 0.95, "ProtBERT MISS\nESM-2 HIT", transform=ax.transAxes, fontsize=8,
            color="#E67E22", fontweight="bold", ha="center", va="top", alpha=0.5)
    ax.text(0.75, 0.05, "ProtBERT HIT\nESM-2 MISS", transform=ax.transAxes, fontsize=8,
            color="#E67E22", fontweight="bold", ha="center", va="bottom", alpha=0.5)

    # Diagonal (perfect agreement)
    ax.plot([0, 1], [0, 1], "k-", alpha=0.15, linewidth=1, zorder=0)

    # Shade quadrants
    ax.axvspan(THRESHOLD, 1.05, ymin=0.5/1.1, ymax=1.0, alpha=0.03, color="green", zorder=0)
    ax.axvspan(0, THRESHOLD, ymin=0, ymax=0.5/1.1, alpha=0.03, color="red", zorder=0)

    # Legend for scaffold groups
    legend_handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_COLORS]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9, title="Scaffold Group")

    # Annotate mean positions for each compound
    for c in COMPOUNDS:
        e2_vals = [phase1["consensus"]["per_seed"][s]["per_compound"][c]["esm2_prob"] for s in SEEDS]
        pb_vals = [phase1["consensus"]["per_seed"][s]["per_compound"][c]["protbert_prob"] for s in SEEDS]
        me2, mpb = np.mean(e2_vals), np.mean(pb_vals)
        ax.annotate(SHORT[c], (mpb, me2), fontsize=7, color="#333",
                    textcoords="offset points", xytext=(6, 6),
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))

    ax.set_xlabel("ProtBERT P(binder)", fontsize=11)
    ax.set_ylabel("ESM-2 P(binder)", fontsize=11)
    ax.set_title("Cross-Encoder Agreement — All Seeds\n(each dot = 1 compound × 1 seed)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_cross_encoder_scatter.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig4_cross_encoder_scatter.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Figure 4: Cross-encoder scatter plot")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating figures → {OUT_DIR}/\n")
    fig1_per_compound_probs()
    fig2_multiseed_heatmap()
    fig3_tanimoto_heatmap()
    fig4_cross_encoder_scatter()
    print(f"\nDone! All figures saved to {OUT_DIR}/")
