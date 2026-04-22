#!/usr/bin/env python3
"""Generate slide-optimized composite figures for GSDSEF Slides 8–10.

Each figure is sized for a 16:9 slide (13.33" × 7.5") where the figure
occupies ~65% of the slide width alongside a narrow text column.

Output: 300 DPI PNGs in outputs/figures/
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------
OK = {
    "blue":   "#0173B2",
    "orange": "#DE8F05",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#ECE133",
    "vermil": "#D55E00",
    "purple": "#CC78BC",
    "grey":   "#999999",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         12,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "axes.labelweight":  "bold",
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.grid":         False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_loocv(seed):
    rows = []
    with open(f"outputs/logs/esm2_seed{seed}_stage3_loocv_folds.csv") as f:
        for r in csv.DictReader(f):
            if r["compound"] != "GANT61":
                rows.append(r)
    return rows

def load_all_seeds():
    return {s: load_loocv(s) for s in SEEDS}

def load_candidates():
    rows = []
    with open("outputs/final_candidates_v2.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def short_name(c):
    c = c.replace("_Barardozi", " (Bar.)").replace("_Lospinoso", " (Los.)")
    c = c.replace("_Manetti", " (Man.)").replace("_Quinoline", " (Quin.)")
    c = c.replace("Compound_", "Cmpd ")
    return c


# ===================================================================
# SLIDE 8 — LOOCV Performance (composite: bar chart + seed summary)
# Layout on slide: figure LEFT ~65%, text RIGHT ~35%
# ===================================================================
def slide8_figure():
    """Composite: per-compound bar chart (left 70%) + multi-seed table (right 30%)."""
    all_data = load_all_seeds()
    rows42 = load_loocv(42)
    rows42.sort(key=lambda r: float(r["held_out_prob"]))

    compounds = [r["compound"] for r in rows42]
    probs = [float(r["held_out_prob"]) for r in rows42]
    uncs = [float(r["held_out_uncertainty"]) for r in rows42]
    hits = [int(r["held_out_correct"]) for r in rows42]
    colors = [OK["blue"] if h else OK["orange"] for h in hits]

    # --- Create figure with gridspec ---
    fig = plt.figure(figsize=(11, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3.5], wspace=0.35)

    # ---- LEFT: Per-compound bar chart ----
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(compounds))
    ax.barh(y, probs, xerr=uncs, height=0.72, color=colors, edgecolor="none",
            error_kw=dict(ecolor="#666", capsize=2, linewidth=0.8))
    ax.axvline(x=0.5, color="#333", ls="--", lw=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels([short_name(c) for c in compounds], fontsize=8.5)
    ax.set_xlabel("Held-Out P(binder)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Compound LOOCV Probability (Seed 42)", fontsize=13)
    ax.invert_yaxis()

    # Compact value labels
    for i, (p, h) in enumerate(zip(probs, hits)):
        ax.text(min(p + uncs[i] + 0.015, 1.04), i, f".{int(p*1000):03d}",
                va="center", fontsize=7, color="#444")

    n_hits = sum(hits)
    hit_p = mpatches.Patch(color=OK["blue"], label=f"HIT: {n_hits}")
    miss_p = mpatches.Patch(color=OK["orange"], label=f"MISS: {28 - n_hits}")
    ax.legend(handles=[hit_p, miss_p], fontsize=9, loc="upper right",
              framealpha=0.9, handlelength=1.2)

    # ---- RIGHT: Multi-seed summary panels ----
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1],
                                                 hspace=0.55)

    # Panel A: Hit rate per seed
    ax_hr = fig.add_subplot(gs_right[0])
    seed_hr = []
    for s in SEEDS:
        rr = all_data[s]
        seed_hr.append(sum(int(r["held_out_correct"]) for r in rr) / len(rr) * 100)
    bars = ax_hr.bar([str(s) for s in SEEDS], seed_hr, color=OK["blue"],
                     edgecolor="white", width=0.65)
    mean_hr = np.mean(seed_hr)
    ax_hr.axhline(mean_hr, color=OK["vermil"], ls="--", lw=1.2)
    for b, v in zip(bars, seed_hr):
        ax_hr.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}%",
                   ha="center", fontsize=8.5, fontweight="bold")
    ax_hr.set_ylim(0, 95)
    ax_hr.set_ylabel("Hit Rate (%)", fontsize=10)
    ax_hr.set_title(f"5-Seed Stability: {mean_hr:.1f}% ± {np.std(seed_hr):.1f}%",
                    fontsize=10)
    ax_hr.tick_params(labelsize=9)

    # Panel B: AUROC per seed
    ax_auc = fig.add_subplot(gs_right[1])
    seed_auc = []
    for s in SEEDS:
        rr = all_data[s]
        seed_auc.append(np.mean([float(r["val_auroc"]) for r in rr]))
    bars2 = ax_auc.bar([str(s) for s in SEEDS], seed_auc, color=OK["green"],
                       edgecolor="white", width=0.65)
    for b, v in zip(bars2, seed_auc):
        ax_auc.text(b.get_x() + b.get_width()/2, v + 0.0003,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax_auc.set_ylim(0.990, 1.001)
    ax_auc.set_ylabel("AUC-ROC", fontsize=10)
    ax_auc.set_title(f"Mean AUC-ROC: {np.mean(seed_auc):.4f}", fontsize=10)
    ax_auc.tick_params(labelsize=9)

    # Panel C: FPR per seed
    ax_fpr = fig.add_subplot(gs_right[2])
    seed_fpr = []
    for s in SEEDS:
        rr = all_data[s]
        seed_fpr.append(np.mean([float(r["fold_fpr_default"]) for r in rr]) * 100)
    bars3 = ax_fpr.bar([str(s) for s in SEEDS], seed_fpr, color=OK["purple"],
                       edgecolor="white", width=0.65)
    for b, v in zip(bars3, seed_fpr):
        ax_fpr.text(b.get_x() + b.get_width()/2, v + 0.08,
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax_fpr.set_ylim(0, max(seed_fpr) * 1.5)
    ax_fpr.set_ylabel("FPR (%)", fontsize=10)
    ax_fpr.set_xlabel("Seed", fontsize=10)
    ax_fpr.set_title(f"Mean FPR @ 0.5: {np.mean(seed_fpr):.2f}%", fontsize=10)
    ax_fpr.tick_params(labelsize=9)

    plt.tight_layout()
    out = OUT / "slide8_loocv_composite.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===================================================================
# SLIDE 9 — Ablation Study (composite: ablation bars + detection heatmap)
# Layout on slide: figure TOP ~70%, text BOTTOM ~30%
# ===================================================================
def slide9_figure():
    """Composite: ablation horizontal bars (left) + compact heatmap (right)."""
    all_data = load_all_seeds()

    # ---- Ablation data ----
    conds = [
        ("Full Model\n(baseline)",          75.0, 1.72,  0.0),
        ("No BindingDB\nPretraining",       67.9, 0.38, -7.1),
        ("No Focal Loss\n(BCE instead)",    67.9, 0.87, -7.1),
        ("No ZF Domain\nAdaptation",        71.4, 1.88, -3.6),
        ("No Morgan\nFingerprints",         71.4, 11.5, -3.6),
        ("No SMILES\nAugmentation",         78.6, 9.52, +3.6),
    ]
    labels = [c[0] for c in conds]
    hrs    = [c[1] for c in conds]
    fprs   = [c[2] for c in conds]
    deltas = [c[3] for c in conds]

    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 6], wspace=0.3)

    # ---- LEFT: Ablation bar chart (horizontal, single panel) ----
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(labels))

    # Bar colors: baseline=blue, worse=orange, better-but-overfitting=sky
    bar_colors = []
    for i, hr in enumerate(hrs):
        if i == 0:   bar_colors.append(OK["blue"])
        elif hr < 75: bar_colors.append(OK["orange"])
        else:         bar_colors.append(OK["sky"])

    ax.barh(y, hrs, height=0.6, color=bar_colors, edgecolor="none")
    ax.axvline(75.0, color="#333", ls="--", lw=0.9, alpha=0.5)

    for i, (hr, fpr, delta) in enumerate(zip(hrs, fprs, deltas)):
        # Hit rate + delta label
        if i == 0:
            lbl = f"{hr:.0f}%"
        else:
            lbl = f"{hr:.0f}% ({delta:+.1f}%)"
        ax.text(hr + 0.8, i - 0.08, lbl, va="center", fontsize=9.5,
                fontweight="bold" if i == 0 else "normal")
        # FPR below
        fpr_color = OK["vermil"] if fpr > 3.0 else OK["green"]
        ax.text(hr + 0.8, i + 0.22, f"FPR: {fpr:.1f}%", va="center",
                fontsize=7.5, color=fpr_color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("LOOCV Hit Rate (%)", fontsize=11)
    ax.set_xlim(0, 95)
    ax.set_title("Ablation: Component Contribution", fontsize=12)
    ax.invert_yaxis()

    # ---- RIGHT: Compact detection consistency heatmap ----
    ax2 = fig.add_subplot(gs[1])

    compound_names = [r["compound"] for r in all_data[42]]
    # Sort by mean prob across seeds (descending)
    mean_probs = {}
    for name in compound_names:
        ps = []
        for s in SEEDS:
            for r in all_data[s]:
                if r["compound"] == name:
                    ps.append(float(r["held_out_prob"]))
        mean_probs[name] = np.mean(ps)
    compound_names.sort(key=lambda c: mean_probs[c], reverse=True)

    matrix = np.zeros((len(compound_names), len(SEEDS)))
    for j, s in enumerate(SEEDS):
        for r in all_data[s]:
            if r["compound"] in compound_names:
                i = compound_names.index(r["compound"])
                matrix[i, j] = float(r["held_out_prob"])

    cmap = LinearSegmentedColormap.from_list("hm", [
        (0.0,  OK["orange"]), (0.35, "#FADDB5"),
        (0.5,  "#F0F0F0"),    (0.65, "#B3D7F0"),
        (1.0,  OK["blue"]),
    ])

    im = ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell text
    for i in range(len(compound_names)):
        for j in range(len(SEEDS)):
            v = matrix[i, j]
            tc = "white" if v > 0.82 or v < 0.18 else "#333"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.5, color=tc, fontweight="bold")

    ax2.set_xticks(range(len(SEEDS)))
    ax2.set_xticklabels([f"S{s}" for s in SEEDS], fontsize=9)
    ax2.set_yticks(range(len(compound_names)))
    ax2.set_yticklabels([short_name(c) for c in compound_names], fontsize=7.5)

    # n/5 labels on right edge
    for i in range(len(compound_names)):
        n_det = sum(1 for j in range(len(SEEDS)) if matrix[i, j] >= 0.5)
        col = OK["blue"] if n_det >= 3 else OK["orange"]
        ax2.text(len(SEEDS) - 0.35, i, f"{n_det}/5", ha="left", va="center",
                 fontsize=7, fontweight="bold", color=col)

    ax2.set_title("Detection Consistency (5 Seeds)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label("P(binder)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUT / "slide9_ablation_composite.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===================================================================
# SLIDE 10 — Screening & Docking (composite: scatter + pipeline funnel)
# Layout on slide: figure LEFT ~70%, text RIGHT ~30%
# ===================================================================
def slide10_figure():
    """Composite: ML-vs-docking scatter (left) + screening funnel (right)."""
    candidates = load_candidates()

    probs, docks, sites, novels = [], [], [], []
    for r in candidates:
        try:
            p = float(r["ensemble_prob"])
            d = float(r["best_dock_score"])
            site = r.get("best_site", "ZF4-5")
            novel = r.get("novel", "").lower() == "true"
        except (ValueError, KeyError):
            continue
        probs.append(p); docks.append(d); sites.append(site); novels.append(novel)

    probs = np.array(probs)
    docks = np.array(docks)

    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6.5, 4], wspace=0.35)

    # ---- LEFT: Scatter plot ----
    ax = fig.add_subplot(gs[0])
    zf23 = np.array([s == "ZF2-3" for s in sites])
    zf45 = ~zf23

    ax.scatter(probs[zf45], docks[zf45], c=OK["blue"], alpha=0.4, s=28,
               label=f"ZF4-5 (n={zf45.sum()})", marker="o", edgecolors="none")
    ax.scatter(probs[zf23], docks[zf23], c=OK["orange"], alpha=0.5, s=35,
               label=f"ZF2-3 (n={zf23.sum()})", marker="^", edgecolors="none")

    # Priority zone
    ax.axhspan(min(docks) - 0.3, -7.0, alpha=0.03, color=OK["green"])
    ax.axvline(0.85, color="#BBB", ls=":", lw=0.7)
    ax.axhline(-7.0, color="#BBB", ls=":", lw=0.7)

    n_pri = sum(1 for p, d in zip(probs, docks) if p >= 0.85 and d <= -7.0)
    ax.text(0.925, min(docks) + 0.15, f"Priority\n(n={n_pri})",
            fontsize=9, fontweight="bold", color=OK["green"],
            ha="center", va="bottom", alpha=0.8)

    # Best point annotation
    best_i = np.argmin(docks)
    ax.annotate(f"Best: {docks[best_i]:.2f}",
                xy=(probs[best_i], docks[best_i]),
                xytext=(probs[best_i] - 0.06, docks[best_i] - 0.3),
                fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1),
                bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="#aaa"))

    ax.set_xlabel("Ensemble P(binder)", fontsize=11)
    ax.set_ylabel("Best Docking Score (kcal/mol)", fontsize=11)
    ax.set_title(f"ML Score vs. Docking Affinity (n={len(probs)})", fontsize=12)
    ax.invert_yaxis()

    # Combined stats + legend box (bottom-right, above x-axis)
    ax.text(0.97, 0.15,
            f"Novel: {sum(novels)}/{len(novels)} ({sum(novels)/len(novels)*100:.0f}%)\n"
            f"Mean dock: {np.mean(docks):.2f} kcal/mol",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bbb", alpha=0.9))
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
              bbox_to_anchor=(0.99, 0.02))

    # ---- RIGHT: Screening pipeline funnel ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    stages = [
        ("Raw Library",          1176000, OK["blue"],   "Enamine + COCONUT"),
        ("Prescreened",          1070832, OK["sky"],    "MW, PAINS, Lipinski"),
        ("Very High\nConfidence", 9262,   OK["green"],  "P > 0.85"),
        ("Diverse Hits",          500,    OK["purple"], "Butina clustering"),
        ("Docked",                499,    OK["orange"], "GLI1 ZF2-3 & ZF4-5"),
    ]

    n_stages = len(stages)
    max_log = np.log10(stages[0][1])
    bar_h = 0.65

    for i, (label, count, color, note) in enumerate(stages):
        y_center = n_stages - 1 - i
        w = np.log10(count) / max_log
        left = (1 - w) / 2
        ax2.barh(y_center, w, left=left, height=bar_h,
                 color=color, edgecolor="white", linewidth=1.5)

        # Count inside bar
        fc = "white" if w > 0.45 else "#333"
        ax2.text(0.5, y_center + 0.02, f"{count:,}", ha="center", va="center",
                 fontsize=13, fontweight="bold", color=fc)
        # Label left
        ax2.text(left - 0.02, y_center, label, ha="right", va="center",
                 fontsize=9, color="#333")
        # Note right
        ax2.text(left + w + 0.02, y_center, note, ha="left", va="center",
                 fontsize=7.5, color="#777", style="italic")

        # Arrow
        if i < n_stages - 1:
            next_y = n_stages - 2 - i
            ax2.annotate("", xy=(0.5, next_y + bar_h/2 + 0.03),
                         xytext=(0.5, y_center - bar_h/2 - 0.03),
                         arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.2))

    ax2.set_xlim(-0.38, 1.38)
    ax2.set_ylim(-0.7, n_stages - 0.3)
    ax2.set_title("Screening Pipeline", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = OUT / "slide10_screening_composite.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===================================================================
def main():
    print("Generating slide-optimized composite figures...\n")
    print("[1/3] Slide 8 — LOOCV composite...")
    slide8_figure()
    print("[2/3] Slide 9 — Ablation + heatmap composite...")
    slide9_figure()
    print("[3/3] Slide 10 — Screening + docking composite...")
    slide10_figure()
    print(f"\nAll saved to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
