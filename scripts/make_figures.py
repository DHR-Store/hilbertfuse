"""Figures for slides (PNG, docs/figures/).  python scripts/make_figures.py results/benchmark_results_v2.csv
Needs only matplotlib + pandas (no torch)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
C = dict(data="#dbe7f5", prep="#e6f0d8", img="#fdebc8", net="#f6d5d5", lin="#e3dcf2", out="#d6efe9", edge="#444444")


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec=C["edge"], lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=C["edge"]))


def hilbert_xy(side, d):
    x = y = 0; t = d; s = 1
    while s < side:
        rx = 1 & (t // 2); ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx; y += s * ry; t //= 4; s *= 2
    return x, y


# 1 pipeline -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
ax.set_title("HilbertFuse workflow: from a gene-expression table to a prediction", fontsize=13, fontweight="bold", loc="left")
box(ax, 0.2, 3.5, 1.9, 1.0, "Gene table\nrows = samples\ncolumns = genes", C["data"])
box(ax, 2.5, 3.5, 2.0, 1.0, "1  Clean\nlog2(1+x) if needed\nMin-Max (train only)", C["prep"])
box(ax, 4.9, 3.5, 2.2, 1.0, "2  Select genes\nMutual information\n+ drop redundant (|r|>0.9)", C["prep"])
box(ax, 7.5, 3.5, 2.2, 1.0, "3  Order genes\ncluster by co-expression\nHilbert-curve layout", C["img"])
box(ax, 10.1, 3.5, 2.7, 1.0, "4  Gene image\n32 x 32, 3 channels:\nexpression | MI prior | rank", C["img"])
for x1, x2 in [(2.1, 2.5), (4.5, 4.9), (7.1, 7.5), (9.7, 10.1)]:
    arrow(ax, x1, 4.0, x2, 4.0)
box(ax, 8.0, 1.7, 3.2, 1.0, "HilbertFuseNet CNN\nDenseNet + SE gates\n+ multi-scale head", C["net"])
box(ax, 2.5, 1.7, 3.6, 1.0, "Logistic regression\non the selected genes\n(tuned C, seconds to train)", C["lin"])
arrow(ax, 11.4, 3.5, 10.3, 2.7)
arrow(ax, 5.9, 3.5, 4.6, 2.7)
box(ax, 5.1, 0.2, 3.2, 1.0, "5  Fuse: average the two\nprobability vectors", C["out"])
arrow(ax, 9.0, 1.7, 7.6, 1.2); arrow(ax, 4.6, 1.7, 5.6, 1.2)
box(ax, 9.0, 0.2, 3.8, 1.0, "Prediction + confidence\n(UNCERTAIN flag if < 0.70)\nGuided Grad-CAM -> genes", C["out"])
arrow(ax, 8.3, 0.7, 9.0, 0.7)
fig.savefig(f"{OUT}/fig_workflow.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 2 layout: row-major vs Hilbert -----------------------------------------
side = 16
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, name in zip(axs, ["Row-major fill (paper)", "Hilbert curve (HilbertFuse)"]):
    ax.set_xlim(-0.5, side - 0.5); ax.set_ylim(side - 0.5, -0.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    pts = [(i % side, i // side) for i in range(side * side)] if name.startswith("Row") else [hilbert_xy(side, i) for i in range(side * side)]
    idx = np.zeros((side, side))
    for i, (x, y) in enumerate(pts):
        idx[y, x] = i
    ax.imshow(idx, cmap="viridis", alpha=0.55)
    ax.plot([p[0] for p in pts[:128]], [p[1] for p in pts[:128]], color="black", lw=0.9)
    ax.set_title(name, fontsize=12, fontweight="bold")
axs[0].set_xlabel("gene i+1 is the right neighbour, but gene i+16 is the lower neighbour", fontsize=9)
axs[1].set_xlabel("genes close in the order stay close in 2D (every step moves 1 pixel)", fontsize=9)
fig.suptitle("Why the layout matters: colour = position of the gene in the ordered list; line = first 128 genes", fontsize=11)
fig.savefig(f"{OUT}/fig_layout.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 3 architecture ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3.9)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.9); ax.axis("off")
ax.set_title("HilbertFuseNet architecture (about 0.4 M parameters)", fontsize=13, fontweight="bold", loc="left")
box(ax, 0.1, 1.6, 1.3, 1.0, "Input\n3 x 32 x 32", C["img"], 8.5)
box(ax, 1.7, 1.6, 1.3, 1.0, "BatchNorm\n+ 3x3 conv stem", C["prep"], 8.5)
x = 3.3
for i, n in enumerate([4, 6, 8, 6]):
    box(ax, x, 1.6, 1.4, 1.0, f"Dense block {i+1}\n({n} layers)\n+ SE gate", C["net"], 8.5)
    arrow(ax, x - 0.3, 2.1, x, 2.1)
    if i < 3:
        box(ax, x + 1.5, 1.85, 0.55, 0.5, "Trans.", C["prep"], 7.5); arrow(ax, x + 1.4, 2.1, x + 1.5, 2.1)
        x += 2.4
    else:
        x += 1.4
arrow(ax, 1.4, 2.1, 1.7, 2.1); arrow(ax, 3.0, 2.1, 3.3, 2.1)
box(ax, 10.8, 2.7, 2.1, 0.8, "GAP of block 2", C["lin"], 8.5)
box(ax, 10.8, 1.65, 2.1, 0.8, "GAP of block 3", C["lin"], 8.5)
box(ax, 10.8, 0.6, 2.1, 0.8, "GAP of last block", C["lin"], 8.5)
ax.text(11.85, 0.2, "concatenate -> dropout -> linear -> class probabilities", ha="center", fontsize=8.5)
arrow(ax, x - 0.0, 2.1, 10.8, 1.0)
fig.savefig(f"{OUT}/fig_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 4 results bar chart -----------------------------------------------------
if len(sys.argv) > 1:
    res = pd.read_csv(sys.argv[1]); res["method"] = res.method.replace({"new": "hfnet", "new_lr": "fuse"})
    meths = [("lr_mi", "MI_LogReg", "#7aa6d6"), ("paper", "Paper-style DenseNet", "#bbbbbb"),
             ("hfnet", "HilbertFuseNet (CNN)", "#e08b8b"), ("fuse", "HilbertFuse (CNN+linear)", "#4c9a6a")]
    g = res[res.method.isin([m[0] for m in meths])].groupby(["dataset", "method"]).accuracy.mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(12, 4.8)); w = 0.2
    for k, (m, lab, col) in enumerate(meths):
        if m in g:
            ax.bar(np.arange(len(g)) + (k - 1.5) * w, g[m], w, label=lab, color=col)
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(g.index); ax.set_ylim(40, 103); ax.set_ylabel("accuracy (%), 5-fold CV")
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False); ax.set_title("Accuracy on 8 real cancer data sets", fontweight="bold", loc="left")
    ax.axhline(100, color="#999", lw=0.6)
    fig.savefig(f"{OUT}/fig_results.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
