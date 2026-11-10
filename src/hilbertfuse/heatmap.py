"""Heat maps for HilbertFuse models (needs matplotlib).

Figures written to the output folder:
  heatmap_gene_image_all_classes.png / heatmap_gene_image_<class>.png
      Guided Grad-CAM map on the Hilbert gene image: where (which genes) the CNN looked for each class
      (the counterpart of Figure 1B of the paper; CNN-based models only).
  heatmap_top_genes_expression.png
      expression of the top genes (rows) across all samples (columns) grouped by class (counterpart of Figure 4).
  heatmap_class_gene_scores.png
      importance of each top gene for each class.
  heatmap_sample_<i>.png   (with --sample i)
      one sample: its gene image, the map for the predicted class, and the overlay.
  heatmap_gene_scores.csv
"""
from __future__ import annotations
import os
import re
from typing import Optional
import numpy as np
import pandas as pd


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise SystemExit("Heat maps need matplotlib. Install it with:  pip install matplotlib") from e


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _unwrap(clf):
    n = clf.network if hasattr(clf, "network") else clf
    return n.members[0] if hasattr(n, "members") else n


def _grid(values_flat, pix, side):
    """gene-ordered vector -> (side, side) array with NaN on pixels that hold no gene."""
    g = np.full(side * side, np.nan)
    g[pix] = values_flat
    return g.reshape(side, side)


def _annotate(ax, grid, genes, pix, side, n, fs=6):
    order = np.argsort(-np.nan_to_num(grid.ravel()[pix], nan=-1))[:n]
    for r, gi in enumerate(order, 1):
        y, x = divmod(int(pix[gi]), side)
        ax.plot(x, y, "o", ms=3.2, mfc="none", mec="white", mew=0.8)
        ax.text(x + 0.6, y - 0.6, f"{genes[gi]}", fontsize=fs, color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))


def make_heatmaps(clf, X: pd.DataFrame, y: Optional[np.ndarray], out_dir: str, top_k: int = 40,
                  sample: Optional[int] = None, max_per_class: int = 100, method: str = "guided_gradcam") -> list:
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    os.makedirs(out_dir, exist_ok=True)
    net = _unwrap(clf)
    classes = list(clf.classes_)
    labeled = y is not None
    ys = pd.Series(np.asarray(y).astype(str) if labeled else clf.predict((X, None)), index=range(len(X)))
    src = "true labels" if labeled else "PREDICTED classes (no label column given)"
    files = []

    # ---- gene scores (works for CNN, fused, ensemble and linear models)
    res = clf.explain((X, ys.values), top_k=max(top_k, 10), max_per_class=max_per_class) \
        if not hasattr(clf, "head") else clf.explain(None, top_k=max(top_k, 10))
    glob, pc = res["global"], res["per_class"]
    pd.concat([glob.assign(table="global", **{"class": ""}), pc.assign(table="per_class")]).to_csv(
        os.path.join(out_dir, "heatmap_gene_scores.csv"), index=False)
    files.append("heatmap_gene_scores.csv")

    pre = net.pre_
    pos = {g: i for i, g in enumerate(pre.gene_names_)}
    imgs = pre.transform(X)
    G = imgs[:, 0].reshape(len(X), -1)[:, pre.pix_]             # samples x genes (min-max scaled, training statistics)
    genes = [g for g in glob.gene if g in pos][:top_k]
    idx = [pos[g] for g in genes]
    score = dict(zip(glob.gene, glob.score))

    # ---- 1) expression heat map of the top genes across samples, grouped by class
    cls_idx = np.array([classes.index(c) if c in classes else len(classes) for c in ys])
    order_s = np.argsort(cls_idx, kind="stable")
    Z = G[:, idx]
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-8)
    Z = np.clip(Z, -2.5, 2.5)
    cm = np.array([Z[cls_idx == c].mean(0) if (cls_idx == c).any() else np.full(Z.shape[1], -9) for c in range(len(classes))])
    best = cm.argmax(0)
    order_g = sorted(range(len(genes)), key=lambda i: (best[i], -score.get(genes[i], 0)))
    Zp = Z[order_s][:, order_g].T
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(classes))]
    fig_w = min(16, max(7, 0.06 * len(X) + 4))
    fig_h = min(18, max(5, 0.2 * len(genes) + 2.2))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.25, 12], width_ratios=[40, 1], hspace=0.02, wspace=0.03)
    axc, ax, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axc.imshow(np.array([[cls_idx[i] for i in order_s]]), aspect="auto", cmap=ListedColormap(colors + ["#888888"]),
               vmin=0, vmax=len(classes))
    axc.axis("off")
    im = ax.imshow(Zp, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([genes[i] for i in order_g], fontsize=max(4, min(8, 300 / max(len(genes), 1))))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(X)} samples, grouped by class ({src})")
    fig.colorbar(im, cax=axb, label="expression z-score (blue low, red high)")
    fig.legend(handles=[Patch(color=colors[i], label=str(c)) for i, c in enumerate(classes)], loc="upper right",
               ncol=min(len(classes), 4), fontsize=8, frameon=False)
    fig.suptitle(f"Top {len(genes)} genes of the model across samples", x=0.02, ha="left", fontweight="bold")
    f = "heatmap_top_genes_expression.png"
    fig.savefig(os.path.join(out_dir, f), dpi=150, bbox_inches="tight"); plt.close(fig); files.append(f)

    # ---- 2) class x gene importance
    M = np.zeros((len(classes), len(genes)))
    for _, r in pc.iterrows():
        if r["gene"] in genes and r["class"] in classes:
            M[classes.index(r["class"]), genes.index(r["gene"])] = r["score"]
    M = M[:, order_g]
    fig, ax = plt.subplots(figsize=(min(16, 0.3 * len(genes) + 3), 0.5 * len(classes) + 2.6))
    div = (M < 0).any()
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r" if div else "viridis",
                   vmin=-np.abs(M).max() if div else 0, vmax=np.abs(M).max() if M.max() > 0 else 1)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xticks(range(len(genes))); ax.set_xticklabels([genes[i] for i in order_g], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, label="coefficient (class vs rest)" if div else "importance (Guided Grad-CAM, normalised per class)")
    ax.set_title("Which gene matters for which class", loc="left", fontweight="bold")
    f = "heatmap_class_gene_scores.png"
    fig.savefig(os.path.join(out_dir, f), dpi=150, bbox_inches="tight"); plt.close(fig); files.append(f)

    # ---- 3) gene-image (Hilbert) maps - CNN models only
    if not hasattr(net, "attribution_maps"):
        print("NOTE: gene-image maps need the CNN; this is a quick 'linear' model, so only the gene heat maps were written.")
        return files
    mp = net.attribution_maps((X, ys.values), "label", method=method, max_per_class=max_per_class)
    S, pix, gnames = mp["side"], mp["pix"], mp["genes"]
    cmaps = {}
    for c, name in enumerate(classes):
        sel = (mp["target"] == c) & (mp["pred"] == c)
        if sel.sum() < 3:
            sel = mp["target"] == c
        if sel.any():
            m = mp["attr"][sel].mean(0)
            cmaps[name] = (m / max(m.max(), 1e-12), int(sel.sum()))
    valid = np.zeros(S * S, bool); valid[pix] = True; valid = valid.reshape(S, S)

    def draw(ax, m, title, n_lab, fs):
        mm = np.ma.masked_where(~valid, m)
        cm_ = plt.get_cmap("magma").copy(); cm_.set_bad("#e6e6e6")
        im_ = ax.imshow(mm, cmap=cm_, vmin=0, vmax=1, interpolation="nearest")
        _annotate(ax, np.where(valid, m, np.nan), gnames, pix, S, n_lab, fs)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return im_
    n = len(cmaps)
    cols = min(n, 4); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    im_ = None
    for ax, (name, (m, k)) in zip(axs.ravel(), cmaps.items()):
        ax.axis("on"); im_ = draw(ax, m, f"{name}  (n={k})", 5, 5.5)
    fig.colorbar(im_, ax=axs, shrink=0.6, label="importance (0-1)")
    fig.suptitle(f"Guided Grad-CAM on the Hilbert gene image, mean per class ({src}); circles = top genes", fontweight="bold")
    f = "heatmap_gene_image_all_classes.png"
    fig.savefig(os.path.join(out_dir, f), dpi=150, bbox_inches="tight"); plt.close(fig); files.append(f)
    for name, (m, k) in cmaps.items():
        fig, ax = plt.subplots(figsize=(7, 6.2))
        im_ = draw(ax, m, f"Class {name}: where the CNN looked (n={k})", 10, 7)
        fig.colorbar(im_, ax=ax, shrink=0.8, label="importance (0-1)")
        f = f"heatmap_gene_image_{_safe(name)}.png"
        fig.savefig(os.path.join(out_dir, f), dpi=150, bbox_inches="tight"); plt.close(fig); files.append(f)

    # ---- 4) one sample
    if sample is not None:
        if not 0 <= sample < len(X):
            raise SystemExit(f"--sample must be between 0 and {len(X) - 1}")
        pr = clf.predict_proba((X.iloc[[sample]], None))[0]
        pname = classes[int(pr.argmax())]
        one = net.attribution_maps((X.iloc[[sample]], np.array([pname])), "label", method=method, max_per_class=1)
        m = one["attr"][0]; m = m / max(m.max(), 1e-12)
        expr = _grid(G[sample], pix, S)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
        cm_ = plt.get_cmap("viridis").copy(); cm_.set_bad("#e6e6e6")
        axs[0].imshow(np.ma.masked_where(~valid, expr), cmap=cm_, interpolation="nearest")
        axs[0].set_title(f"sample {sample} ({X.index[sample]}): gene image (expression)", fontsize=9)
        draw(axs[1], m, f"map for predicted class {pname} (confidence {pr.max():.2f})", 8, 6)
        axs[2].imshow(np.ma.masked_where(~valid, expr), cmap="gray", interpolation="nearest")
        ov = np.ma.masked_where((~valid) | (m < 0.15), m)
        axs[2].imshow(ov, cmap="magma", alpha=0.75, vmin=0, vmax=1, interpolation="nearest")
        axs[2].set_title("overlay (map > 0.15)", fontsize=9)
        for a in axs[[0, 2]]:
            a.set_xticks([]); a.set_yticks([])
        f = f"heatmap_sample_{sample}.png"
        fig.savefig(os.path.join(out_dir, f), dpi=150, bbox_inches="tight"); plt.close(fig); files.append(f)
    return files
