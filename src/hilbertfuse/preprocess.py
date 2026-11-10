"""Leak-free preprocessing: (log) -> Min-Max -> MI selection (+redundancy pruning) -> gene layout -> images.

v0.2 additions over the paper (Wang et al. 2021):
  * fit on TRAINING data only (the paper selected genes on all data before CV);
  * selection="mi_rd": MI relevance + correlation-redundancy pruning (mRMR-like);
  * layout="hilbert": co-expressed genes are clustered, then laid on a Hilbert curve so that
    neighbours in the gene order stay neighbours in 2D (row-major breaks vertical locality);
  * channels="multi": 3 channels = expression | MI prior | per-sample rank;
  * images are (N, C, S, S) float32 in [0, 1].
"""
from __future__ import annotations
import math
import warnings
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.feature_selection import f_classif, mutual_info_classif

from .data import TableLike, order_genes


def hilbert_d2xy(side: int, d: int):
    """Index d along the Hilbert curve of a side x side grid (side = power of two) -> (x, y)."""
    x = y = 0
    t, s = d, 1
    while s < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _next_pow2(v: int) -> int:
    return 1 << max(0, (v - 1).bit_length())


class GeneImagePreprocessor:
    """Fit on a training table, transform any table into (N, C, S, S) images.

    Parameters
    ----------
    n_genes : genes kept (capped at #available).
    selection : "mi" (paper) | "mi_rd" (MI + redundancy pruning).
    rd_threshold : |Pearson r| above which a lower-MI gene is dropped as redundant (mi_rd).
    layout : "rowmajor" (paper; genomic order if gene_annotation) | "cluster" (cluster order,
             row-major fill) | "hilbert" (cluster order on a Hilbert curve).
    channels : "expr" (1 channel) | "multi" (expression, MI prior, per-sample rank).
    transform : "auto" (log2(1+x) if non-negative and heavy-tailed) | "log2" | "none".
    """

    def __init__(self, n_genes: int = 3600, gene_annotation: Optional[TableLike] = None,
                 mi_prefilter: Optional[int] = 8000, mi_n_neighbors: int = 3,
                 selection: str = "mi", rd_threshold: float = 0.9,
                 layout: str = "rowmajor", channels: str = "expr", transform: str = "auto",
                 image_side: Optional[int] = None, random_state: int = 0):
        assert selection in ("mi", "mi_rd") and layout in ("rowmajor", "cluster", "hilbert")
        assert channels in ("expr", "multi") and transform in ("auto", "log2", "none")
        self.n_genes, self.gene_annotation = n_genes, gene_annotation
        self.mi_prefilter, self.mi_n_neighbors = mi_prefilter, mi_n_neighbors
        self.selection, self.rd_threshold = selection, rd_threshold
        self.layout, self.channels, self.transform_mode = layout, channels, transform
        self.image_side, self.random_state = image_side, random_state

    @property
    def n_channels(self) -> int:
        return 3 if self.channels == "multi" else 1

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: Sequence) -> "GeneImagePreprocessor":
        y = np.asarray(y)
        genes = list(X.columns)
        A = X.to_numpy(dtype=np.float32)
        if self.transform_mode == "auto":
            self.log_ = bool(A.min() >= 0 and np.percentile(A, 99.9) > 50)
        else:
            self.log_ = self.transform_mode == "log2"
        A = self._log(A)

        mn, mx = A.min(0), A.max(0)
        keep = (mx - mn) > 1e-8
        if keep.sum() == 0:
            raise ValueError("All genes are constant.")
        self.min_, self.max_ = mn, mx
        A01 = (A[:, keep] - mn[keep]) / (mx[keep] - mn[keep])
        kept_idx = np.where(keep)[0]

        # candidate pool (ANOVA prefilter) -> MI ranking
        cand = np.arange(A01.shape[1])
        pool = self.mi_prefilter
        if pool and self.selection == "mi_rd":
            pool = max(pool, 3 * self.n_genes)
        if pool and A01.shape[1] > pool:
            F, _ = f_classif(A01, y)
            cand = np.argsort(-np.nan_to_num(F, nan=0.0))[:pool]
        mi = mutual_info_classif(A01[:, cand], y, n_neighbors=self.mi_n_neighbors,
                                 random_state=self.random_state)
        by_mi = np.argsort(-mi)
        n = min(self.n_genes, len(cand))
        if self.selection == "mi":
            pick = by_mi[:n]
        else:
            pick = self._prune_redundant(A01[:, cand[by_mi]], by_mi, n)
        chosen = cand[pick]                                   # indices into A01 columns
        sel_cols = kept_idx[chosen]                           # indices into original columns
        mi_sel = mi[pick]

        # gene order
        if self.layout == "rowmajor":
            names = [genes[i] for i in sel_cols]
            if self.gene_annotation is not None:
                ordered = order_genes(names, self.gene_annotation)
                perm = [names.index(g) for g in ordered]
                sel_cols, mi_sel = sel_cols[perm], mi_sel[perm]
        else:
            order = self._cluster_order(A01[:, chosen])
            sel_cols, mi_sel = sel_cols[order], mi_sel[order]

        self.input_genes_ = genes
        self.col_index_ = np.asarray(sel_cols)
        self.gene_names_ = [genes[i] for i in sel_cols]
        self.mi_prior_ = (mi_sel / max(mi_sel.max(), 1e-12)).astype(np.float32)
        self.mi_scores_ = pd.Series(mi_sel, index=self.gene_names_).sort_values(ascending=False)

        g = len(self.gene_names_)
        if self.layout == "hilbert":
            self.side_ = self.image_side or _next_pow2(math.ceil(math.sqrt(g)))
            xy = [hilbert_d2xy(self.side_, d) for d in range(g)]
            self.pix_ = np.array([y_ * self.side_ + x_ for x_, y_ in xy], dtype=np.int64)
        else:
            self.side_ = self.image_side or math.ceil(math.sqrt(g))
            self.pix_ = np.arange(g, dtype=np.int64)
        if self.side_ ** 2 < g:
            raise ValueError("image_side too small for n_genes")
        return self

    def _prune_redundant(self, Asorted: np.ndarray, by_mi: np.ndarray, n: int) -> np.ndarray:
        """Greedy: walk genes in MI order, skip any with |r| > threshold to an already kept gene."""
        Z = Asorted - Asorted.mean(0)
        Z /= np.maximum(Z.std(0), 1e-8)
        N, M = Z.shape
        maxc = np.zeros(M, dtype=np.float32)
        kept, skipped = [], []
        for i in range(M):
            if len(kept) >= n:
                break
            if maxc[i] > self.rd_threshold:
                skipped.append(i)
                continue
            kept.append(i)
            if i + 1 < M:
                c = np.abs(Z[:, i] @ Z[:, i + 1:]) / N
                np.maximum(maxc[i + 1:], c.astype(np.float32), out=maxc[i + 1:])
        for i in skipped:                                       # top up if pruning was too strict
            if len(kept) >= n:
                break
            kept.append(i)
        return by_mi[np.array(kept[:n])]

    @staticmethod
    def _cluster_order(A: np.ndarray) -> np.ndarray:
        if A.shape[1] < 3:
            return np.arange(A.shape[1])
        with np.errstate(all="ignore"):
            d = np.nan_to_num(pdist(A.T, metric="correlation"), nan=1.0)
        return leaves_list(linkage(d, method="average"))

    # ------------------------------------------------------------ transform
    def _log(self, A):
        return np.log2(np.maximum(A, 0.0) + 1.0) if self.log_ else A

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        missing = [g for g in self.input_genes_ if g not in X.columns]
        if len(missing) > 0.05 * len(self.input_genes_):
            warnings.warn(f"{len(missing)} of {len(self.input_genes_)} training genes are missing "
                          "from the new table; imputed with the training minimum.")
        sel_missing = [g for g in self.gene_names_ if g not in X.columns]
        if sel_missing:
            warnings.warn(f"{len(sel_missing)} selected genes missing at inference (e.g. {sel_missing[:3]}).")
        A = self._log(X.reindex(columns=self.input_genes_).to_numpy(dtype=np.float32))
        return np.where(np.isnan(A), self.min_[None, :], A)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = self._align(X)[:, self.col_index_]
        mn, mx = self.min_[self.col_index_], self.max_[self.col_index_]
        A = np.clip((A - mn) / (mx - mn), 0.0, 1.0)             # out-of-range values are clipped
        n, g, s = A.shape[0], A.shape[1], self.side_
        img = np.zeros((n, self.n_channels, s * s), dtype=np.float32)
        img[:, 0, self.pix_] = A
        if self.channels == "multi":
            img[:, 1, self.pix_] = self.mi_prior_[None, :]
            rank = A.argsort(1).argsort(1).astype(np.float32) / max(g - 1, 1)
            img[:, 2, self.pix_] = rank
        return img.reshape(n, self.n_channels, s, s)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def pixel_to_gene(self, flat_index: int) -> Optional[str]:
        hit = np.where(self.pix_ == flat_index)[0]
        return self.gene_names_[hit[0]] if len(hit) else None
