"""Table loading and gene-order helpers."""
from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

TableLike = Union[str, os.PathLike, pd.DataFrame]


def _read(path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if p.endswith((".tsv", ".txt", ".tab", ".tsv.gz", ".txt.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)  # csv / csv.gz


def load_table(
    data: TableLike,
    label_col: Optional[str] = "label",
    id_col: Optional[str] = None,
    transpose: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return (X, y). X: samples x genes (float32). y: labels (str) or None.

    Expected layout: one row per sample, one column per gene, plus a label column.
    For genes x samples files (TCGA/GDAC style) pass transpose=True: the first column
    holds gene names, the remaining columns are samples, and a row named `label_col`
    must hold the labels.
    """
    df = data.copy() if isinstance(data, pd.DataFrame) else _read(data)
    if transpose:
        df = df.set_index(df.columns[0]).T
        df.index.name = id_col or "sample"
        df = df.reset_index()
    if id_col and id_col in df.columns:
        df = df.set_index(id_col)
    y = None
    if label_col is not None and label_col in df.columns:
        y = df[label_col].astype(str)
        df = df.drop(columns=[label_col])
    X = df.apply(pd.to_numeric, errors="coerce")
    bad = X.columns[X.isna().all()].tolist()  # leftover text columns
    if bad:
        X = X.drop(columns=bad)
    X = X.fillna(0.0).astype(np.float32)
    X.columns = X.columns.astype(str)
    return X, y


_CHROM_RANK = {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24, "M": 25, "MT": 25}


def _chrom_rank(c) -> int:
    c = str(c).upper().replace("CHR", "")
    return _CHROM_RANK.get(c, 99)


def order_genes(genes, annotation: Optional[TableLike]) -> list:
    """Order genes by (chromosome, start). Genes absent from the annotation go last,
    keeping their original relative order.

    annotation: CSV/TSV/DataFrame with columns `gene`, `chromosome` and optionally `start`.
    """
    genes = list(genes)
    if annotation is None:
        return genes
    ann = annotation.copy() if isinstance(annotation, pd.DataFrame) else _read(annotation)
    ann.columns = [c.lower() for c in ann.columns]
    if "gene" not in ann.columns or "chromosome" not in ann.columns:
        raise ValueError("gene annotation needs columns: gene, chromosome[, start]")
    if "start" not in ann.columns:
        ann["start"] = 0
    ann = ann.drop_duplicates("gene").set_index("gene")
    keyed = []
    for i, g in enumerate(genes):
        if g in ann.index:
            keyed.append((_chrom_rank(ann.at[g, "chromosome"]), float(ann.at[g, "start"]), i, g))
        else:
            keyed.append((1000, 0.0, i, g))
    keyed.sort()
    return [k[3] for k in keyed]
