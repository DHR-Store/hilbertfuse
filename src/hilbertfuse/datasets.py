"""Real cancer gene-expression benchmark datasets (public microarray collection `datamicroarray`).

Bundled copies live in <repo>/datasets/*.csv.gz; anything missing is downloaded from GitHub on demand
(needs `pip install rdata`). Columns: gene probes (X1..Xn) + `label`.
"""
from __future__ import annotations
import os
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

INFO = {  # name: (samples, genes, task, citation)
    "alon": (62, 2000, "colon tumor vs normal", "Alon et al. 1999 PNAS"),
    "golub": (72, 7129, "ALL vs AML leukemia", "Golub et al. 1999 Science"),
    "khan": (63, 2308, "4 small round blue cell tumors (SRBCT)", "Khan et al. 2001 Nat Med"),
    "shipp": (77, 7129, "DLBCL vs follicular lymphoma", "Shipp et al. 2002 Nat Med"),
    "singh": (102, 12600, "prostate tumor vs normal", "Singh et al. 2002 Cancer Cell"),
    "su": (102, 5565, "4 carcinoma tissue types", "Su et al. 2001 Cancer Res"),
    "gordon": (181, 12533, "lung: mesothelioma vs adenocarcinoma", "Gordon et al. 2002 Cancer Res"),
    "pomeroy": (60, 7128, "CNS embryonal tumor outcome (hard)", "Pomeroy et al. 2002 Nature"),
    "chiaretti": (128, 12625, "ALL molecular subtypes (6, imbalanced)", "Chiaretti et al. 2004 Blood"),
    "burczynski": (127, 22283, "Crohn's / colitis / normal (non-cancer)", "Burczynski et al. 2006 J Mol Diagn"),
    "yeoh": (248, 12625, "6 pediatric ALL subtypes", "Yeoh et al. 2002 Cancer Cell"),
}
_URL = "https://raw.githubusercontent.com/ramhiser/datamicroarray/master/data/{}.RData"


def list_datasets() -> pd.DataFrame:
    return pd.DataFrame([(k, *v) for k, v in INFO.items()],
                        columns=["name", "samples", "genes", "task", "citation"])


def _search_dirs(data_dir):
    dirs = [data_dir, os.environ.get("HILBERTFUSE_DATA"), "datasets",
            Path(__file__).resolve().parents[2] / "datasets", Path.home() / ".cache" / "hilbertfuse"]
    return [Path(d) for d in dirs if d]


def load(name: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame (samples x genes + 'label'). Downloads on first use if not bundled."""
    if name not in INFO:
        raise KeyError(f"unknown dataset {name}; choose from {sorted(INFO)}")
    for d in _search_dirs(data_dir):
        p = d / f"{name}.csv.gz"
        if p.exists():
            return pd.read_csv(p)
    return fetch(name, data_dir)


def fetch(name: str, out_dir: Optional[str] = None) -> pd.DataFrame:
    try:
        import rdata
    except ImportError as e:  # pragma: no cover
        raise ImportError("downloading needs `pip install rdata`") from e
    out = Path(out_dir or Path.home() / ".cache" / "hilbertfuse")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{name}.RData"
    urllib.request.urlretrieve(_URL.format(name), tmp)
    obj = rdata.read_rda(tmp)[name]
    x = obj["x"]
    df = x.to_pandas() if hasattr(x, "to_pandas") else pd.DataFrame(np.asarray(x))
    df.columns = [str(c) for c in df.columns]
    df["label"] = np.asarray(obj["y"]).astype(str)
    df.round(4).to_csv(out / f"{name}.csv.gz", index=False, compression="gzip")
    tmp.unlink(missing_ok=True)
    return df
