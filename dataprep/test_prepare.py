import os, sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(__file__))
import prepare_data as pdp


def _intensity(n=60, g=80, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.gamma(3, 100, (n, g)), columns=[f"G{i}" for i in range(g)])
    y = pd.Series(np.where(np.arange(n) % 2, "a", "b"))
    X.loc[y == "a", ["G0", "G1"]] += 400
    return X, y


def test_transform_has_no_cross_sample_dependence():
    X, _ = _intensity()
    prep = pdp.DataPrep(pdp.Config()).fit(X.iloc[:40])
    a = prep.transform(X.iloc[40:45])
    b = prep.transform(X.iloc[40:60]).iloc[:5]            # same rows, different company
    np.testing.assert_allclose(a.values, b.values, atol=1e-5)


def test_counts_library_size_invariance():
    r = np.random.default_rng(0)
    C = pd.DataFrame(r.poisson(30, (30, 60)), columns=[f"g{i}" for i in range(60)]).astype(float)
    prep = pdp.DataPrep(pdp.Config()).fit(C)
    assert prep.data_type_ == "counts" and prep.norm_ == "cpm"
    scaled = C.iloc[:3] * 5                                # same sample sequenced 5x deeper
    np.testing.assert_allclose(prep.transform(C.iloc[:3]).values, prep.transform(scaled).values, atol=1e-4)


def test_quantile_matches_reference():
    X, _ = _intensity()
    prep = pdp.DataPrep(pdp.Config(normalize="quantile")).fit(X)
    T = pdp.DataPrep.transform(prep, X)
    # after quantile normalisation every sample has the same sorted values over the retained genes
    full = prep._core(np.where(np.isnan(prep._align(X)), prep.med_raw_, prep._align(X)))
    np.testing.assert_allclose(np.sort(full, 1), np.tile(prep.q_ref_, (len(X), 1)), atol=1e-9)
    assert T.shape[1] <= X.shape[1]


def test_missing_filter_impute_and_constant_removed():
    X, _ = _intensity()
    X["BAD"] = np.where(np.arange(len(X)) % 2, np.nan, 1.0)          # 50 % missing
    X["CONST"] = 3.0
    X.iloc[0, 3] = np.nan
    prep = pdp.DataPrep(pdp.Config()).fit(X)
    out = prep.transform(X)
    assert "BAD" not in out.columns and "CONST" not in out.columns
    assert not out.isna().any().any()


def _write_messy(tmp_path):
    X, y = _intensity(80, 40, seed=3)
    df = X.copy()
    df["sid"] = [f"s{i}" for i in range(len(df))]
    df["pid"] = [f"p{i // 2}" for i in range(len(df))]               # two samples per patient
    df["label"] = [" A " if v == "a" else "B" for v in y]
    df.loc[0, "label"] = None
    df.loc[2, :] = df.loc[4, :]                                      # duplicate of another sample (same patient id too)
    df.loc[2, "sid"] = "dup"
    df.loc[6, list(X.columns)] = df.loc[8, list(X.columns)].values   # identical values ...
    df.loc[6, "label"] = "B" if df.loc[8, "label"].strip() == "A" else " A "   # ... but conflicting label
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p, df


def test_cli_cleaning_split_and_apply(tmp_path):
    raw, df = _write_messy(tmp_path)
    out = tmp_path / "out"
    pdp.main(["fit", "--input", str(raw), "--label-col", "label", "--id-col", "sid", "--group-col", "pid",
              "--label-case", "upper", "--out", str(out), "--min-class-size", "3"])
    tr = pd.read_csv(out / "train.csv.gz"); te = pd.read_csv(out / "test.csv.gz")
    assert set(tr.label) == {"A", "B"}                                # labels cleaned (strip + case)
    s_tr, s_te = pd.read_csv(out / "samples_train.csv"), pd.read_csv(out / "samples_test.csv")
    grp = dict(zip(df.sid, df.pid))
    assert not ({grp[s] for s in s_tr["sample"]} & {grp[s] for s in s_te["sample"]})   # no patient on both sides
    kept = set(s_tr["sample"]) | set(s_te["sample"])
    assert len({"dup", "s4"} & kept) == 1                             # exact duplicate pair: exactly one copy kept
    assert df.loc[0, "sid"] not in set(s_tr["sample"]) | set(s_te["sample"])           # missing label removed
    # apply() reproduces the test file exactly
    keep_ids = list(s_te["sample"])
    raw_te = df[df.sid.isin(keep_ids)].set_index("sid").loc[keep_ids].reset_index()
    p2 = tmp_path / "new.csv"; raw_te.to_csv(p2, index=False)
    pdp.main(["apply", "--prep", str(out / "prep_model.joblib"), "--input", str(p2), "--id-col", "sid",
              "--group-col", "pid", "--out", str(tmp_path / "new_clean.csv")])
    new = pd.read_csv(tmp_path / "new_clean.csv")
    assert "label" in new.columns
    np.testing.assert_allclose(new.drop(columns="label").values, te.drop(columns="label").values, atol=1e-4)


def test_apply_refuses_wrong_genes(tmp_path):
    raw, df = _write_messy(tmp_path)
    out = tmp_path / "o"
    pdp.main(["fit", "--input", str(raw), "--label-col", "label", "--id-col", "sid", "--out", str(out), "--min-class-size", "3"])
    bad = pd.DataFrame(np.random.rand(3, 10), columns=[f"other{i}" for i in range(10)])
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        pdp.main(["apply", "--prep", str(out / "prep_model.joblib"), "--input", str(tmp_path / "bad.csv"), "--out", str(tmp_path / "x.csv")])
