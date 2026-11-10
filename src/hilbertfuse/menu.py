"""Interactive menu:  hilbertfuse menu   (started automatically by start_windows.bat / start.sh)."""
from __future__ import annotations
import json
from pathlib import Path


def _models_dir() -> Path:
    for d in (Path("models"), Path(__file__).resolve().parents[2] / "models"):
        if d.is_dir():
            return d
    return Path("models")


def list_models():
    """[(path, task, not_useful_flag)] for every trained model folder found."""
    d = _models_dir()
    out = []
    for p in (sorted(d.iterdir()) if d.exists() else []):
        if (p / "meta.json").exists():
            info = json.loads((p / "train_info.json").read_text(encoding="utf-8")) if (p / "train_info.json").exists() else {}
            card = (p / "MODEL_CARD.md")
            bad = card.exists() and "NOT useful" in card.read_text(encoding="utf-8")
            out.append((p, info.get("task", "your own model"), bad))
    return out


def show_scores():
    """Print the stored, leak-free cross-validated scores of every trained model."""
    rows = []
    for p, task, _ in list_models():
        f = p / "cv_scores.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        print("\nNo stored scores found (models you trained yourself: run option 5 on your data).")
        return
    print("\nScores of the trained models (5-fold cross-validation, preprocessing re-fit inside every fold)")
    print(f"{'model':9s} {'samples':>7s} {'accuracy':>17s} {'macro-F1':>9s} {'logreg':>7s} {'majority':>9s}  verdict")
    for r in rows:
        verdict = "NOT USEFUL (no better than guessing the common class)" if r["not_useful"] else "usable for research"
        print(f"{r['model']:9s} {r['samples']:7d} {r['cv_accuracy']:9.3f} +/- {r['cv_accuracy_std']:.3f} "
              f"{r['cv_macro_f1']:9.3f} {r['logreg_cv_accuracy']:7.3f} {r['majority_baseline']:9.3f}  {verdict}")
    print("\n logreg = tuned logistic regression alone; majority = accuracy of always answering the most common class.")
    print(" These estimate accuracy on new samples from the SAME study. On other hospitals/platforms it is unknown and usually lower.")
    print(" Raw per-fold numbers: results/benchmark_results.csv   Details per model: models/<name>/MODEL_CARD.md")


def _ask(prompt, default=None):
    s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip().strip('"').strip("'")
    return s if s else (default if default is not None else "")


def _show_models(models):
    print("\nTrained models:")
    for i, (p, task, bad) in enumerate(models, 1):
        print(f"  {i}. {p.name:10s} {task}" + ("   [NOT USEFUL - do not use]" if bad else ""))
    print("  (details: models/<name>/MODEL_CARD.md)")


def _pick_model(models):
    _show_models(models)
    s = _ask("Model number, model name, or folder path")
    if s.isdigit() and 1 <= int(s) <= len(models):
        return str(models[int(s) - 1][0])
    for p, _, _ in models:
        if p.name == s:
            return str(p)
    return s if Path(s).is_dir() else None


def _file(prompt):
    s = _ask(prompt)
    if not s or not Path(s).exists():
        print(f"  File not found: {s!r}. Tip: drag the file into this window to paste its path.")
        return None
    return s


def _run(args):
    from .cli import main
    print("\n> hilbertfuse " + " ".join(args) + "\n", flush=True)
    try:
        main(args)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"\n(stopped: {e.code})")
    except Exception as e:  # keep the menu alive
        print(f"\nERROR: {type(e).__name__}: {e}")


MENU = """
================  HilbertFuse  (research use only - not for diagnosing patients)  ================
  1  List trained models
  2  Predict: give a model your samples, get predicted classes
  3  Evaluate: give a model LABELED samples, get accuracy
  4  Train my own model from my data
  5  Honest accuracy of my data (cross-validation, before trusting a model)
  6  Explain: which genes did the model rely on
  7  Show the accuracy scores of all trained models
  0  Exit
"""


def run_menu():
    while True:
        print(MENU)
        c = _ask("Choose")
        models = list_models()
        if c in ("0", "q", "exit"):
            print("Bye.")
            return
        if c == "1":
            _show_models(models)
        elif c == "7":
            show_scores()
        elif c == "8":
            m = _pick_model(models)
            f = _file("Samples file (CSV; with a label column it uses true classes, otherwise predicted classes)") if m else None
            if f:
                args = ["heatmap", "--model", m, "--data", f, "--label-col", _ask("Name of the label column", "label"),
                        "--top-k", _ask("How many top genes", "40"), "--out", _ask("Save heat maps to folder", "heatmaps")]
                one = _ask("Also draw one sample? Enter its row number (0 = first) or press Enter to skip", "")
                if one.isdigit():
                    args += ["--sample", one]
                _run(args)
        elif c in ("2", "3", "6"):
            m = _pick_model(models)
            if not m:
                print("  Model not found.")
                continue
            if c == "2":
                f = _file("Your samples file (CSV: rows = samples, columns = genes)")
                if f:
                    _run(["predict", "--model", m, "--input", f, "--out", _ask("Save predictions to", "predictions.csv")])
            elif c == "3":
                f = _file("Labeled samples file (CSV with a label column)")
                if f:
                    _run(["evaluate", "--model", m, "--test", f, "--label-col", _ask("Name of the label column", "label")])
            else:
                f = _file("Labeled samples file (CSV with a label column)")
                if f:
                    _run(["explain", "--model", m, "--data", f, "--label-col", _ask("Name of the label column", "label"),
                          "--top-k", _ask("How many top genes", "40"), "--out", _ask("Save gene lists to folder", "explain_out")])
        elif c in ("4", "5"):
            tr = _file("Your data file (CSV; rows = samples, columns = genes + a label column)")
            if not tr:
                continue
            lab = _ask("Name of the label column", "label")
            tp = ["--transpose"] if _ask("Is the file genes x samples instead? (y/N)", "n").lower().startswith("y") else []
            if c == "5":
                _run(["cv", "--data", tr, "--label-col", lab, "--folds", _ask("Number of folds", "5"), "--preset", "fuse"] + tp)
            else:
                test = _ask("Separate test file for a final accuracy (press Enter to skip)", "")
                name = _ask("Name for your new model", "mymodel")
                print("  Speed: 1 = quick (seconds, logistic regression)   2 = fast (CNN + linear, ~1-3 min)   3 = full (slowest)")
                preset = {"1": "linear", "2": "fast", "3": "fuse"}.get(_ask("Choose speed", "1"), "linear")
                args = ["train", "--train", tr, "--label-col", lab, "--out", str(_models_dir() / name),
                        "--preset", preset] + tp
                if preset == "fuse":
                    args += ["--epochs", _ask("Epochs (80 is a good start)", "80"), "--n-genes", _ask("Number of genes", "1024")]
                if test:
                    args += ["--test", test]
                _run(args)
                print(f"\nYour model is in: {_models_dir() / name}  (zip this folder to share it)")
        else:
            print("  Please type a number from the menu.")
        _ask("\nPress Enter to return to the menu", "")
