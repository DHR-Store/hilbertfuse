"""Write models/<name>/MODEL_CARD.md and models/README.md.
   python scripts/make_model_cards.py results/benchmark_results.csv models"""
import json, os, re, sys
import numpy as np
import pandas as pd

res_path, models_dir = sys.argv[1], sys.argv[2]
res = pd.read_csv(res_path)
res["method"] = res.method.replace({"new": "hfnet", "new_lr": "fuse"})
rows = []
for name in sorted(os.listdir(models_dir)):
    d = os.path.join(models_dir, name)
    info_p = os.path.join(d, "train_info.json")
    if not os.path.exists(info_p):
        continue
    info = json.load(open(info_p))
    genes = open(os.path.join(d, "required_genes.txt")).read().split("\n")
    positional = all(re.fullmatch(r"(GENE)?\d+", g) for g in genes[:50])
    r = res[res.dataset == name]
    def ms(m, col):
        v = r[r.method == m][col]
        return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)
    fa, fs, fn = ms("fuse", "accuracy"); ff, _, _ = ms("fuse", "f1_macro")
    la, ls, _ = ms("lr_mi", "accuracy")
    ids = ("positional column names (" + ", ".join(genes[:3]) + ", ...): works ONLY on files with exactly the same "
           "columns in the same order") if positional else ("probe/gene IDs (" + ", ".join(genes[:3]) +
           ", ...): works on other data from the same platform that uses the same IDs and the same normalisation")
    counts = ", ".join(f"{k}: {v}" for k, v in info["class_counts"].items())
    maj = max(info["class_counts"].values()) / info["n_samples"]
    useless = (not np.isnan(fa)) and fa <= maj + 0.03
    warn = (f"\n> **WARNING: this model is NOT useful.** Its cross-validated accuracy ({fa:.3f}) is no better than always "
            f"answering the most common class ({maj:.3f}). Do not use it.\n") if useless else ""
    note_cls = "\nClass labels are the numeric codes used in the source data file.\n" if all(c.isdigit() for c in info["classes"]) else ""
    card = f"""# Model card: `{name}`  ({info['task']})

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**
{warn}{note_cls}
| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: {', '.join(info['classes'])} |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | {info['n_samples']} samples ({counts}); {info['n_genes_in_file']} genes; source: {info['citation']} |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), {info['n_genes_used']} selected genes, {info['epochs']} epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy {fa:.3f} ± {fs:.3f}**, macro-F1 {ff:.3f} ({fn} folds). Tuned logistic regression alone: {la:.3f} ± {ls:.3f} |
| Always-answer-the-majority-class baseline | {maj:.3f} |
| Input required | {ids} |

## Use
```powershell
hilbertfuse predict  --model models\\{name} --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\\{name} --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort ({info['n_samples']} samples): the accuracy above has wide uncertainty (fold std ±{fs:.2f}).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
"""
    open(os.path.join(d, "MODEL_CARD.md"), "w", encoding="utf-8").write(card)
    json.dump(dict(model=name, task=info["task"], samples=info["n_samples"], classes=len(info["classes"]),
                   folds=int(fn), cv_accuracy=float(fa), cv_accuracy_std=float(fs), cv_macro_f1=float(ff),
                   logreg_cv_accuracy=float(la), majority_baseline=float(maj), not_useful=bool(useless)),
              open(os.path.join(d, "cv_scores.json"), "w"), indent=2)
    rows.append(dict(model=name, task=info["task"], samples=info["n_samples"], classes=len(info["classes"]),
                     cv_accuracy=f"{fa:.3f} ± {fs:.3f}", majority_baseline=f"{maj:.3f}", cv_macro_f1=f"{ff:.3f}", logreg_cv_accuracy=f"{la:.3f}",
                     portable_to_new_data="same probe IDs only" if not positional else "same column order only",
                     verdict="NOT USEFUL (no better than majority class)" if useless else "usable for research"))
tab = pd.DataFrame(rows).to_markdown(index=False)
open(os.path.join(models_dir, "README.md"), "w", encoding="utf-8").write(f"""# Trained models (real cancer data sets)

Each folder is a complete model trained on 100 % of one real data set. Accuracies are **5-fold cross-validated** estimates for samples from the same study, not guarantees for new hospitals or platforms.

{tab}

Research use only; not for diagnosing patients. See each `MODEL_CARD.md`.
""")
print(tab)
