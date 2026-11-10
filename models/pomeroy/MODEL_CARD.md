# Model card: `pomeroy`  (CNS embryonal tumor outcome (hard))

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

> **WARNING: this model is NOT useful.** Its cross-validated accuracy (0.600) is no better than always answering the most common class (0.650). Do not use it.

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: died, survived |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 60 samples (survived: 39, died: 21); 7128 genes; source: Pomeroy et al. 2002 Nature |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 0.600 ± 0.160**, macro-F1 0.539 (5 folds). Tuned logistic regression alone: 0.617 ± 0.126 |
| Always-answer-the-majority-class baseline | 0.650 |
| Input required | probe/gene IDs (AFFX-BioB-5-at, AFFX-BioB-M-at, AFFX-BioB-3-at, ...): works on other data from the same platform that uses the same IDs and the same normalisation |

## Use
```powershell
hilbertfuse predict  --model models\pomeroy --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\pomeroy --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (60 samples): the accuracy above has wide uncertainty (fold std ±0.16).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
