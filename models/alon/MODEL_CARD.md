# Model card: `alon`  (colon tumor vs normal)

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: n, t |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 62 samples (t: 40, n: 22); 2000 genes; source: Alon et al. 1999 PNAS |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 0.794 ± 0.099**, macro-F1 0.764 (5 folds). Tuned logistic regression alone: 0.827 ± 0.136 |
| Always-answer-the-majority-class baseline | 0.645 |
| Input required | probe/gene IDs (X1, X2, X3, ...): works on other data from the same platform that uses the same IDs and the same normalisation |

## Use
```powershell
hilbertfuse predict  --model models\alon --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\alon --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (62 samples): the accuracy above has wide uncertainty (fold std ±0.10).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
