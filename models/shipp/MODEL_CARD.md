# Model card: `shipp`  (DLBCL vs follicular lymphoma)

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: DLBCL, FL |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 77 samples (DLBCL: 58, FL: 19); 7129 genes; source: Shipp et al. 2002 Nat Med |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 0.920 ± 0.119**, macro-F1 0.898 (5 folds). Tuned logistic regression alone: 0.920 ± 0.110 |
| Always-answer-the-majority-class baseline | 0.753 |
| Input required | positional column names (0, 1, 2, ...): works ONLY on files with exactly the same columns in the same order |

## Use
```powershell
hilbertfuse predict  --model models\shipp --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\shipp --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (77 samples): the accuracy above has wide uncertainty (fold std ±0.12).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
