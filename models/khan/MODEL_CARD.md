# Model card: `khan`  (4 small round blue cell tumors (SRBCT))

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: BL, EWS, NB, RMS |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 63 samples (EWS: 23, RMS: 20, NB: 12, BL: 8); 2308 genes; source: Khan et al. 2001 Nat Med |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 1.000 ± 0.000**, macro-F1 1.000 (5 folds). Tuned logistic regression alone: 0.977 ± 0.039 |
| Always-answer-the-majority-class baseline | 0.365 |
| Input required | positional column names (GENE1, GENE2, GENE3, ...): works ONLY on files with exactly the same columns in the same order |

## Use
```powershell
hilbertfuse predict  --model models\khan --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\khan --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (63 samples): the accuracy above has wide uncertainty (fold std ±0.00).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
