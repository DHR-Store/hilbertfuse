# Model card: `su`  (4 carcinoma tissue types)

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

Class labels are the numeric codes used in the source data file.

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: 1, 2, 3, 4 |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 102 samples (3: 28, 2: 26, 1: 25, 4: 23); 5565 genes; source: Su et al. 2001 Cancer Res |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 0.952 ± 0.067**, macro-F1 0.947 (5 folds). Tuned logistic regression alone: 0.981 ± 0.026 |
| Always-answer-the-majority-class baseline | 0.275 |
| Input required | positional column names (0, 1, 2, ...): works ONLY on files with exactly the same columns in the same order |

## Use
```powershell
hilbertfuse predict  --model models\su --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\su --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (102 samples): the accuracy above has wide uncertainty (fold std ±0.07).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
