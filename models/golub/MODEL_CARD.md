# Model card: `golub`  (ALL vs AML leukemia)

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: ALL, AML |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 72 samples (ALL: 47, AML: 25); 7129 genes; source: Golub et al. 1999 Science |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 0.945 ± 0.056**, macro-F1 0.940 (5 folds). Tuned logistic regression alone: 0.987 ± 0.030 |
| Always-answer-the-majority-class baseline | 0.653 |
| Input required | probe/gene IDs (AFFX-BioB-5_at, AFFX-BioB-M_at, AFFX-BioB-3_at, ...): works on other data from the same platform that uses the same IDs and the same normalisation |

## Use
```powershell
hilbertfuse predict  --model models\golub --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\golub --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (72 samples): the accuracy above has wide uncertainty (fold std ±0.06).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
