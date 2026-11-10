# Model card: `gordon`  (lung: mesothelioma vs adenocarcinoma)

**Research use only. Not a diagnostic device. Do not use for decisions about patients.**

| | |
|---|---|
| What it predicts | the class label of a tumor/sample expression profile: adenocarcinoma, mesothelioma |
| What it does NOT do | it does not identify the *cause* of a cancer, and it does not diagnose a patient. It recognises expression patterns that resembled each class in this one study. |
| Training data | 181 samples (adenocarcinoma: 150, mesothelioma: 31); 12533 genes; source: Gordon et al. 2002 Cancer Res |
| Model | HilbertFuse (HilbertFuseNet CNN + tuned logistic regression), 1024 selected genes, 80 epochs, trained on ALL samples (no hold-out) |
| **Honest accuracy** | 5-fold cross-validation (leak-free): **accuracy 1.000 ± 0.000**, macro-F1 1.000 (5 folds). Tuned logistic regression alone: 0.994 ± 0.012 |
| Always-answer-the-majority-class baseline | 0.829 |
| Input required | probe/gene IDs (1000_at, 1001_at, 1002_f_at, ...): works on other data from the same platform that uses the same IDs and the same normalisation |

## Use
```powershell
hilbertfuse predict  --model models\gordon --input your_samples.csv --out predictions.csv
hilbertfuse evaluate --model models\gordon --test your_labeled_samples.csv      # file needs a `label` column
```
The command first checks gene coverage and refuses files that do not contain this model's genes.

## Limits you must know
* Small cohort (181 samples): the accuracy above has wide uncertainty (fold std ±0.00).
* The cross-validation accuracy is for samples from the *same study*; accuracy on another hospital, platform or normalisation is unknown and usually lower (batch effects).
* Only the classes above are possible answers: a sample from any other tumor type will still be forced into one of them.
* The gene list from `hilbertfuse explain` shows what the model used; it is not proof of biological cause.
