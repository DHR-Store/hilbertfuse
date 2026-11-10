# Trained models (real cancer data sets)

Each folder is a complete model trained on 100 % of one real data set. Accuracies are **5-fold cross-validated** estimates for samples from the same study, not guarantees for new hospitals or platforms.

| model   | task                                   |   samples |   classes | cv_accuracy   |   majority_baseline |   cv_macro_f1 |   logreg_cv_accuracy | portable_to_new_data   | verdict                                    |
|:--------|:---------------------------------------|----------:|----------:|:--------------|--------------------:|--------------:|---------------------:|:-----------------------|:-------------------------------------------|
| alon    | colon tumor vs normal                  |        62 |         2 | 0.794 ± 0.099 |               0.645 |         0.764 |                0.827 | same probe IDs only    | usable for research                        |
| golub   | ALL vs AML leukemia                    |        72 |         2 | 0.945 ± 0.056 |               0.653 |         0.94  |                0.987 | same probe IDs only    | usable for research                        |
| gordon  | lung: mesothelioma vs adenocarcinoma   |       181 |         2 | 1.000 ± 0.000 |               0.829 |         1     |                0.994 | same probe IDs only    | usable for research                        |
| khan    | 4 small round blue cell tumors (SRBCT) |        63 |         4 | 1.000 ± 0.000 |               0.365 |         1     |                0.977 | same column order only | usable for research                        |
| pomeroy | CNS embryonal tumor outcome (hard)     |        60 |         2 | 0.600 ± 0.160 |               0.65  |         0.539 |                0.617 | same probe IDs only    | NOT USEFUL (no better than majority class) |
| shipp   | DLBCL vs follicular lymphoma           |        77 |         2 | 0.920 ± 0.119 |               0.753 |         0.898 |                0.92  | same column order only | usable for research                        |
| singh   | prostate tumor vs normal               |       102 |         2 | 0.932 ± 0.064 |               0.51  |         0.932 |                0.912 | same column order only | usable for research                        |
| su      | 4 carcinoma tissue types               |       102 |         4 | 0.952 ± 0.067 |               0.275 |         0.947 |                0.981 | same column order only | usable for research                        |

Research use only; not for diagnosing patients. See each `MODEL_CARD.md`.
