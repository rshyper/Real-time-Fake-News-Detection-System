# Model Evaluation and Comparison

## 1. Evaluation Setup

The project compared three supervised machine-learning classifiers using
the cleaned dataset and a stratified train/test split.

  Item                         Value
  --------------------- ------------
  Rows after cleaning         38,826
  Training rows               31,060
  Test rows                    7,766
  Selected model          Linear SVM

Class labels:

-   `0` --- Likely Fake
-   `1` --- Likely Real

## 2. Model Comparison

The following values are taken directly from
`reports/model_comparison.csv`.

  Model                         Accuracy    Precision       Recall     F1-score
  ------------------------- ------------ ------------ ------------ ------------
  Logistic Regression             99.11%       98.95%       99.40%       99.18%
  Multinomial Naive Bayes         96.36%       96.88%       96.34%       96.61%
  **Linear SVM**              **99.67%**   **99.64%**   **99.74%**   **99.69%**

Linear SVM achieved the highest accuracy and F1-score among the three
evaluated models and was selected as the best model in
`reports/evaluation.json`.

## 3. Linear SVM Classification Report

The detailed classification report for the selected model is:

  Class / Metric       Precision   Recall   F1-score   Support
  ------------------ ----------- -------- ---------- ---------
  Likely Fake             99.69%   99.58%     99.64%     3,580
  Likely Real             99.64%   99.74%     99.69%     4,186
  Macro average           99.67%   99.66%     99.66%     7,766
  Weighted average        99.67%   99.67%     99.67%     7,766

**Overall test accuracy: 99.6652% (approximately 99.67%).**

## 4. Confusion Matrix

The recorded confusion matrix is:

  Actual class  Predicted class     Likely Fake (0)   Likely Real (1)
  ------------------------------- ----------------- -----------------
  Likely Fake (0)                             3,565                15
  Likely Real (1)                                11             4,175

Interpreting rows as actual labels and columns as predicted labels in
the label order `[0, 1]`:

-   3,565 fake-labelled test examples were classified as Likely Fake.
-   15 fake-labelled examples were classified as Likely Real.
-   11 real-labelled examples were classified as Likely Fake.
-   4,175 real-labelled examples were classified as Likely Real.

There were 26 incorrect predictions among 7,766 test examples.

## 5. Interpretation

On this held-out test split, Linear SVM performed best among the three
tested classifiers. Its precision, recall and F1-score were all
approximately 99.6% or higher.

These are results on the project's test split, not a guarantee of
performance on unseen real-world news. The dataset is historical, and
real-time articles can differ in topic, publisher, language and writing
style. The model's output should be presented as **Likely Fake** or
**Likely Real**, not as a definitive fact-check verdict.

The real-time application adds separate evidence sources: Google News
RSS for headline discovery and Google's Fact Check Tools API for
searching existing published reviews. Neither RSS retrieval nor the ML
prediction independently verifies a claim.

## 6. Reproducibility

The reported values are taken from:

-   `reports/evaluation.json`
-   `reports/model_comparison.csv`

To regenerate the model and reports, run `python train_model.py` from
the project root with the required dependencies installed. This can
overwrite the existing saved model and reports, so only retrain when
intended.
