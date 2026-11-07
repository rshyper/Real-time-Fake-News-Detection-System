# Dataset Description

## 1. Dataset Overview

This project uses the ISOT Fake News Dataset, provided as two CSV files:

-   `Fake.csv` --- articles labelled as fake.
-   `True.csv` --- articles labelled as true/real.

The project assigns the following labels:

    Numeric label Class
  --------------- -------------
                0 Likely Fake
                1 Likely Real

## 2. Data Cleaning and Preparation

According to the project's training evaluation file, **38,826 rows
remained after cleaning**.

The training workflow described in the project README combines article
titles and text, performs basic cleaning of punctuation and URLs, and
removes empty and duplicate articles.

The cleaned data was divided using a stratified train/test split:

  Split                              Rows
  -------------------------- ------------
  Training set                     31,060
  Test set                          7,766
  **Total after cleaning**     **38,826**

The split is approximately 80% training and 20% testing.

## 3. Test-Set Class Distribution

The classification report records the following test-set support:

  Class           Test examples
  ------------- ---------------
  Likely Fake             3,580
  Likely Real             4,186
  **Total**           **7,766**

These figures describe the test set. The supplied evaluation output does
not provide the complete post-cleaning class distribution for all 38,826
rows.

## 4. Dataset Limitations

The dataset is historical and reflects the sources, topics, time period
and collection choices used by its creators. Performance on this dataset
does not guarantee equally strong performance on current news, different
publishers, other languages or newly emerging topics.

A model prediction should therefore be treated as a text-pattern
classification, not as independent proof that a news report is true or
false.

## 5. Dataset Sources

-   Kaggle:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
-   University of Victoria ISOT Lab:
    https://onlineacademiccommunity.uvic.ca/isot/datasets/
-   Dataset description:
    https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
