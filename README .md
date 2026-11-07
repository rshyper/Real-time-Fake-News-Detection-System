# Fake News Detection System

### B.Tech Mini Project \| Machine Learning, NLP and Real-Time News Analysis

## 1. Project Overview

The **Fake News Detection System** analyses English news text and
classifies it as **Likely Fake** or **Likely Real** using a trained
machine-learning pipeline.

The project provides two Streamlit applications:

-   `app.py` --- original machine-learning news classification
    application.
-   `app_realtime.py` --- real-time extension with Google News RSS
    search, article extraction from public URLs, and lookup of existing
    published fact-check reviews through Google's Fact Check Tools API.

The real-time application reuses the existing trained model. It does not
replace the original dataset or model.

> **Important limitation:** A model prediction is not proof that a story
> is true or false. The model learns text patterns from a historical
> dataset. Google News RSS discovers articles; it does not verify them.
> The Fact Check Tools API searches for existing published fact-check
> reviews. No matching review means inconclusive, not true.

## 2. Objectives

-   Classify news text as likely fake or likely real.
-   Apply text preprocessing and TF-IDF feature extraction.
-   Train and compare supervised machine-learning classifiers.
-   Save and reuse the selected trained pipeline.
-   Provide Streamlit interfaces for news analysis.
-   Retrieve current headlines through RSS.
-   Extract article text from publicly accessible URLs.
-   Search for existing fact-check reviews related to a headline or
    claim.

## 3. Technologies Used

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  Programming                         Python

  Machine learning                    scikit-learn

  Text representation                 TF-IDF

  Classifiers                         Logistic Regression, Multinomial
                                      Naive Bayes, Linear SVM

  Data processing                     pandas

  Web interface                       Streamlit

  News retrieval                      Google News RSS, feedparser

  Article extraction                  trafilatura

  Fact-check lookup                   Google Fact Check Tools API

  HTTP/configuration                  requests, python-dotenv

  Model persistence                   joblib
  -----------------------------------------------------------------------

## 4. Dataset

The project uses the **ISOT Fake News Dataset**, distributed as two CSV
files:

-   `data/Fake.csv` --- articles labelled fake.
-   `data/True.csv` --- articles labelled true.

The training script assigns `0` to Fake and `1` to True/Real. It
combines article titles and text, performs basic cleaning, removes empty
and duplicate records, and uses a stratified 80:20 train/test split.

Dataset sources:

-   Kaggle:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
-   University of Victoria ISOT Lab:
    https://onlineacademiccommunity.uvic.ca/isot/datasets/
-   Dataset description:
    https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

The dataset is historical and has scope and collection limitations.
Review its documentation and terms before redistribution. Do not include
raw CSV files in a public repository unless redistribution is permitted.

## 5. Machine-Learning Methodology

The `train_model.py` workflow:

1.  Loads `Fake.csv` and `True.csv`.
2.  Assigns the corresponding class labels.
3.  Combines article titles and text.
4.  Cleans basic punctuation and URLs and removes empty or duplicate
    records.
5.  Creates stratified training and testing sets.
6.  Trains TF-IDF pipelines using Logistic Regression, Multinomial Naive
    Bayes and Linear SVM.
7.  Compares accuracy, precision, recall and F1-score.
8.  Saves the selected pipeline to `model/fake_news_pipeline.joblib`.
9.  Saves evaluation results to `reports/evaluation.json` and
    `reports/model_comparison.csv`.

Use the metrics produced by your actual training run. Do not assume a
particular accuracy in advance.

## 6. Real-Time Features

The real-time functionality is implemented in `realtime/` and exposed
through `app_realtime.py`.

### 6.1 Live News Search

`realtime/news_fetcher.py` retrieves headlines and RSS metadata from
Google News RSS based on a search query. These are discovery results,
not truth assessments.

### 6.2 Article URL Analysis

`realtime/article_extractor.py` uses Trafilatura to attempt to extract
article text from a public HTTP/HTTPS URL. Some websites block automated
access, require JavaScript or provide incomplete content. If extraction
fails, try another URL or paste the article text manually.

### 6.3 Existing Fact-Check Search

`realtime/fact_checker.py` calls Google's Fact Check Tools API to search
for existing fact-check claims and reviews. Results may include a claim,
publisher, rating, review date and source link when available.

The API searches published fact-check material; it does not
independently investigate every article or generate a new verdict.

### 6.4 Real-Time Application Tabs

The real-time interface provides:

-   **Live News Search** --- search current headlines and look up
    fact-checks for a headline.
-   **Check News URL** --- extract an article, run the existing ML model
    and search for related fact-checks.
-   **Check Article Text** --- analyse text pasted by the user and
    search for related fact-checks.

The ML classification and fact-check search are separate evidence
sources. Review the linked source material and interpret results
cautiously.

## 7. Project Structure

``` text
Fake_News_Detection_Project/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   └── fake_news_pipeline.joblib
├── realtime/
│   ├── __init__.py
│   ├── news_fetcher.py
│   ├── article_extractor.py
│   └── fact_checker.py
├── reports/
│   ├── evaluation.json
│   └── model_comparison.csv
├── app.py
├── app_realtime.py
├── train_model.py
├── download_dataset.py
├── requirements.txt
├── requirements_realtime.txt
├── .env.example
├── .gitignore
└── README.md
```

A local `venv/` virtual-environment directory may also be present. Do
not commit it to version control.

## 8. Installation and Setup

### Prerequisites

-   Python 3.10 or newer.
-   VS Code or another Python editor.
-   Internet connection for live RSS and fact-check lookups.

Open a terminal in the project root.

### 8.1 Activate the existing virtual environment

If the project already has a working `venv/`, activate it rather than
creating another.

**Windows Command Prompt:**

``` cmd
venv\Scripts\activate.bat
```

**Windows PowerShell:**

``` powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, select Command Prompt as the VS
Code terminal and use the Command Prompt command.

If no environment exists, create one:

``` cmd
py -m venv venv
```

### 8.2 Install dependencies

Install the original application dependencies:

``` cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the real-time feature dependencies:

``` cmd
pip install -r requirements_realtime.txt
```

## 9. Google Fact Check API Configuration

The real-time fact-check lookup requires a Google Fact Check Tools API
key.

1.  Open https://console.cloud.google.com/
2.  Create or select a Google Cloud project.
3.  Open
    https://console.cloud.google.com/apis/library/factchecktools.googleapis.com
4.  Enable the Fact Check Tools API for the selected project.
5.  Open https://console.cloud.google.com/apis/credentials
6.  Create an API key. Restrict it to the Fact Check Tools API where
    possible.
7.  Create a file named `.env` in the project root, beside
    `app_realtime.py`.
8.  Add this line, replacing the placeholder with your own key:

``` env
GOOGLE_FACT_CHECK_API_KEY=YOUR_ACTUAL_API_KEY
```

Keep `.env` private and ensure `.env` is listed in `.gitignore`. Never
commit, publish or share the actual API key. Keep `.env.example` as a
placeholder template only.

Google Cloud may apply quotas, account requirements or other
restrictions. Check the Cloud Console if requests fail.

## 10. Running the Applications

Run commands from the project root with the virtual environment
activated.

**Original ML application:**

``` cmd
streamlit run app.py
```

**Real-time news application:**

``` cmd
streamlit run app_realtime.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open
the displayed address in a browser. The applications are separate entry
points; the real-time application does not replace `app.py`.

## 11. Retraining the Model

To retrain and regenerate the model and evaluation reports:

``` cmd
python train_model.py
```

This may overwrite the existing pipeline and reports. Run it only when
you intend to retrain. Normal use of the real-time application reuses
`model/fake_news_pipeline.joblib` and does not require retraining.

## 12. Troubleshooting

  -----------------------------------------------------------------------
  Problem                             What to check
  ----------------------------------- -----------------------------------
  `Fake.csv` or `True.csv` not found  Confirm both files are directly
                                      inside `data/`.

  `ModuleNotFoundError`               Activate the virtual environment
                                      and install the relevant
                                      requirements file.

  Model file not found                Confirm
                                      `model/fake_news_pipeline.joblib`
                                      exists; if missing, run
                                      `python train_model.py`.

  API key not loaded                  Confirm `.env` is in the project
                                      root, named exactly `.env`, and
                                      contains
                                      `GOOGLE_FACT_CHECK_API_KEY=...`.

  Fact Check API error                Check API enablement, key validity,
                                      key restrictions, quota and Cloud
                                      account settings.

  No fact-check results               There may be no matching published
                                      review in the indexed results. This
                                      does not prove the claim true.

  Article extraction fails            The publisher may block automated
                                      extraction or require JavaScript.
                                      Try another public URL or paste the
                                      article text.

  Live news search fails              Check the internet connection and
                                      try again later.
  -----------------------------------------------------------------------

## 13. Limitations

-   The classifier is trained on a historical dataset and may not
    generalise to all current topics, publishers or writing styles.
-   A `Likely Fake` or `Likely Real` prediction is a model
    classification, not a verified factual conclusion.
-   RSS retrieves news items but does not establish credibility.
-   Article extraction depends on publisher accessibility and webpage
    structure.
-   Fact-check search depends on whether relevant reviews have already
    been published and indexed.
-   No matching fact-check result is inconclusive.
-   The system does not replace source investigation, professional
    journalism or human fact-checking.

## 14. Future Scope

Possible improvements include:

-   More diverse and regularly updated training data.
-   Evaluation on recent, independently labelled news articles.
-   Source credibility and publication-date analysis.
-   Better support for multilingual news.
-   Explainability showing influential text patterns.
-   A combined evidence-based credibility report with explicit
    uncertainty.
-   More robust article extraction and additional fact-check sources.

These are proposed improvements, not claims about features already
implemented.

## 15. Suggested Final-Year Project Report Sections

1.  Abstract
2.  Introduction and problem statement
3.  Objectives and scope
4.  Literature review and related work
5.  Dataset description
6.  Text preprocessing and TF-IDF
7.  Model architecture and training methodology
8.  Experimental results and model comparison
9.  System architecture and implementation
10. Real-time news retrieval and fact-check integration
11. Testing and application screenshots
12. Limitations and future scope
13. Conclusion
14. References

## 16. Dataset Reference

The ISOT Fake News Dataset is provided by the Information Security and
Object Technology (ISOT) Research Lab at the University of Victoria.
Consult the original dataset page and documentation for details about
collection, scope, attribution and terms of use:

https://onlineacademiccommunity.uvic.ca/isot/datasets/

------------------------------------------------------------------------

**Project:** Fake News Detection System\
**Academic level:** B.Tech Mini Project\
**Primary application:** Streamlit\
**Core approach:** TF-IDF and supervised machine learning, with
RSS-based news discovery and published fact-check lookup
