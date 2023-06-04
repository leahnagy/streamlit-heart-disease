---

# Classification Project

## Heart Disease Predictor

Created By: Leah Nagy

### Table of Contents

1.  [Slide
    Presentation](https://github.com/leahnagy/heart-disease-predictor/blob/main/slides.pdf)
2.  [Project
    Scripts](https://github.com/leahnagy/heart-disease-predictor/blob/main/code.ipynb)
3.  [Application Source
    Code](https://github.com/leahnagy/heart-disease-predictor/blob/main/streamlit_code.py)
4.  [Interactive Heart Disease Predictor
    App](https://share.streamlit.io/leahnagy/streamlit-heart-disease/main/proj_streamlit2.py)

### Project Description

The objective of this project was to construct classification models
capable of predicting the presence or history of heart disease. By
providing Functional Medicine practitioners with a tool to identify
patients at risk, the application facilitates the creation of
personalized prevention and treatment plans. The dataset used for the
project is derived from the CDC's Behavioral Risk Factor Surveillance
survey. Upon addressing class imbalance within the target variable, a
Logistic Regression model was deployed, and an interactive StreamLit app
was developed to predict user-specific probabilities of heart disease.

### Project Design

The utilized dataset originates from the CDC's survey and was obtained
via
[Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset).
With a binary classification of whether respondents have heart disease,
the insights derived from the model empower healthcare professionals to
preemptively identify at-risk patients. The rationale is simple but
powerful: early detection and risk mitigation can save lives.

### Data Collection and Processing

The dataset consists of 253,680 respondents and 21 features, including
lifestyle-related questions (e.g., alcohol consumption, smoking habits,
exercise habits) and physiological metrics (e.g., blood pressure, age,
weight, sex). Notably, the target variable exhibited severe class
imbalance, with heart disease reported in only 9% of respondents and 91%
being disease-free.

### Methodology

***Models***

A Logistic Regression model was employed due to its compatibility with
the project's requirement for probabilistic (soft) classification.
Although a Random Forest model was also tested, it did not significantly
improve the scores nor offer probabilistic predictions.

***Class Imbalance***

Various strategies were tested, including class weights, random
over-sampling, SMOTE, and random under-sampling. Ultimately, class
weights were selected as they preserve all original samples without
generating new ones, thereby reducing the risk of overfitting.

***Model Evaluation and Selection***

The dataset was split (80/20 train vs. holdout), and the model was
assessed using stratified 5-fold cross-validation on the training set.
The holdout set was used solely to score the final model.

F2 score was chosen as the primary evaluation metric to underscore the
importance of recall in this context---minimizing false negatives is
crucial in a healthcare setting.

**Final Logistic Regression 5-Fold CV Scores:**

-   Accuracy Score:0.694

-   Recall Score: 0.794

-   Precision Score: 0.20

-   F2 Score: 0.506

-   Brier Score: 0.694

**Holdout Scores:**

-   Accuracy Score: 0.700

-   Recall Score: 0.792

-   Precision Score: 0.210

-   F2 Score: 0.510

-   Brier Score: 0.700

### Tools Employed

-   Data manipulation: Numpy and Pandas
-   Modeling: Scikit-learn

### 
