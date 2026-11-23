# Disaster Tweet Classification Using NLP

## 1. Introduction

The objective of this project is to develop a Natural Language Processing (NLP) and Machine Learning system that classifies tweets as disaster-related or non-disaster-related.

The project covers data exploration, text preprocessing, feature extraction, model training, evaluation, and deployment through a Flask web application.

## 2. Dataset Description

The dataset contains 7,613 tweet records and five columns:

- `id`
- `keyword`
- `location`
- `text`
- `target`

The target variable represents:
- 0: Non-Disaster
- 1: Disaster

**Dataset note:** The project instructions mention 10,000 tweets, but the supplied dataset contains 7,613 records.

### Target Distribution

| Category | Records | Percentage |
|---|---:|---:|
| Non-Disaster | 4,342 | 57.03% |
| Disaster | 3,271 | 42.97% |

## 3. Data Preprocessing

The following preprocessing operations were performed:

1. Converted tweet text to lowercase.
2. Removed URLs and website links.
3. Removed user mentions.
4. Removed hashtag symbols.
5. Removed special characters.
6. Normalized extra whitespace.
7. Tokenized the cleaned text.

The dataset was divided into training and testing sets using an 80:20 split with stratification.

- Training records: 6,090
- Testing records: 1,523

## 4. Feature Engineering

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text into numerical features.

Configuration:
- Maximum features: 10,000
- N-gram range: (1, 2)
- Minimum document frequency: 2
- Maximum document frequency: 0.95
- Sublinear term frequency: Enabled

## 5. Model Development

Three machine learning algorithms were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest

Five-fold cross-validation was also performed to compare model performance.

### Cross-Validation Results

| Model | Mean F1-score | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.7401 | 0.0137 |
| Multinomial Naive Bayes | 0.7161 | 0.0177 |
| Random Forest | 0.7060 | 0.0170 |

## 6. Model Evaluation

The original Logistic Regression model achieved the following test results:

| Metric | Score |
|---|---:|
| Accuracy | 82.01% |
| Precision | 84.67% |
| Recall | 70.95% |
| F1-score | 77.20% |
| ROC-AUC | 86.86% |

### Hyperparameter Tuning

GridSearchCV was used to tune Logistic Regression parameters.

Best parameters:
- C: 1
- Class weight: balanced
- Solver: lbfgs

The tuned model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 80.04% |
| Precision | 76.76% |
| Recall | 76.76% |
| F1-score | 76.76% |

The tuned model improved recall, while the original Logistic Regression model achieved slightly better accuracy and F1-score.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.01% | 84.67% | 70.95% | 77.20% |
| Tuned Logistic Regression | 80.04% | 76.76% | 76.76% | 76.76% |
| Multinomial Naive Bayes | 81.75% | 88.84% | 65.75% | 75.57% |
| Random Forest | 78.79% | 82.90% | 63.76% | 72.08% |

The original Logistic Regression model was selected for the Flask application because it achieved the highest test accuracy and F1-score among the evaluated models.

## 7. Generalization and Overfitting Check

The original Logistic Regression model achieved:

| Metric | Training | Testing |
|---|---:|---:|
| Accuracy | 87.73% | 82.01% |
| F1-score | 84.40% | 77.20% |

The difference between training and testing performance indicates some overfitting, although the model maintains reasonable performance on the test dataset.

Four additional sample tweets were tested as a basic prediction sanity check. All four predictions matched their expected categories.

## 8. Model Saving and Deployment

The trained Logistic Regression model and TF-IDF vectorizer were saved using Joblib.

Saved files:
- `logistic_regression_model.pkl`
- `tfidf_vectorizer.pkl`

A Flask web application was developed to allow users to enter a tweet and receive a classification result.

The application was tested locally with multiple sample tweets. The predictions matched the expected categories in the tested examples.

## 9. Conclusion

This project demonstrates an end-to-end NLP classification workflow, from data exploration and preprocessing to model evaluation and web application development.

Among the evaluated models, the original Logistic Regression model achieved the strongest test F1-score and was selected for deployment.

The project also demonstrates the use of TF-IDF, cross-validation, hyperparameter tuning, model persistence, and Flask integration.

## 10. Limitations

- The dataset contains 7,613 records rather than the 10,000 mentioned in the project instructions.
- Predictions depend on patterns learned from the supplied dataset.
- The four additional sample tweets provide only a basic sanity check, not proof of performance on all unseen data.
- The application is running locally and has not been deployed to a public production server.
- The model should not be used as the sole basis for real-world emergency decisions.

## 11. Tools and Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TF-IDF, Logistic Regression, Multinomial Naive Bayes, Random Forest, Joblib, and Flask.