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

- `0`: Non-Disaster
- `1`: Disaster

**Dataset Note:** The project instructions mention 10,000 tweets, but the supplied dataset contains 7,613 records. The analysis was therefore performed using all 7,613 available records.

### 2.1 Target Distribution

| Category | Records | Percentage |
|---|---:|---:|
| Non-Disaster | 4,342 | 57.03% |
| Disaster | 3,271 | 42.97% |
| **Total** | **7,613** | **100%** |

### 2.2 Keyword Analysis

The 15 most frequent keywords in disaster-related tweets were identified using keyword frequency analysis. A bar chart was created to visualize the most frequently occurring keywords and understand common terms associated with disaster-related tweets.

## 3. Data Preprocessing

The following preprocessing operations were performed:

1. Converted tweet text to lowercase.
2. Removed URLs and website links.
3. Removed user mentions.
4. Removed hashtag symbols.
5. Removed special characters.
6. Normalized extra whitespace.
7. Tokenized the cleaned text.

The dataset was divided into training and testing sets using an 80:20 split with stratification to maintain the target class distribution.

| Dataset | Records |
|---|---:|
| Training set | 6,090 |
| Testing set | 1,523 |
| **Total** | **7,613** |

## 4. Feature Engineering

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the cleaned tweet text into numerical features suitable for machine learning.

The vectorizer was configured with the following parameters:

| Parameter | Configuration |
|---|---|
| Maximum features | 10,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Sublinear term frequency | Enabled |

Both unigrams and bigrams were included to capture individual words and short word combinations.

## 5. Model Development

Three machine learning algorithms were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest

Five-fold cross-validation was also performed to compare model performance and assess consistency across different data folds.

### 5.1 Cross-Validation Results

| Model | Mean F1-score | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.7401 | 0.0137 |
| Multinomial Naive Bayes | 0.7161 | 0.0177 |
| Random Forest | 0.7060 | 0.0170 |

Logistic Regression achieved the highest mean F1-score among the three original models.

## 6. Model Evaluation

The original Logistic Regression model achieved the following results on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 82.01% |
| Precision | 84.67% |
| Recall | 70.95% |
| F1-score | 77.20% |
| ROC-AUC | 86.86% |

These metrics were used to evaluate the model's classification performance.

### 6.1 Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression model through a search across 24 parameter combinations with five-fold cross-validation.

The best parameters were:

- `C`: 1
- `class_weight`: balanced
- `solver`: lbfgs

The best cross-validation F1-score reported by GridSearchCV was 0.7495.

The tuned model achieved the following test results:

| Metric | Score |
|---|---:|
| Accuracy | 80.04% |
| Precision | 76.76% |
| Recall | 76.76% |
| F1-score | 76.76% |

The tuned model improved recall compared with the original Logistic Regression model, while the original model achieved better test accuracy and F1-score.

### 6.2 Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.01% | 84.67% | 70.95% | 77.20% |
| Tuned Logistic Regression | 80.04% | 76.76% | 76.76% | 76.76% |
| Multinomial Naive Bayes | 81.75% | 88.84% | 65.75% | 75.57% |
| Random Forest | 78.79% | 82.90% | 63.76% | 72.08% |

The original Logistic Regression model was selected for deployment because it achieved the highest test accuracy (82.01%) and F1-score (77.20%) among the evaluated models. It also achieved the highest mean cross-validation F1-score among the three original models. The tuned model achieved a higher GridSearchCV cross-validation score and improved test recall, but its test F1-score was slightly lower.

### 6.3 Evaluation Visualizations

The following evaluation visualizations were generated:

- **Confusion Matrix:** Used to examine correct and incorrect classifications.
- **ROC Curve:** Used to evaluate the model's ability to distinguish between disaster-related and non-disaster-related tweets.
- **Precision-Recall Curve:** Used to examine the relationship between precision and recall at different classification thresholds.

The ROC-AUC score of 86.86% provides an additional measure of the original Logistic Regression model's classification performance.

## 7. Generalization and Overfitting Check

The original Logistic Regression model achieved the following training and testing results:

| Metric | Training | Testing |
|---|---:|---:|
| Accuracy | 87.73% | 82.01% |
| F1-score | 84.40% | 77.20% |

The difference between training and testing performance indicates some overfitting. However, the model maintains reasonable performance on the held-out test dataset.

Four additional sample tweets were tested as a basic prediction sanity check. All four predictions matched their expected categories.

These additional examples provide a basic functional check and should not be considered a comprehensive evaluation of performance on unseen data.

## 8. Model Saving and Deployment

The trained Logistic Regression model and TF-IDF vectorizer were saved using Joblib for reuse during prediction.

The saved files are:

- `logistic_regression_model.pkl`
- `tfidf_vectorizer.pkl`

A Flask web application was developed to allow users to enter tweet text and receive a classification result.

The application was initially tested locally with multiple sample tweets. It was then deployed publicly using Render.

**Live Application:** [Disaster Tweet Classification – Live Web Application](https://disaster-tweet-classification-nlp.onrender.com)

The deployed application was tested using disaster-related and non-disaster-related sample tweets. Both predictions returned the expected categories during testing.

The application uses Gunicorn as its production web server and loads the saved model and TF-IDF vectorizer to process incoming text and generate predictions.

The web interface provides a text input field for entering tweets and displays the predicted category, making the classification process simple and accessible.

## 9. Conclusion

This project demonstrates an end-to-end NLP classification workflow, from data exploration and text preprocessing to machine learning evaluation and web application deployment.

Three machine learning algorithms were evaluated, and Logistic Regression achieved the strongest test F1-score among the evaluated models. Cross-validation and hyperparameter tuning were also performed to compare model performance and investigate improvements.

The project demonstrates the practical use of TF-IDF, Logistic Regression, Multinomial Naive Bayes, Random Forest, model evaluation, Joblib, Flask, Gunicorn, GitHub, and public cloud deployment using Render.

The resulting application provides a working interface for classifying tweet text into disaster-related and non-disaster-related categories.

## 10. Limitations

- The supplied dataset contains 7,613 records rather than the 10,000 mentioned in the project instructions.
- Predictions depend on patterns learned from the supplied dataset.
- The training and testing performance indicates some overfitting.
- The four additional sample tweets provide only a basic sanity check, not proof of performance across all unseen data.
- The model should not be used as the sole basis for real-world emergency decisions.
- The deployed application may take longer to respond after a period of inactivity because it uses a free hosting service.

## 11. Tools and Technologies

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Feature Extraction:** TF-IDF
- **Algorithms:** Logistic Regression, Multinomial Naive Bayes, Random Forest
- **Model Persistence:** Joblib
- **Web Framework:** Flask
- **Production Server:** Gunicorn
- **Version Control:** Git and GitHub
- **Deployment Platform:** Render

## 12. Project Links

- **GitHub Repository:** [Disaster Tweet Classification Using NLP](https://github.com/mahimathorecha/disaster-tweet-classification-nlp)
- **Live Application:** [Open Deployed Application](https://disaster-tweet-classification-nlp.onrender.com)

---

**End of Project Report**