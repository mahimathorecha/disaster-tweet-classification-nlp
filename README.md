# Disaster Tweet Classification Using NLP

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify tweets into two categories:

- **Disaster:** Tweets related to real disaster events.
- **Non-Disaster:** Tweets that are not related to real disaster events.

The project includes data exploration, text preprocessing, TF-IDF feature extraction, model training, evaluation, and a Flask web application.

## Dataset

- Dataset: Disaster Tweets
- Total records: 7,613
- Target column: `target`
- `0` = Non-Disaster
- `1` = Disaster

**Dataset note:** The project instructions mention 10,000 tweets, but the supplied dataset contains 7,613 records.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Flask
- Joblib

## Project Workflow

1. Data loading and exploration
2. Text cleaning and preprocessing
3. Tokenization
4. Train-test splitting
5. TF-IDF feature extraction
6. Model training and comparison
7. Cross-validation and hyperparameter tuning
8. Model evaluation
9. Saving and loading trained model artifacts
10. Flask web application development

## Model Results

The original Logistic Regression model achieved:

- Accuracy: 82.01%
- Precision: 84.67%
- Recall: 70.95%
- F1-score: 77.20%
- ROC-AUC: 86.86%

### Cross-Validation Results

| Model | Mean F1-score | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.7401 | 0.0137 |
| Multinomial Naive Bayes | 0.7161 | 0.0177 |
| Random Forest | 0.7060 | 0.0170 |

The tuned Logistic Regression model achieved an F1-score of 76.76%, with improved recall.

The original Logistic Regression model was selected for the Flask application because it achieved the highest test accuracy and F1-score among the evaluated models.

## Web Application

A Flask application allows users to enter a tweet and receive a prediction:

- Disaster
- Non-Disaster

### Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application from the main project directory:

```bash
python App/app.py
```

Open this address in your browser:

`http://127.0.0.1:5000`

## Project Structure

```text
P7/
|-- App/
|   |-- app.py
|   |-- templates/
|       |-- index.html
|-- Dataset/
|-- Models/
|   |-- logistic_regression_model.pkl
|   |-- tfidf_vectorizer.pkl
|-- Notebooks/
|   |-- 01_data_exploration.ipynb
|-- Reports/
|   |-- project_report.md
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## Disclaimer

This project is developed for educational purposes. Predictions are based on patterns learned from the training dataset and should not be used as the sole basis for real-world emergency decisions.