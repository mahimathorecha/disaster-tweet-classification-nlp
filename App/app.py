
from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

app = Flask(__name__)

# Get project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"

# Load saved model and vectorizer
model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


# Clean tweet text
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()

        if tweet:
            cleaned_tweet = clean_tweet(tweet)
            tweet_vector = tfidf.transform([cleaned_tweet])
            result = model.predict(tweet_vector)[0]

            prediction = "Disaster" if result == 1 else "Non-Disaster"

    return render_template(
        "index.html",
        prediction=prediction,
        tweet=tweet
    )


if __name__ == "__main__":
    app.run(debug=True)
