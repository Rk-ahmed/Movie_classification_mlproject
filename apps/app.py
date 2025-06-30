from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from scipy.sparse import hstack

app = Flask(__name__)

# === Get absolute paths to models ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/genre_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../models/genre_label_encoder.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/desc_vectorizer.pkl")
CAT_COLUMNS_PATH = os.path.join(BASE_DIR, "../models/categorical_columns.pkl")

# === Load model and components ===
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    cat_columns = joblib.load(CAT_COLUMNS_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or encoders: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        description = request.form.get("description", "")
        language = request.form.get("language", "Unknown")
        country = request.form.get("country", "Unknown")
        content_rating = request.form.get("content_rating", "Unknown")
        rating = request.form.get("rating", 0)
        votes = request.form.get("votes", 0)
        budget = request.form.get("budget", 0)
        duration = request.form.get("duration", 0)

        if description.strip() == "":
            prediction = "Please enter a movie description."
        else:
            try:
                # Create a DataFrame for preprocessing
                input_df = pd.DataFrame([{
                    "Description": description,
                    "Language": language,
                    "Country": country,
                    "Content_Rating": content_rating,
                    "Rating": float(rating),
                    "Votes": float(votes),
                    "Budget_USD": float(budget),
                    "Duration": float(duration)
                }])

                # TF-IDF vectorization
                desc_vector = vectorizer.transform(input_df["Description"])

                # One-hot encode categorical features
                input_df[["Language", "Country", "Content_Rating"]] = input_df[["Language", "Country", "Content_Rating"]].fillna("Unknown")
                cat_features = pd.get_dummies(input_df[["Language", "Country", "Content_Rating"]], drop_first=True)

                # Align columns
                for col in cat_columns:
                    if col not in cat_features:
                        cat_features[col] = 0
                cat_features = cat_features[cat_columns]

                # Numeric features
                num_features = input_df[["Rating", "Votes", "Budget_USD", "Duration"]].fillna(0)

                # Combine all features
                final_input = hstack([desc_vector, cat_features.values, num_features.values])

                # Predict
                genre_pred = model.predict(final_input)[0]
                genre_name = label_encoder.inverse_transform([genre_pred])[0]
                prediction = f"🎬 Predicted Genre: {genre_name}"

            except Exception as e:
                prediction = f"Error during prediction: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
# To run the app, use the command: python app.py