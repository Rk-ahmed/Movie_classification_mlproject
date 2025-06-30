# predict_genre.py

import pandas as pd
import joblib
from scipy.sparse import hstack

# Load saved components
model = joblib.load("models/genre_model.pkl")
le = joblib.load("models/genre_label_encoder.pkl")
tfidf = joblib.load("models/desc_vectorizer.pkl")
cat_columns = joblib.load("models/categorical_columns.pkl")

# Load new data
new_df = pd.read_csv("path_to_new_data.csv")  # Replace with your actual path
new_df['Description'].fillna('', inplace=True)

# TF-IDF transform
desc_tfidf = tfidf.transform(new_df['Description'])

# Handle missing values in categorical columns
new_df['Language'].fillna('Unknown', inplace=True)
new_df['Country'].fillna('Unknown', inplace=True)
new_df['Content_Rating'].fillna('Unknown', inplace=True)

# One-hot encode and align categorical features
categorical_features = pd.get_dummies(new_df[['Language', 'Country', 'Content_Rating']], drop_first=True)
for col in cat_columns:
    if col not in categorical_features:
        categorical_features[col] = 0
categorical_features = categorical_features[cat_columns]

# Numeric features
numeric_features = new_df[['Rating', 'Votes', 'Budget_USD', 'Duration']].fillna(0)

# Combine all features
X_new = hstack([desc_tfidf, categorical_features.values, numeric_features.values])

# Predict
predictions = model.predict(X_new)
predicted_genres = le.inverse_transform(predictions)

# Output results
new_df['Predicted_Genre'] = predicted_genres
print(new_df[['Description', 'Predicted_Genre']])