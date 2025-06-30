# train_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# Load data
df = pd.read_csv(r'D:\Shakti Foundation\Data Science Project\Movie Genre Classification\data\movie_genre_classification_final.csv', encoding='utf-8')
df.dropna(subset=['Description', 'Genre'], inplace=True)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=500)
desc_tfidf = tfidf.fit_transform(df['Description'])

# Handle missing values in categorical columns
df['Language'].fillna('Unknown', inplace=True)
df['Country'].fillna('Unknown', inplace=True)
df['Content_Rating'].fillna('Unknown', inplace=True)

# One-hot encode categorical features
categorical_features = pd.get_dummies(df[['Language', 'Country', 'Content_Rating']], drop_first=True)
cat_columns = categorical_features.columns  # Save for later use

# Numeric features
numeric_features = df[['Rating', 'Votes', 'Budget_USD', 'Duration']].fillna(0)

# Combine all features
X = hstack([desc_tfidf, categorical_features.values, numeric_features.values])

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Genre'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/genre_model.pkl")
joblib.dump(le, "models/genre_label_encoder.pkl")
joblib.dump(tfidf, "models/desc_vectorizer.pkl")
joblib.dump(cat_columns, "models/categorical_columns.pkl")

print("✅ Model, LabelEncoder, TF-IDF vectorizer, and categorical columns saved successfully!")