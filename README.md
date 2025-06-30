# 🎬 Movie Genre Classification with Machine Learning

This project is an end-to-end machine learning application that predicts the **genre of a movie** based on its description and other metadata such as rating, budget, and director. The final model is deployed using a Flask web application.

---

## 🚀 Features

- Predicts movie genres using a trained classification model
- Uses TF-IDF vectorization of descriptions
- Interactive web interface built with Flask
- Fully modular code and folder structure
- Supports retraining and testing via scripts
- Ready for deployment or integration

---

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Flask** for web deployment
- **Plotly**, **Matplotlib**, **Seaborn** for EDA
- **Joblib** for saving/loading models
- **VS Code**, **Git**, **GitHub**

---

## 📁 Project Structure

---
...
movie-genre-classification/
│
├── apps/                 # Flask app
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── data/                 # Dataset
│   └── movie_genre_classification_final.csv
│
├── models/               # Saved model files
│   ├── genre_model.pkl
│   ├── desc_vectorizer.pkl
│   ├── genre_label_encoder.pkl
│   └── categorical_columns.pkl
│
├── notebooks/            # EDA notebook
│   └── eda.ipynb
│
├── src/                  # Core scripts
│   ├── train_model.py
│   └── predict_genre.py
│
├── venv/                 # Virtual environment (ignored in Git)
├── requirements.txt
├── .gitignore
└── README.md

...
---

## 📊 Dataset

The dataset used for training contains metadata and descriptions of 50,000 movies, with the following key columns:

- `Title`, `Director`, `Duration`, `Rating`, `Votes`
- `Description`, `Language`, `Country`
- `Budget_USD`, `BoxOffice_USD`, `Genre`, etc.

---

## 🛠️ How to Run the Project

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/rk-ahmed/movie-genre-classification.git
cd movie-genre-classification
````

### 🔹 2. Set Up the Environment

```bash
python -m venv venv
venv\Scripts\activate   
```

### 🔹 3. Train the Model (Optional)

```bash
python src/train_model.py
```

### 🔹 4. Run the Flask App

```bash
python apps/app.py
```

Then go to:
🔗 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Predict from CLI (Optional)

You can also use `src/predict_genre.py` to test predictions from the terminal.

---

## 📷 Sample Screenshot

*Include a screenshot of your web app here if possible.*

---

## 📌 To Do

* [ ] Add Docker support
* [ ] Deploy to Render/Heroku
* [ ] Add more features to UI

---

##  Credits

Created by **\[Md. Rakib Ahmed]**
Inspired by real-world use cases in movie recommendation and classification.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).


