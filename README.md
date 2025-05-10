# 🎧 Classifying Audio Genres and Predicting Track Popularity

## 📌 Problem Statement
This project aims to predict the popularity of songs using audio features and classify the genre of music tracks. These tasks help improve recommendation systems and user engagement on platforms like Spotify.

---

## 💼 Business Use Case
- Predicting which songs are likely to become popular based on audio characteristics.
- Automatically classifying new tracks into genres.
- Enhancing streaming service recommendations and playlist generation.

---

## 🧰 Technologies Used
- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- SHAP (for model interpretability)
- Jupyter Notebook

---

## 🔍 Project Workflow
1. **Data Preprocessing** (Label encoding, scaling)
2. **Regression Model**: Predict `popularity`
3. **Classification Model**: Predict `track_genre`
4. **Evaluation Metrics**:
   - Regression: R², MAE, MSE
   - Classification: Accuracy, Precision, Recall, F1 Score
5. **Visualization**:
   - Popularity distribution
   - Feature importance
   - Genre vs popularity
6. **Interpretability with SHAP**
7. **Conclusion and Insights**

---

## 📊 Results
- **Popularity prediction** showed high accuracy with RandomForestRegressor.
- **Genre classification** achieved excellent F1-scores with RandomForestClassifier.
- Feature importance analysis revealed key contributors like `valence`, `danceability`, and `energy`.

---

## 🧠 Key Insights
- Pop and rap tracks often rank high in popularity due to higher energy and danceability.
- Acousticness and instrumentalness are strong genre differentiators.
- SHAP values provide transparency into model decisions.

---

## 📁 How to Run
1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Open `audio.ipynb` in Jupyter
4. Run all cells to explore predictions and visualizations

---

## 🧑‍💻 Contact
For questions or collaboration:
- **Name**: Humaira Fathima N
- **Email**: humaira2004super@gmail.com
- **LinkedIn**: www.linkedin.com/in/humairafathima-n-778415295

