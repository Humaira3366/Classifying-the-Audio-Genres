import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Audio Genre Classifier", layout="wide", page_icon="🎵")

st.title("🎧 Classifying Audio Genres & Predicting Popularity")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("music_dataset_sample_100.csv")  # fallback

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Preprocessing
df['track_genre_encoded'] = LabelEncoder().fit_transform(df['track_genre'])
features = df.drop(columns=['popularity', 'track_genre', 'track_name', 'album_name', 'artists'])
target_popularity = df['popularity']
target_genre = df['track_genre_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_popularity, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor().fit(X_train, y_train)

y_train_cls, y_test_cls = target_genre[X_train.shape[0]:], target_genre[X_train.shape[0]:]
model_cls = RandomForestClassifier().fit(X_train, target_genre[:X_train.shape[0]])

# Sidebar input
st.sidebar.header("🎛️ Enter Track Features")
input_data = {}
for col in features.columns:
    if col in ['key', 'mode', 'time_signature']:
        input_data[col] = st.sidebar.slider(col, 0, 11 if col == 'key' else 1 if col == 'mode' else 7, 1)
    else:
        input_data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Predict Buttons
if st.sidebar.button("Predict Popularity"):
    pred_popularity = model_reg.predict(input_scaled)[0]
    st.sidebar.success(f"🎵 Predicted Popularity: {int(pred_popularity)} / 100")

if st.sidebar.button("Predict Genre"):
    pred_genre = model_cls.predict(input_scaled)[0]
    genre_label = df[['track_genre', 'track_genre_encoded']].drop_duplicates().set_index('track_genre_encoded')
    st.sidebar.success(f"🎼 Predicted Genre: {genre_label.loc[pred_genre][0]}")

# Visualization
st.subheader("📊 Popularity Distribution")
fig, ax = plt.subplots()
ax.hist(df['popularity'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("Popularity")
ax.set_ylabel("Number of Tracks")
st.pyplot(fig)

# Feature Importance
st.subheader("🔍 Feature Importance (Popularity Prediction)")
importances = model_reg.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
st.bar_chart(pd.Series(importances[sorted_idx], index=features.columns[sorted_idx]))
