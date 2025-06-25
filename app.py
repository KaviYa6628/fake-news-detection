import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean and predict
def predict_news(text):
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    label = "🟢 REAL News" if prediction == 1 else "🔴 FAKE News"
    return label

# Streamlit UI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or headline below to check if it's fake or real.")

user_input = st.text_area("Enter news text here:", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    elif len(user_input.strip().split()) < 5:
        st.warning("Please enter a full news sentence (minimum 5 words).")
    else:
        cleaned = re.sub(r"[^\w\s]", "", user_input.lower())
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]
        prob = model.predict_proba(vect_text)[0][1]  # Confidence for "REAL" class

        label = "🟢 REAL News" if prediction == 1 else "🔴 FAKE News"

        st.subheader("Prediction:")
        st.success(label)
        st.info(f"Confidence Score: {prob:.2f}")

