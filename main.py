import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import streamlit as st

# -----------------------------
# Load IMDb word index and model
# -----------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model("simple_rnn.h5")

# -----------------------------
# Decode review (optional)
# -----------------------------
def decode_review(encoded_review):
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in encoded_review]
    )

# -----------------------------
# Preprocess user input
# -----------------------------
def preprocess_text(text):
    words = text.lower().split()

    # Unknown words -> 2
    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review):
    processed_review = preprocess_text(review)

    prediction = model.predict(processed_review, verbose=0)

    sentiment = " Positive" if prediction[0][0] > 0.5 else " Negative"

    confidence = prediction[0][0]

    return sentiment, confidence


# =============================
# Streamlit UI
# =============================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="",
    layout="centered"
)

st.title(" IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the trained Simple RNN model will predict whether the review is **Positive** or **Negative**."
)

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was absolutely amazing. I loved every minute of it!"
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if "Positive" in sentiment:
            st.success(sentiment)
        else:
            st.error(sentiment)

        st.write(f"**Model Score:** `{confidence:.4f}`")

        st.progress(float(confidence))