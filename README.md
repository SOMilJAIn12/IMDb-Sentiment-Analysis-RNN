# 🎬 IMDb Sentiment Analysis using Simple RNN

A deep learning project that classifies IMDb movie reviews as **Positive** or **Negative** using a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras. The project also includes an interactive Streamlit web application for real-time sentiment prediction.

---

## 🚀 Features

- Predicts sentiment of custom movie reviews.
- Built using a Simple RNN architecture.
- Interactive web interface with Streamlit.
- Uses the IMDb Movie Review Dataset from TensorFlow/Keras.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit

---

## 📂 Project Structure

```
IMDb-Sentiment-Analysis/
│── main.py              # Streamlit application
│── simpleRnn.ipynb      # Model training notebook
│── prediction.ipynb     # Prediction testing notebook
│── simple_rnn.h5        # Trained model
│── requirements.txt
└── README.md
```

---

## 📊 Model

- Dataset: IMDb Movie Review Dataset
- Model: Simple Recurrent Neural Network (Simple RNN)
- Sequence Length: 500
- Output: Positive / Negative Sentiment

---

## ▶️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run main.py
```

---

## 💡 Example

**Input**

```
This movie was absolutely amazing. I loved every minute of it.
```

**Prediction**

```
Positive 
```

---

## 📌 Future Improvements

- Replace Simple RNN with LSTM/GRU.
- Improve preprocessing and tokenization.
- Deploy the application online using Streamlit Community Cloud.
- Add confidence visualization and probability charts.

---

## 👨‍💻 Author

**Somil Jain**

If you found this project useful, feel free to ⭐ the repository.