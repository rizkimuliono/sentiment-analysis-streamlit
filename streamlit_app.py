import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd

# Data contoh untuk pelatihan model
data = {
    'text': [
        "Saya sangat senang dengan pelayanan yang diberikan",
        "Pelayanan yang diberikan sangat buruk",
        "Produk ini luar biasa bagus",
        "Saya kecewa dengan produk ini",
        "Ini pengalaman belanja yang menyenangkan",
        "Saya tidak puas dengan produk ini"
    ],
    'sentiment': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative'
    ]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan judul
st.title("Analisis Sentimen Menggunakan SVM")

# Menampilkan form input teks
user_input = st.text_area("Masukkan teks untuk analisis sentimen:")

# Fungsi untuk melatih model SVM
def train_model():
    X = df['text']
    y = df['sentiment']
    
    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat pipeline dengan TfidfVectorizer dan SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    
    # Melatih model
    model.fit(X_train, y_train)
    
    return model

# Melatih model
model = train_model()

# Jika ada teks yang dimasukkan oleh pengguna
if user_input:
    # Memprediksi sentimen dari teks yang dimasukkan
    prediction = model.predict([user_input])
    
    # Menampilkan hasil prediksi
    st.write("Prediksi sentimen:", prediction[0])
else:
    st.write("Masukkan teks di atas untuk mendapatkan prediksi sentimen.")

