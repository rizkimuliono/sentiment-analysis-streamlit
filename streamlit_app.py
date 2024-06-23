import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Fungsi untuk melatih model
@st.cache
def train_model():
    url = "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_tweet_sentimen_tayangan_tv.csv"
    data = pd.read_csv(url)
    
    # Periksa apakah kolom 'Sentiment' ada dalam data
    if 'Sentiment' not in data.columns or 'Text Tweet' not in data.columns:
        st.error("Kolom 'Sentiment' atau 'Text Tweet' tidak ditemukan dalam dataset.")
        return None
    
    data['label'] = data['Sentiment'].map({'positive': 1, 'negative': 0})
    
    # Periksa apakah mapping berhasil
    if data['label'].isnull().any():
        st.error("Nilai label tidak dapat dimapping dengan benar.")
        return None

    X = data['Text Tweet']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

# Memuat model yang sudah dilatih atau melatih model baru
model = train_model()

if model is not None:
    st.title('Analisis Sentimen Komentar Media Sosial')

    # Fungsi untuk menganalisis sentimen
    def analyze_sentiment(texts):
        sentiments = model.predict(texts)
        return sentiments

    # Input dari pengguna untuk keyword pencarian
    keyword = st.text_input("Masukkan keyword untuk pencarian:")

    if keyword:
        # Memuat data dari URL
        url = "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_tweet_sentimen_tayangan_tv.csv"
        tweets_df = pd.read_csv(url)
        
        # Memastikan kolom 'Text Tweet' ada dalam data
        if 'Text Tweet' in tweets_df.columns:
            # Filter data berdasarkan keyword
            filtered_df = tweets_df[tweets_df['Text Tweet'].str.contains(keyword, case=False, na=False)]
            
            if not filtered_df.empty:
                # Menganalisis sentimen
                filtered_df['Sentiment'] = analyze_sentiment(filtered_df['Text Tweet'])
                
                # Menampilkan hasil dalam bentuk tabel
                st.write(filtered_df)
                
                # Menampilkan grafik persentase sentimen
                sentiment_counts = filtered_df['Sentiment'].value_counts(normalize=True) * 100
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
            else:
                st.write("Tidak ada tweet yang mengandung keyword tersebut.")
        else:
            st.error("Kolom 'Text Tweet' tidak ditemukan dalam data.")
else:
    st.error("Gagal memuat model. Periksa kembali data dan coba lagi.")
