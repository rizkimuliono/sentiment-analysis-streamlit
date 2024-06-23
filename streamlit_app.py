import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import re
from collections import Counter
import matplotlib.pyplot as plt

# Baca dataset dari URL
url = "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_tweet_sentimen_tayangan_tv.csv"
df = pd.read_csv(url)

# Menghapus baris yang memiliki nilai kosong
df = df.dropna()

# Pisahkan komentar dan labelnya
comments = df['Text Tweet'].tolist()
true_sentiments = df['Sentiment'].tolist()

# TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transform data
X = vectorizer.fit_transform(comments)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, true_sentiments)

def extract_hashtags(comments):
    hashtags = []
    for comment in comments:
        hashtags.extend(re.findall(r'#\w+', comment))
    return hashtags

def analyze_sentiments(filtered_comments):
    filtered_X = vectorizer.transform(filtered_comments)
    sentiments = knn.predict(filtered_X)

    total_comments = len(filtered_comments)
    positive_percentage = (list(sentiments).count('positive') / total_comments) * 100 if total_comments > 0 else 0
    neutral_percentage = (list(sentiments).count('netral') / total_comments) * 100 if total_comments > 0 else 0
    negative_percentage = (list(sentiments).count('negative') / total_comments) * 100 if total_comments > 0 else 0

    filtered_true_sentiments = [true_sentiments[i] for i, comment in enumerate(comments) if comment in filtered_comments]

    correct_predictions = sum(1 for pred, true in zip(sentiments, filtered_true_sentiments) if pred == true) if total_comments > 0 else 0
    accuracy = (correct_predictions / len(sentiments) * 100) if total_comments > 0 else 0

    hashtags = extract_hashtags(filtered_comments)
    hashtag_counts = Counter(hashtags).most_common()

    return sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts

st.title('Analisis Sentimen Komentar Media Sosial')

# Input dari pengguna untuk keyword pencarian
keyword = st.text_input("Masukkan keyword untuk pencarian:")

if keyword:
    # Filter data berdasarkan keyword
    filtered_comments = [comment for comment in comments if keyword.lower() in comment.lower()]

    if filtered_comments:
        sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts = analyze_sentiments(filtered_comments)

        # Menampilkan hasil dalam bentuk tabel
        st.write(pd.DataFrame({'Comment': filtered_comments, 'Sentiment': sentiments}))

        # Menampilkan grafik persentase sentimen
        fig, ax = plt.subplots()
        ax.pie([positive_percentage, neutral_percentage, negative_percentage], labels=['Positive', 'Neutral', 'Negative'],
               autopct='%1.1f%%', colors=['#4CAF50', '#FFC107', '#F44336'])
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)

        st.write(f"Accuracy of Sentiment Analysis: {accuracy:.2f}%")

        # Menampilkan top hashtags
        st.write("Top Hashtags:")
        for hashtag, count in hashtag_counts:
            st.write(f"{hashtag}: {count}")
    else:
        st.write("Tidak ada tweet yang mengandung keyword tersebut.")
