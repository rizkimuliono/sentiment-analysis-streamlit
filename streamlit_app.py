import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

    positive_comments = " ".join([comment for comment, sentiment in zip(filtered_comments, sentiments) if sentiment == 'positive'])
    negative_comments = " ".join([comment for comment, sentiment in zip(filtered_comments, sentiments) if sentiment == 'negative'])

    return sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts, positive_comments, negative_comments

st.title('Analisis Sentimen Komentar Media Sosial')

with st.form(key='search_form'):
    keyword = st.text_input("Masukkan keyword untuk pencarian:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if keyword:
        # Filter data berdasarkan keyword
        filtered_comments = [comment for comment in comments if keyword.lower() in comment.lower()]
    else:
        # Jika input text kosong, analisis seluruh data
        filtered_comments = comments

    if filtered_comments:
        sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts, positive_comments, negative_comments = analyze_sentiments(filtered_comments)

        # Menampilkan hasil dalam bentuk tabel
        st.markdown("### Comments and Sentiments")
        st.write(pd.DataFrame({'Comment': filtered_comments, 'Sentiment': sentiments}))

        # Menampilkan grafik persentase sentimen
        st.markdown("### Sentiment Distribution")
        fig, ax = plt.subplots()
        ax.pie([positive_percentage, neutral_percentage, negative_percentage], labels=['Positive', 'Neutral', 'Negative'],
               autopct='%1.1f%%', colors=['#4CAF50', '#FFC107', '#F44336'])
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)

        st.markdown("### Accuracy")
        st.write(f"Accuracy of Sentiment Analysis: **{accuracy:.2f}%**")

        # Menampilkan top hashtags dalam bentuk tabel
        st.markdown("### Top Hashtags")
        hashtag_df = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Jumlah'])
        st.dataframe(hashtag_df)

        # Menampilkan Word Cloud untuk sentimen positif
        st.markdown("### Word Cloud for Positive Sentiments")
        if positive_comments.strip():
            positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
            fig, ax = plt.subplots()
            ax.imshow(positive_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Tidak ada kata untuk sentimen positif")

        # Menampilkan Word Cloud untuk sentimen negatif
        st.markdown("### Word Cloud for Negative Sentiments")
        if negative_comments.strip():
            negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
            fig, ax = plt.subplots()
            ax.imshow(negative_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Tidak ada kata untuk sentimen negatif")
    else:
        st.write("Tidak ada tweet yang mengandung keyword tersebut.")
