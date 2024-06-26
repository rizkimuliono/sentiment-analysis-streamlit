import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Daftar sumber data
data_sources = {
    "Cyber Bullying": "https://raw.githubusercontent.com/rizkimuliono/sentiment-analysis-streamlit/main/dataset_komentar_instagram_cyberbullying.csv",
    "Tayangan TV": "https://raw.githubusercontent.com/rizkimuliono/sentiment-analysis-streamlit/main/dataset_tweet_sentimen_tayangan_tv.csv",
    "Cellular Service Provider": "https://raw.githubusercontent.com/rizkimuliono/sentiment-analysis-streamlit/main/dataset_tweet_sentiment_cellular_service_provider.csv",
    "Opini Film": "https://raw.githubusercontent.com/rizkimuliono/sentiment-analysis-streamlit/main/dataset_tweet_sentiment_opini_film.csv"
}

st.title('Analisis Sentimen Komentar Media Sosial')

# Pilih sumber data
selected_source = st.selectbox("Pilih sumber data:", list(data_sources.keys()))

if selected_source:
    url = data_sources[selected_source]
    df = pd.read_csv(url)

    # Menghapus baris yang memiliki nilai kosong
    df = df.dropna()

    # Pisahkan komentar dan labelnya
    comments = df.iloc[:, -1].tolist()  # Kolom terakhir adalah komentar
    true_sentiments = df.iloc[:, -2].tolist()  # Kolom kedua terakhir adalah sentimen

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

        # Confusion Matrix
        conf_matrix = confusion_matrix(filtered_true_sentiments, sentiments, labels=['positive', 'netral', 'negative'])

        # Classification Report
        class_report = classification_report(filtered_true_sentiments, sentiments, output_dict=True, zero_division=0)
        metrics = {
            "Accuracy": [accuracy],
            "Precision": [class_report['macro avg']['precision']],
            "Recall": [class_report['macro avg']['recall']],
            "F1-Score": [class_report['macro avg']['f1-score']]
        }

        return sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts, positive_comments, negative_comments, conf_matrix, metrics

    with st.form(key='search_form'):
        keyword = st.text_input("Masukkan keyword untuk pencarian:")
        submit_button = st.form_submit_button(label='Proses')

        # Menggunakan HTML dan CSS untuk tombol Bootstrap
        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #007bff;
                    color: white;
                    border-radius: 5px;
                    border: none;
                    padding: 10px 20px;
                    font-size: 16px;
                    cursor: pointer;
                }
                .stButton>button:hover {
                    background-color: #0056b3;
                }
            </style>
        """, unsafe_allow_html=True)

    if submit_button:
        if keyword:
            # Filter data berdasarkan keyword
            filtered_comments = [comment for comment in comments if keyword.lower() in comment.lower()]
        else:
            # Jika input text kosong, analisis seluruh data
            filtered_comments = comments

        if filtered_comments:
            sentiments, positive_percentage, neutral_percentage, negative_percentage, accuracy, hashtag_counts, positive_comments, negative_comments, conf_matrix, metrics = analyze_sentiments(filtered_comments)

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

            # Menampilkan confusion matrix
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            # Menampilkan metrik dalam bentuk tabel
            st.markdown("### Evaluation Metrics")
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df)

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
                st.markdown('<p style="color:red; font-size: 20px;">Tidak ada kata untuk sentimen positif</p>', unsafe_allow_html=True)

            # Menampilkan Word Cloud untuk sentimen negatif
            st.markdown("### Word Cloud for Negative Sentiments")
            if negative_comments.strip():
                negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
                fig, ax = plt.subplots()
                ax.imshow(negative_wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.markdown('<p style="color:red; font-size: 20px;">Tidak ada kata untuk sentimen negatif</p>', unsafe_allow_html=True)

            # Menampilkan grafik sebaran data titik cluster KNN
            st.markdown("### KNN Cluster Data Points")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[{'positive': 1, 'netral': 0, 'negative': -1}[s] for s in true_sentiments], cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ticks=[-1, 0, 1], format=plt.FuncFormatter(lambda val, loc: ['Negative', 'Neutral', 'Positive'][loc]))
            plt.title('KNN Cluster Data Points')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            st.pyplot(plt)
        else:
            st.markdown('<p style="color:red; font-size: 20px;">Tidak ada tweet yang mengandung keyword tersebut.</p>', unsafe_allow_html=True)
