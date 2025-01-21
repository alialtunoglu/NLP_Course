# import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import libraries
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords 

# veri seti yukle
df = pd.read_csv("sms_spam.csv")

# veri temizleme hw
def clean_text(text):
    
    # buyuk kucuk harf cevrimi
    text = text.lower()
    
    # rakamlari temizleme
    text = re.sub(r"\d+", "", text)
    
    # ozel karakterlerin kaldirilmasi
    text = re.sub(r"[^\w\s]", "", text)
    
    # kisa kelimelerin temizlenmesi
    text = " ".join([word for word in text.split() if len(word) > 2])

    # stop words'lerin temizlenmesi
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    

    return text # temizlenmis text'i return et

# metinleri temizle
cleaned_doc = [clean_text(row) for row in df.text]
# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_doc)

# kelime kumesini incele
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # her kelimenin ortalama tf-idf degerleri

# tfidf skorlarini iceren bir df olustur
df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score": tfidf_score})

# skorlari sirala ve sonuclari incele
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))