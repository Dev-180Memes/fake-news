

import numpy as np
import pandas as pd

import nltk
import re
import string
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel("covid_only.xlsx")

data1 = data.sample(frac=1)

data1 = data1.dropna()

clean_news = data1.copy()

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

clean_news['content']=clean_news['content'].apply(lambda x:review_cleaning(x))

clean_news['title']=clean_news['title'].apply(lambda x:review_cleaning(x))

import nltk

nltk.download('stopwords')

stop = stopwords.words('english')

clean_news['content'] = clean_news['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

clean_news['title'] = clean_news['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

news_features=clean_news.copy()
news_features=news_features[['content']].reset_index(drop=True)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

corpus = []
for i in range(0, len(news_features)):
    news = re.sub('[^a-zA-Z]', ' ', news_features['content'][i])
    news= news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if not word in stop_words]
    news = ' '.join(news)
    corpus.append(news)
    
news_features.to_excel('transform.xlsx')

tfidf_vectorizer = TfidfVectorizer(max_features=30000,ngram_range=(2,2))
# TF-IDF feature matrix
X= tfidf_vectorizer.fit_transform(news_features['content'])

clean_news['Label'] = clean_news['Label'].replace({'fake': 0, 'real': 1})

y=clean_news['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.25, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

import pickle

pickle.dump(logreg, open('model.pkl','wb'))
