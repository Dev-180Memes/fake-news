import numpy as np
import pandas as pd
import re
import string
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=30000,ngram_range=(2,2))
data = pd.read_excel('transform.xlsx')
X = tfidf_vectorizer.fit_transform(data['content'])

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

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

@app.route('/predict',methods=['POST'])
def predict():
    input = request.get_json(force=True)
    textinput = [input['text']]
    newdata = pd.DataFrame(textinput, columns=['content'])
    newdata['content']= newdata['content'].apply(lambda x:review_cleaning(x))
    predData = tfidf_vectorizer.transform(newdata['content'])
    pred = model.predict(predData)
    if pred[0] == 1:
        return "Real"
    elif pred[0] == 0:
        return "Fake"

if __name__ == '__main__':
    app.run(port=5000, debug=True)
