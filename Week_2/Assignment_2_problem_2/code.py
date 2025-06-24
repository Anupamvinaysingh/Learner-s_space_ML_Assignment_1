import pandas as pd
import numpy as np
import re
import nltk
import gensim.downloader as api
import string
import contractions

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv("Tweets.csv")[['airline_sentiment', 'text']]
df.dropna(inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

print("Loading Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")

def tweet_to_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

print("Preprocessing tweets...")
X_vectors = []
for tweet in df['text']:
    tokens = preprocess(tweet)
    vec = tweet_to_vector(tokens, w2v_model)
    X_vectors.append(vec)

X = np.array(X_vectors)
y = df['airline_sentiment'].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vec = tweet_to_vector(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vec)[0]
    return pred

example = "I love flying with Delta, great experience!"
print("Predicted sentiment:", predict_tweet_sentiment(model, w2v_model, example))
