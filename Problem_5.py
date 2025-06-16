import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


random.seed(42)
np.random.seed(42)


good_feedback = [
    "Great product", "Very satisfied", "Works as expected", "Excellent quality", 
    "I love it", "Highly recommend", "Amazing value", "Fast shipping", 
    "Customer service was great", "Best purchase ever"
]
bad_feedback = [
    "Terrible quality", "Very disappointed", "Does not work", "Waste of money", 
    "I hate it", "Not recommended", "Too expensive", "Poor packaging", 
    "Slow delivery", "Worst experience"
]

texts = [random.choice(good_feedback) for _ in range(50)] + [random.choice(bad_feedback) for _ in range(50)]
labels = [1]*50 + [0]*50  # 1 = good, 0 = bad


combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)


vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')


X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)


X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["bad", "good"]))


def text_preprocess_vectorize(texts, vectorizer):
    """
    Takes a list of text samples and a fitted TfidfVectorizer,
    returns the vectorized feature matrix.
    """
    return vectorizer.transform(texts)

example_texts = ["The product is amazing", "It was a terrible experience"]
example_features = text_preprocess_vectorize(example_texts, vectorizer)         #This problem looked really complicated so has to use some AI.
print("Vectorized example texts shape:", example_features.shape)