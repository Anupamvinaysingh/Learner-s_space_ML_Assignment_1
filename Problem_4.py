import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

positive_reviews = [
    "Amazing movie!", "Loved the acting.", "Great direction and story.",
    "Fantastic experience.", "Brilliant performance.", "Heartwarming and beautiful.",
    "Incredible visuals.", "Touching story.", "Outstanding film.",
    "Highly recommend!", "Best film of the year.", "Well written and directed.",
    "Absolutely loved it.", "Perfect pacing.", "Wonderful characters.",
    "A masterpiece.", "Entertaining and fun.", "Very moving.",
    "Loved every minute.", "Emotionally powerful.",
    "Top-notch storytelling.", "It was a joy to watch.",
    "Just perfect.", "Really impressive work.", "Uplifting experience.",
    "I couldn't stop watching.", "It made me cry (in a good way).",
    "Pure cinematic joy.", "A must-watch.", "Excellent movie.",
    "Beautifully shot.", "A feel-good masterpiece.",
    "Totally worth it.", "Captivating from start to finish.",
    "Perfectly done.", "Superb cast.", "Truly unforgettable.",
    "So inspirational.", "I’ll watch it again.", "Just amazing.",
    "Smart and engaging.", "It exceeded my expectations.",
    "I was blown away.", "Simply beautiful.",
    "A story well told.", "Touching and genuine.",
    "Delightfully crafted.", "A gem of a movie.",
    "Fell in love with the story.", "Bravo to the filmmakers.", "A visual treat."
]

negative_reviews = [
    "Terrible movie.", "Boring and slow.", "Awful acting.",
    "Disappointing storyline.", "I hated it.", "Very predictable.",
    "Poorly made.", "Waste of time.", "Lacked emotion.",
    "Bad direction.", "Nothing interesting.", "Too cheesy.",
    "Fell asleep halfway.", "Worst film of the year.",
    "Extremely dull.", "Characters were flat.", "Couldn’t relate to the story.",
    "Horrible script.", "Just awful.", "So boring.",
    "Mediocre at best.", "I wanted to leave.",
    "What a mess.", "Completely unoriginal.", "Annoying soundtrack.",
    "It made no sense.", "Painful to watch.", "Worst experience.",
    "Unwatchable.", "Didn't enjoy it at all.",
    "Felt like a waste.", "Plot was all over the place.",
    "Cringe-worthy moments.", "Too long and boring.",
    "A total flop.", "Very disappointing.",
    "Predictable and stale.", "Dialogue was bad.",
    "Didn't live up to the hype.", "Uninteresting characters.",
    "One of the worst I've seen.", "Laughably bad.",
    "Uninspired and lazy.", "Missed the mark.",
    "Confusing and weird.", "Terribly executed.",
    "Nothing redeeming.", "Visually unappealing.",
    "Emotionally hollow.", "Forgettable movie."
]

all_reviews = positive_reviews + negative_reviews
all_sentiments = ['positive'] * 50 + ['negative'] * 50

data = list(zip(all_reviews, all_sentiments))
random.shuffle(data)

df = pd.DataFrame(data, columns=['Review', 'Sentiment'])

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

def predict_review_sentiment(model, vectorizer, review):
    vectorized_review = vectorizer.transform([review])
    prediction = model.predict(vectorized_review)
    return prediction[0]

user_review = input("Enter a movie review to predict sentiment: ")
predicted_sentiment = predict_review_sentiment(model, vectorizer, user_review)
print(f"Predicted Sentiment: {predicted_sentiment}")