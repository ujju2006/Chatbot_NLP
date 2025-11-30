import json
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import csv
import os

# Fix SSL issue
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')

# Load intents
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Prepare training data
tags = []
patterns = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X, tags)

# Get response
def chatbot_response(text):
    text_vector = vectorizer.transform([text])
    tag = clf.predict(text_vector)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Streamlit UI
st.title("Implementation of Chatbot using NLP")

user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", response)

    # Save chat history
    with open("chat_log.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, response, datetime.now()])
