from flask import Flask, request, jsonify

# Create Flask app instance - THIS IS CRITICAL
app = Flask(__name__)

# Your other imports and code here...
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# Define preprocessing function
def preprocess_text(text):
    STOP_WORDS = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# Load model
try:
    model = pickle.load(open('nb_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Create a simple HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Detector</title>
</head>
<body>
    <h1>Spam Detector</h1>
    <form method="POST" action="/predict">
        <textarea name="message" rows="5" cols="50"></textarea><br>
        <input type="submit" value="Check Message">
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return html_template

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    
    if not message or not model_loaded:
        return "Error: No message or model not loaded"
    
    # Process the message
    preprocessed = preprocess_text(message)
    message_vector = vectorizer.transform([preprocessed])
    prediction = model.predict(message_vector)[0]
    
    return f"Result: {prediction}"

# Don't include if __name__ == '__main__' for production
