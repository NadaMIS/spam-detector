from flask import Flask, request, jsonify, send_from_directory
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os


# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__, static_folder='static')

# Define stopwords set
STOP_WORDS = set(stopwords.words('english'))

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters, numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# Load the model and vectorizer
try:
    model = pickle.load(open('nb_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Write the HTML file to match the design in the image
with open('static/index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detector</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #5C5FEF;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --card-bg: #ffffff;
            --main-bg: #5C5FEF;
            --border-radius: 12px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: var(--main-bg);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .card {
            background-color: var(--card-bg);
            width: 100%;
            max-width: 600px;
            border-radius: var(--border-radius);
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .shield-icon {
            color: var(--primary-color);
            font-size: 24px;
        }
        
        h1 {
            color: #333;
            font-size: 28px;
            font-weight: 600;
        }
        
        .theme-toggle {
            position: relative;
            width: 50px;
            height: 24px;
        }
        
        .theme-toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: #333;
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        
        .subtitle::after {
            content: '';
            display: block;
            width: 80px;
            height: 4px;
            background-color: var(--primary-color);
            margin: 15px auto 0;
            border-radius: 2px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            color: #444;
            font-weight: 500;
        }
        
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            resize: none;
            font-size: 16px;
            margin-bottom: 20px;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--primary-color);
        }
        
        .char-count {
            text-align: right;
            font-size: 14px;
            color: #999;
            margin-top: -15px;
            margin-bottom: 20px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        .btn {
            padding: 12px 20px;
            border-radius: var(--border-radius);
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.3s;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #4a4ddd;
        }
        
        .btn-secondary {
            background-color: #f1f1f1;
            color: #666;
        }
        
        .btn-secondary:hover {
            background-color: #e0e0e0;
        }
        
        .tips-container {
            background-color: #fff5e6;
            border-radius: var(--border-radius);
            padding: 20px;
        }
        
        .tips-title {
            color: #ff9900;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .tips-title i {
            color: #ff9900;
        }
        
        .tips-list {
            list-style-type: none;
        }
        
        .tips-list li {
            margin-bottom: 10px;
            display: flex;
            align-items: baseline;
            gap: 10px;
            color: #555;
        }
        
        .tips-list li:before {
            content: "•";
            color: #ff9900;
            font-weight: bold;
        }
        
        .results {
            margin-top: 30px;
            display: none;
        }
        
        .result-card {
            background-color: #f8f9fa;
            border-radius: var(--border-radius);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .result-title {
            font-size: 18px;
            margin-bottom: 15px;
            color: #333;
            font-weight: 600;
        }
        
        .status {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 15px;
        }
        
        .status-safe {
            background-color: rgba(40, 167, 69, 0.1);
            color: #28a745;
        }
        
        .status-spam {
            background-color: rgba(220, 53, 69, 0.1);
            color: #dc3545;
        }
        
        .confidence-meter {
            margin-top: 15px;
        }
        
        .meter-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .meter-title {
            font-weight: 500;
            color: #555;
        }
        
        .meter-value {
            font-weight: 600;
        }
        
        .meter-value.low {
            color: #28a745;
        }
        
        .meter-value.medium {
            color: #ffc107;
        }
        
        .meter-value.high {
            color: #dc3545;
        }
        
        .meter-bar {
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .meter-fill {
            height: 100%;
            background-color: #28a745;
            transition: width 0.5s ease;
        }
        
        .meter-fill.medium {
            background-color: #ffc107;
        }
        
        .meter-fill.high {
            background-color: #dc3545;
        }
        
        .indicators {
            margin-top: 20px;
        }
        
        .indicators-title {
            font-weight: 500;
            margin-bottom: 10px;
            color: #555;
        }
        
        .indicators-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .indicator-tag {
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            color: #555;
        }
        
        .spinner {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner i {
            font-size: 2rem;
            color: var(--primary-color);
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background-color: rgba(220, 53, 69, 0.1);
            color: #dc3545;
            padding: 12px;
            border-radius: var(--border-radius);
            margin-bottom: 20px;
            display: none;
        }
        
        @media (max-width: 576px) {
            .card {
                padding: 20px;
            }
            
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div class="logo">
                <h1>Spam Detector</h1>
            </div>
            <label class="theme-toggle">
                <input type="checkbox" id="themeToggle">
                <span class="slider"></span>
            </label>
        </div>
                
        <div id="errorMessage" class="error-message"></div>
        
        <label for="messageInput">Enter your message</label>
        <textarea id="messageInput" placeholder="Type or paste your message here..."></textarea>
        <div class="char-count"><span id="charCount">0</span>/2000</div>
        
        <div class="button-group">
            <button id="checkBtn" class="btn btn-primary">
                <i class="fas fa-search"></i> Check Message
            </button>
            <button id="clearBtn" class="btn btn-secondary">
                <i class="fas fa-redo"></i> Clear
            </button>
        </div>
        
        <div id="spinner" class="spinner">
            <i class="fas fa-circle-notch"></i>
        </div>
        
        <div id="results" class="results">
            <div class="result-card">
                <h3 class="result-title">Analysis Results</h3>
                <div id="statusBadge" class="status">Unknown</div>
                
                <div class="confidence-meter">
                    <div class="meter-label">
                        <span class="meter-title">Spam Probability</span>
                        <span id="confidenceValue" class="meter-value">0%</span>
                    </div>
                    <div class="meter-bar">
                        <div id="confidenceFill" class="meter-fill" style="width: 0%"></div>
                    </div>
                </div>
                
                <div id="indicatorsSection" class="indicators">
                    <h4 class="indicators-title">Spam indicators found:</h4>
                    <div id="indicatorsList" class="indicators-list">
                        <!-- Indicators will be added here -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="tips-container">
            <div class="tips-title">
                <i class="fas fa-lightbulb"></i> Tips  to avoid spam
            </div>
            <ul class="tips-list">
                <li>Avoid using ALL CAPS in your message</li>
                <li>Don't overuse exclamation marks!!!!!</li>
                <li>Be cautious with phrases like "Free", "Act Now", etc.</li>
                <li>Check for spelling and grammar errors</li>
            </ul>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const messageInput = document.getElementById('messageInput');
            const charCount = document.getElementById('charCount');
            const checkBtn = document.getElementById('checkBtn');
            const clearBtn = document.getElementById('clearBtn');
            const results = document.getElementById('results');
            const statusBadge = document.getElementById('statusBadge');
            const confidenceValue = document.getElementById('confidenceValue');
            const confidenceFill = document.getElementById('confidenceFill');
            const indicatorsSection = document.getElementById('indicatorsSection');
            const indicatorsList = document.getElementById('indicatorsList');
            const errorMessage = document.getElementById('errorMessage');
            const spinner = document.getElementById('spinner');
            const themeToggle = document.getElementById('themeToggle');
            
            // Character counter
            messageInput.addEventListener('input', function() {
                const count = this.value.length;
                charCount.textContent = count;
                
                if (count > 2000) {
                    this.value = this.value.substring(0, 2000);
                    charCount.textContent = 2000;
                }
            });
            
            // Theme toggle
            themeToggle.addEventListener('change', function() {
                document.body.classList.toggle('dark-mode');
            });
            
            // Clear button
            clearBtn.addEventListener('click', function() {
                messageInput.value = '';
                charCount.textContent = '0';
                results.style.display = 'none';
                errorMessage.style.display = 'none';
            });
            
            // Check message
            checkBtn.addEventListener('click', function() {
                const message = messageInput.value.trim();
                
                if (!message) {
                    errorMessage.textContent = 'Please enter a message to analyze';
                    errorMessage.style.display = 'block';
                    return;
                }
                
                // Hide error and results
                errorMessage.style.display = 'none';
                results.style.display = 'none';
                
                // Show spinner
                spinner.style.display = 'block';
                checkBtn.disabled = true;
                
                // Send request to server
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide spinner
                    spinner.style.display = 'none';
                    checkBtn.disabled = false;
                    
                    if (data.error) {
                        errorMessage.textContent = data.error;
                        errorMessage.style.display = 'block';
                        return;
                    }
                    
                    // Display results
                    displayResults(data);
                })
                .catch(error => {
                    // Hide spinner
                    spinner.style.display = 'none';
                    checkBtn.disabled = false;
                    
                    errorMessage.textContent = 'An error occurred. Please try again.';
                    errorMessage.style.display = 'block';
                    console.error('Error:', error);
                });
            });
            
            function displayResults(data) {
                // Show results section
                results.style.display = 'block';
                
                // Update status badge
                statusBadge.textContent = data.prediction === 'spam' ? 'Spam Detected' : 'Not Spam';
                statusBadge.className = data.prediction === 'spam' ? 'status status-spam' : 'status status-safe';
                
                // Update confidence meter
                const probability = data.spam_probability;
                confidenceValue.textContent = `${probability}%`;
                confidenceFill.style.width = `${probability}%`;
                
                if (probability < 30) {
                    confidenceValue.className = 'meter-value low';
                    confidenceFill.className = 'meter-fill';
                } else if (probability < 70) {
                    confidenceValue.className = 'meter-value medium';
                    confidenceFill.className = 'meter-fill medium';
                } else {
                    confidenceValue.className = 'meter-value high';
                    confidenceFill.className = 'meter-fill high';
                }
                
                // Show/hide indicators section
                if (data.spam_indicators && data.spam_indicators.length > 0) {
                    indicatorsSection.style.display = 'block';
                    indicatorsList.innerHTML = '';
                    
                    data.spam_indicators.forEach(indicator => {
                        const tag = document.createElement('span');
                        tag.className = 'indicator-tag';
                        tag.textContent = indicator;
                        indicatorsList.appendChild(tag);
                    });
                } else {
                    indicatorsSection.style.display = 'none';
                }
                
                // Scroll to results
                results.scrollIntoView({ behavior: 'smooth' });
            }
        });
    </script>
</body>
</html>""")

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Model not available. Please check server logs.'
        })
    
    # Get message from request
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({
            'error': 'No message provided'
        })
    
    # Preprocess the message
    preprocessed = preprocess_text(message)
    
    # Vectorize
    message_vector = vectorizer.transform([preprocessed])
    
    # Predict
    prediction = model.predict(message_vector)[0]
    
    # Get probability scores (for better UI feedback)
    proba = model.predict_proba(message_vector)[0]
    
    # Determine spam probability based on class ordering
    spam_idx = list(model.classes_).index('spam') if 'spam' in model.classes_ else 1
    spam_probability = proba[spam_idx] * 100
    
    # Find common spam indicators
    common_spam_words = [
        "free", "call", "text", "prize", "win", "claim", "urgent",
        "cash", "offer", "mobile", "service", "customer", "please",
        "contact", "msg", "reply", "stop", "send", "credit", "gift", "iphone", "phone", "money","congratulation" , "insurance", "congrats"
    ]
    
    found_indicators = [word for word in common_spam_words if word in preprocessed.split()]
    
    return jsonify({
        'message': message,
        'prediction': prediction,
        'spam_probability': round(spam_probability, 2),
        'spam_indicators': found_indicators
    })

if __name__ == '__main__':
    print("Starting the spam detection app...")
    print(f"Model loaded: {model_loaded}")
    
    # Start the Flask server
    app.run(debug=True)
