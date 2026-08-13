from flask import Flask, render_template, request
from model import predict
import os

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():
    """
    Receive news article from form submission
    Validate input and return prediction from BERT model
    """
    article = request.form['article']

    # Check if input is empty
    if not article.strip():
        return render_template('index.html',
                             error="Please paste a news article before clicking Check News.")

    word_count = len(article.strip().split())

    # Block very short inputs
    if word_count < 5:
        return render_template('index.html',
                             error=f"Too short to analyze! Please enter at least 5 words. You entered {word_count} words.")

    # Warn for short inputs but still process
    warning = None
    if word_count < 20:
        warning = f"Short input ({word_count} words) — results may be less accurate. For best results use a full article."

    try:
        # Run prediction using RoBERTa model
        result = predict(article)
        return render_template('index.html',
                             prediction=result['prediction'],
                             confidence=result['confidence'],
                             warning=warning)
    except Exception as e:
        # Handle any model errors gracefully
        return render_template('index.html',
                             error="Something went wrong while analyzing the article. Please try again.")

if __name__ == '__main__':
    # Use PORT environment variable for cloud deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)