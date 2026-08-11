from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route — receives article and returns prediction
@app.route('/predict', methods=['POST'])
def predict_news():
    article = request.form['article']
    
    # Check if user submitted empty text
    if not article.strip():
        return render_template('index.html',
                             error="Please paste a news article before clicking Check News.")

    word_count = len(article.strip().split())

    # Less than 5 words — block it
    if word_count < 5:
        return render_template('index.html',
                             error=f"Too short to analyze! Please enter at least 5 words. You entered {word_count} words.")

    # 5 to 20 words — allow but warn
    warning = None
    if word_count < 20:
        warning = f"Short input ({word_count} words) — results may be less accurate. For best results use a full article."

    try:
        result = predict(article)
        return render_template('index.html', 
                             prediction=result['prediction'],
                             confidence=result['confidence'],
                             warning=warning)
    except Exception as e:
        return render_template('index.html',
                             error="Something went wrong while analyzing the article. Please try again.")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)