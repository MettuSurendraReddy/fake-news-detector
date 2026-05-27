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
    
    result = predict(article)
    return render_template('index.html', 
                         prediction=result['prediction'],
                         confidence=result['confidence'])

if __name__ == '__main__':
    app.run(debug=True)