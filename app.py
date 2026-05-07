from flask import Flask

# Create the Flask app
app = Flask(__name__)

# Home route — shows when someone visits the main page
@app.route('/')
def home():
    return "Fake News Detector is running!"

# Start the app
if __name__ == '__main__':
    app.run(debug=True)