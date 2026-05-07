from flask import Flask, render_template

# Create the Flask app
app = Flask(__name__)

# Home route — shows when someone visits the main page
@app.route('/')
def home():
    return render_template('index.html')

# Start the app
if __name__ == '__main__':
    app.run(debug=True)