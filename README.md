# 🔍 Fake News Detector

A web application that uses a fine-tuned RoBERTa transformer model to detect whether a news article is **real** or **fake**, along with a confidence score.

## 🎯 Project Overview

With the rise of misinformation online, this tool helps users quickly check the credibility of a news article using state-of-the-art NLP techniques. The app takes a news article as input and predicts whether it is **REAL** or **FAKE** news with a confidence percentage.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **NLP Model:** RoBERTa (HuggingFace Transformers)
- **Frontend:** HTML, CSS, JavaScript

## ✨ Features

- Paste any news article and get instant classification
- Confidence score for each prediction
- Color-coded results (green for real, red for fake)
- Input validation with helpful error messages
- Clean, responsive UI

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/MettuSurendraReddy/fake-news-detector.git
cd fake-news-detector
```

2. **Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install flask pandas scikit-learn transformers torch
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
```
http://127.0.0.1:5000
```

## 🧠 How It Works

1. User pastes a news article into the web interface
2. Flask receives the text via a POST request
3. Input validation checks word count before processing
4. The text is passed to a **RoBERTa transformer model** fine-tuned for fake news classification
5. The model analyzes language patterns, context, and writing style
6. A prediction (**REAL** or **FAKE**) is returned with a confidence score
7. Results are displayed with color coding — 🟢 Green for Real, 🔴 Red for Fake

## 🤖 Model Details

- **Model:** `hamzab/roberta-fake-news-classification`
- **Source:** HuggingFace Model Hub
- **Architecture:** RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task:** Text Classification (Fake vs Real News)
- **Max Input Length:** 512 tokens (longer articles are truncated)

## 📊 Input Validation

| Word Count | Behavior |
|------------|----------|
| Less than 5 words | Blocked — too short to analyze |
| 5 to 20 words | Allowed with accuracy warning |
| More than 20 words | Full prediction with high confidence |

## 🔮 Future Improvements

- [ ] Fine-tune RoBERTa on LIAR dataset for better accuracy
- [ ] Add support for multiple languages
- [ ] Display model confidence chart using Chart.js
- [ ] Add article URL input alongside text paste
- [ ] Deploy on HuggingFace Spaces for public access
- [ ] Add history of previously checked articles

## 👨‍💻 Author

**Mettu Surendra Reddy**
- 🎓 MSc Artificial Intelligence — Brandenburg University of Technology (BTU Cottbus)
- 💼 [LinkedIn](https://github.com/MettuSurendraReddy)
- 🐙 [GitHub](https://github.com/MettuSurendraReddy)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).