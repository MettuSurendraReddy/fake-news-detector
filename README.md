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