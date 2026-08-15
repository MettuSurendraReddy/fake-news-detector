from transformers import pipeline

print("Loading model... please wait")

classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

print("Model loaded successfully!")

def predict(text):
    """
    Input: news article text (string)
    Output: dictionary with prediction (FAKE/REAL) and confidence score
    """
    result = classifier(text, truncation=True, max_length=512)[0]
    label = result['label']
    confidence = round(result['score'] * 100, 2)
    return {
        "prediction": label,
        "confidence": confidence
    }