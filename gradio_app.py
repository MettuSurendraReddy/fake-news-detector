import gradio as gr
from model import predict

def analyze_news(article):
    """
    Takes news article as input
    Returns prediction and confidence score
    """
    # Validate input
    if not article.strip():
        return "⚠️ Please paste a news article first."
    
    word_count = len(article.strip().split())
    
    if word_count < 5:
        return f"⚠️ Too short to analyze! Please enter at least 5 words. You entered {word_count} words."
    
    warning = ""
    if word_count < 20:
        warning = f"⚠️ Short input ({word_count} words) — results may be less accurate.\n\n"
    
    # Get prediction
    result = predict(article)
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Format result
    if prediction == 'FAKE':
        return f"{warning}🔴 Result: FAKE NEWS\n\nConfidence: {confidence}%"
    else:
        return f"{warning}🟢 Result: REAL NEWS\n\nConfidence: {confidence}%"

# Build Gradio interface
demo = gr.Interface(
    fn=analyze_news,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your news article here...",
        label="News Article"
    ),
    outputs=gr.Textbox(
        label="Analysis Result"
    ),
    title="🔍 Fake News Detector",
    description="Powered by RoBERTa — HuggingFace Transformers. Paste a news article to check if it is REAL or FAKE.",
    examples=[
        ["NASA scientists have confirmed the discovery of water ice on the surface of the Moon, which could support future human missions to establish a permanent base."],
        ["Scientists have discovered that eating chocolate every day makes you live 100 years longer according to a secret study hidden by governments worldwide."]
    ]
)

if __name__ == "__main__":
    demo.launch()