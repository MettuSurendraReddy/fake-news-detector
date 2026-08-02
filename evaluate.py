import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load test dataset
print("Loading test dataset...")
test_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
test_df = pd.read_csv(test_url, sep='\t', header=None)

columns = ['id', 'label', 'statement', 'subject', 'speaker',
           'job', 'state', 'party', 'barely_true', 'false_count',
           'half_true', 'mostly_true', 'pants_fire', 'context']
test_df.columns = columns

# Simplify labels
def simplify_label(label):
    if label in ['true', 'mostly-true']:
        return 'REAL'
    else:
        return 'FAKE'

test_df['binary_label'] = test_df['label'].apply(simplify_label)
test_clean = test_df[['statement', 'binary_label']].dropna()

print(f"Test samples: {len(test_clean)}")

# Load fine-tuned model
print("\nLoading fine-tuned model...")
tokenizer = RobertaTokenizer.from_pretrained('fine_tuned_model')
model = RobertaForSequenceClassification.from_pretrained('fine_tuned_model')
model.eval()

print("Model loaded successfully!")

# Run predictions on first 200 test samples
print("\nRunning predictions...")
test_sample = test_clean.head(200)

predictions = []
true_labels = []

for idx, row in test_sample.iterrows():
    inputs = tokenizer(
        row['statement'],
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append('REAL' if pred == 1 else 'FAKE')
        true_labels.append(row['binary_label'])

print(f"Predictions complete! Total: {len(predictions)}")
print(f"\nSample predictions:")
for i in range(5):
    print(f"True: {true_labels[i]} | Predicted: {predictions[i]}")