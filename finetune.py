import pandas as pd

# Download LIAR dataset from GitHub
print("Downloading LIAR dataset...")

train_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
test_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
valid_url = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"

# Load datasets
train_df = pd.read_csv(train_url, sep='\t', header=None)
test_df = pd.read_csv(test_url, sep='\t', header=None)
valid_df = pd.read_csv(valid_url, sep='\t', header=None)

# Add column names
columns = ['id', 'label', 'statement', 'subject', 'speaker', 
           'job', 'state', 'party', 'barely_true', 'false_count',
           'half_true', 'mostly_true', 'pants_fire', 'context']

train_df.columns = columns
test_df.columns = columns
valid_df.columns = columns

# Explore the data
print("\n--- Dataset Info ---")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"Validation samples: {len(valid_df)}")

print("\n--- Label Distribution (Training) ---")
print(train_df['label'].value_counts())

print("\n--- Sample Statements ---")
print(train_df[['label', 'statement']].head(5))

# Preprocess — simplify 6 labels into 2
print("\n--- Preprocessing Labels ---")

def simplify_label(label):
    if label in ['true', 'mostly-true']:
        return 'REAL'
    else:
        return 'FAKE'

train_df['binary_label'] = train_df['label'].apply(simplify_label)
test_df['binary_label'] = test_df['label'].apply(simplify_label)
valid_df['binary_label'] = valid_df['label'].apply(simplify_label)

# Keep only statement and binary label
train_clean = train_df[['statement', 'binary_label']].dropna()
test_clean = test_df[['statement', 'binary_label']].dropna()
valid_clean = valid_df[['statement', 'binary_label']].dropna()

print(f"Training samples after cleaning: {len(train_clean)}")
print(f"Testing samples after cleaning: {len(test_clean)}")
print(f"Validation samples after cleaning: {len(valid_clean)}")

print("\n--- Binary Label Distribution (Training) ---")
print(train_clean['binary_label'].value_counts())

print("\n--- Sample Clean Data ---")
print(train_clean.head(5))

# Fine-tune RoBERTa on LIAR dataset
print("\n--- Fine-tuning RoBERTa ---")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

# Use small subset for training (standard practice for quick experiments)
train_sample = train_clean.sample(1000, random_state=42)
valid_sample = valid_clean.sample(200, random_state=42)

print(f"Training on {len(train_sample)} samples")
print(f"Validating on {len(valid_sample)} samples")

# Label mapping
label2id = {'FAKE': 0, 'REAL': 1}
id2label = {0: 'FAKE', 1: 'REAL'}

# Load tokenizer
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create PyTorch dataset
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = [label2id[l] for l in labels.tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# Create datasets
train_dataset = NewsDataset(train_sample['statement'], train_sample['binary_label'], tokenizer)
valid_dataset = NewsDataset(valid_sample['statement'], valid_sample['binary_label'], tokenizer)

print(f"Dataset created successfully!")
print(f"Training batches: {len(train_dataset)}")
print(f"Validation batches: {len(valid_dataset)}")