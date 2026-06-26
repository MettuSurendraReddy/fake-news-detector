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