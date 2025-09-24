import os
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load dataset and save raw
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
raw_path = os.path.join(RAW_DIR, "raw.csv")
df.to_csv(raw_path, index=False)
print(f"Saved raw data to {raw_path}")

# Very simple cleaning: drop rows with NA (none in this dataset)
df_clean = df.dropna()

# Train-test split
train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
train_path = os.path.join(PROCESSED_DIR, "train.csv")
test_path = os.path.join(PROCESSED_DIR, "test.csv")
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)
print(f"Saved processed train/test to {PROCESSED_DIR}")
