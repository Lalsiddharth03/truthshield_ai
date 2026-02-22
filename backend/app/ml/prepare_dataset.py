import pandas as pd
import os

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "fake_or_real_news.csv")

print("Reading from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

df.columns = [col.lower() for col in df.columns]
df = df[['text', 'label']]

df['label'] = df['label'].apply(lambda x: 1 if x.lower() == 'real' else 0)

real = df[df['label'] == 1].sample(2000, random_state=42)
fake = df[df['label'] == 0].sample(2000, random_state=42)

balanced_df = pd.concat([real, fake])
balanced_df = balanced_df.sample(frac=1, random_state=42)

OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "news.csv")
balanced_df.to_csv(OUTPUT_PATH, index=False)

print("Balanced dataset created successfully!")
print("Saved to:", OUTPUT_PATH)
print("Total rows:", len(balanced_df))