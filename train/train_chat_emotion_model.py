from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

# ✅ Step 1: Load and simplify GoEmotions
print("📥 Loading GoEmotions dataset...")
raw_dataset = load_dataset("go_emotions")
label_names = raw_dataset["train"].features["labels"].feature.names

# Map to 7 core emotions
EMOTION_MAP = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6
}

from collections import defaultdict
import random

MAX_PER_LABEL = 500
emotion_buckets = defaultdict(list)

for row in raw_dataset["train"]:
    label_ids = row["labels"]
    mapped = [label_names[i] for i in label_ids if label_names[i] in EMOTION_MAP]
    if len(mapped) == 1:
        emotion = mapped[0]
        if len(emotion_buckets[emotion]) < MAX_PER_LABEL:
            emotion_buckets[emotion].append((row["text"], EMOTION_MAP[emotion]))

texts, labels = [], []
for emotion, samples in emotion_buckets.items():
    for text, label in samples:
        texts.append(text)
        labels.append(label)

print(f"✅ Sampled {len(texts)} total (≈{MAX_PER_LABEL} per emotion)")

print(f"✅ Loaded {len(texts)} filtered samples")

# ✅ Step 2: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ✅ Step 3: Create PyTorch dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# ✅ Step 4: Prepare dataset
dataset = EmotionDataset(texts, labels, tokenizer)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# ✅ Step 5: Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)

# ✅ Step 6: Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ✅ Step 7: Train and save
trainer.train()
torch.save(model.state_dict(), "models/chat_emotion_model.pth")
print("✅ Chat emotion model saved to models/chat_emotion_model.pth")

