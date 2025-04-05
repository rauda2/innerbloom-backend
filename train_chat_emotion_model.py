from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# ✅ Step 1: Simple training data
texts = [
    "I’m feeling great today!", 
    "This is so frustrating and annoying.",
    "I’m very sad about everything right now.",
    "Life feels amazing!",
    "I can’t stop crying...",
    "Why is this so hard??"
]

labels = [2, 0, 1, 2, 1, 0]  # 0: anger, 1: sadness, 2: joy

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

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

# ✅ Step 4: Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# ✅ Step 5: Train
training_args = TrainingArguments(
    output_dir="./models/text_emotion_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    evaluation_strategy="no",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# ✅ Step 6: Save
model.save_pretrained("./models/text_emotion_model")
tokenizer.save_pretrained("./models/text_emotion_model")
print("✅ Custom mini chat emotion model trained & saved.")

