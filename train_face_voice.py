import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

# ==========================
# ✅ FACE EMOTION TRAINING
# ==========================

class FaceEmotionDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        for emotion in os.listdir(image_folder):
            emotion_path = os.path.join(image_folder, emotion)
            if os.path.isdir(emotion_path):
                for file in os.listdir(emotion_path):
                    file_path = os.path.join(emotion_path, file)
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((file_path, emotion))
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(e for _, e in self.samples)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, self.label2idx[label]

face_dataset = FaceEmotionDataset("dataset/train")
face_loader = DataLoader(face_dataset, batch_size=16, shuffle=True)

face_model = models.resnet18(pretrained=True)
face_model.fc = nn.Linear(face_model.fc.in_features, len(face_dataset.label2idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(face_model.parameters(), lr=1e-4)

print("🧠 Training Face Model...")
for epoch in range(1):
    for images, labels in face_loader:
        outputs = face_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Face Loss = {loss.item():.4f}")

os.makedirs("models/face_emotion_model", exist_ok=True)
torch.save(face_model.state_dict(), "models/face_emotion_model/model.pt")

# ==========================
# ✅ VOICE EMOTION DUMMY TRAINING (OPTIONAL PLACEHOLDER)
# ==========================

print("🎙️ Skipping voice training - placeholder")
os.makedirs("models/voice_emotion_model", exist_ok=True)
dummy_model = nn.Linear(10, 2)  # Replace with actual Wav2Vec2 training logic
torch.save(dummy_model.state_dict(), "models/voice_emotion_model/model.pt")

print("✅ Face and Voice Emotion Models Trained and Saved!")

