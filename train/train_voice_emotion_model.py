import os
import torch
import numpy as np
import librosa
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# ==========================
# ✅ VOICE EMOTION TRAINING
# ==========================

class VoiceEmotionDataset(Dataset):
    def __init__(self, audio_folder, max_per_label=200):
        self.samples = []
        self.label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        self.buckets = {label: [] for label in self.label_map}

        for label in os.listdir(audio_folder):
            label_path = os.path.join(audio_folder, label)
            if os.path.isdir(label_path) and label in self.label_map:
                files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith((".wav", ".mp3"))]
                random.shuffle(files)
                self.buckets[label] = files[:max_per_label]

        for label, files in self.buckets.items():
            for path in files:
                self.samples.append((path, self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32), label

# === Simple FFN Model ===
class VoiceEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.net(x)

# === Training Function ===
def train_voice_model(data_dir, output_path, epochs=10, batch_size=32):
    dataset = VoiceEmotionDataset(data_dir, max_per_label=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VoiceEmotionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Voice model saved to {output_path}")

# === Run Training ===
if __name__ == "__main__":
    train_voice_model("dataset/voice_emotions/sorted", "models/voice_emotion_model.pth")

