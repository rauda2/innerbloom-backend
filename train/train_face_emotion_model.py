import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# === Face Emotion Dataset ===
class FaceEmotionDataset(Dataset):
    def __init__(self, image_folder):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

        for label in os.listdir(image_folder):
            label_path = os.path.join(image_folder, label)
            if os.path.isdir(label_path):
                for fname in os.listdir(label_path):
                    img_path = os.path.join(label_path, fname)
                    self.samples.append((img_path, self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label

# === Model Training ===
def train_face_model(data_dir, output_path, epochs=5, batch_size=32):
    dataset = FaceEmotionDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Face model saved to {output_path}")

# === Run Training ===
if __name__ == "__main__": 
    train_face_model("dataset/face_emotions/FER2013/train", "models/face_emotion_model.pth")

