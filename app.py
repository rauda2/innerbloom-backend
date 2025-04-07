import os
import cv2
import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from torch import nn

app = Flask(__name__)
CORS(app)

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "Server is running ✅"})

# === FACE EMOTION ANALYSIS ===
@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    try:
        image_file = request.files['image']
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        tensor_img = transform(img).unsqueeze(0)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(EMOTION_LABELS))

        model_path = os.path.join("models", "face_emotion_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            outputs = model(tensor_img)
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTION_LABELS[predicted.item()]

        return jsonify({"label": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

