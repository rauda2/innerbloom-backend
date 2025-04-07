import os
import cv2
import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

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
        from torchvision import models, transforms
        from torch import nn

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
        model.load_state_dict(torch.load("models/face_emotion_model/model.pt", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100

        return jsonify({"label": EMOTION_LABELS[predicted], "score": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === VOICE EMOTION ANALYSIS ===
@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    try:
        audio_file = request.files['audio']
        filepath = "temp_audio.wav"
        audio_file.save(filepath)

        y, sr = librosa.load(filepath, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        class VoiceClassifier(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                return self.fc2(x)

        model = VoiceClassifier(40, 128, len(EMOTION_LABELS))
        model.load_state_dict(torch.load("models/voice_emotion_model/model.pt", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            output = model(features)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100

        return jsonify({"label": EMOTION_LABELS[predicted], "score": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === CHAT EMOTION ANALYSIS ===
@app.route('/analyze_chat_history', methods=['POST'])
def analyze_chat():
    try:
        data = request.json
        print("💬 Incoming Chat Data:", data)

        # Your logic here, for example:
        result = chat_pipeline(data['text'])  # make sure 'text' exists!

        # Build response
        label = result[0]['label']
        score = round(result[0]['score'] * 100)
        reason = f"Model predicted {label} with confidence {score}%"
        return jsonify({'label': label, 'score': score, 'reason': reason})

    except Exception as e:
        print("❌ Error in /analyze_chat_history:", str(e))
        return jsonify({"error": "Server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

