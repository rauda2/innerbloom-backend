import os
import cv2
import numpy as np
import torch
import librosa
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from dotenv import load_dotenv
from torch import nn
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CHAT_LABELS = EMOTION_LABELS

# === Lazy-loaded models ===
_face_model = None
_voice_model = None
_chat_model = None
_chat_tokenizer = None

# === File download from Drive if not available ===
MODEL_DIR = "models"
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_emotion_model.pth")
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_emotion_model.pth")
CHAT_MODEL_PATH = os.path.join(MODEL_DIR, "chat_emotion_model.pth")

DRIVE_MODELS = {
    FACE_MODEL_PATH: "https://drive.google.com/uc?id=17laIY8UeiKVhrK_KFXTJb_NJnvBSi2Tj",
    VOICE_MODEL_PATH: "https://drive.google.com/uc?id=1YNX7PRjltdDZDnJaplNLZybx4B2L_WcL",
    CHAT_MODEL_PATH: "https://drive.google.com/uc?id=1RvS7M61kEdJbVOkJyFm6NCAPaH4NwGhK"
}

def ensure_model(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(url, path, quiet=False)

def get_face_model():
    global _face_model
    if _face_model is None:
        ensure_model(FACE_MODEL_PATH, DRIVE_MODELS[FACE_MODEL_PATH])
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(EMOTION_LABELS))
        model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location='cpu'))
        model.eval()
        _face_model = model
        print("✅ Face model loaded")
    return _face_model

def get_voice_model():
    global _voice_model
    if _voice_model is None:
        ensure_model(VOICE_MODEL_PATH, DRIVE_MODELS[VOICE_MODEL_PATH])
        class VoiceClassifier(nn.Module):
            def __init__(self, input_dim=40, hidden_dim=64, output_dim=len(EMOTION_LABELS)):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            def forward(self, x):
                return self.net(x)

        model = VoiceClassifier()
        model.load_state_dict(torch.load(VOICE_MODEL_PATH, map_location='cpu'))
        model.eval()
        _voice_model = model
        print("✅ Voice model loaded")
    return _voice_model

def get_chat_model():
    global _chat_model, _chat_tokenizer
    if _chat_model is None or _chat_tokenizer is None:
        ensure_model(CHAT_MODEL_PATH, DRIVE_MODELS[CHAT_MODEL_PATH])
        _chat_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(CHAT_LABELS))
        model.load_state_dict(torch.load(CHAT_MODEL_PATH, map_location="cpu"))
        model.eval()
        _chat_model = model
        print("✅ Chat model loaded")
    return _chat_model, _chat_tokenizer

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "Server is running ✅"})

@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor_img = transform(img).unsqueeze(0)

        outputs = get_face_model()(tensor_img)
        _, predicted = torch.max(outputs, 1)
        emotion = EMOTION_LABELS[predicted.item()]

        return jsonify({"label": emotion})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        outputs = get_voice_model()(tensor_mfcc)
        _, predicted = torch.max(outputs, 1)
        emotion = EMOTION_LABELS[predicted.item()]

        return jsonify({"label": emotion})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_combined", methods=["POST"])
def analyze_combined():
    try:
        image_file = request.files.get('image')
        audio_file = request.files.get('audio')
        if not image_file or not audio_file:
            return jsonify({"error": "Both image and audio are required"}), 400

        # Face
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor_img = transform(img).unsqueeze(0)
        face_pred = torch.argmax(get_face_model()(tensor_img)).item()

        # Voice
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        voice_pred = torch.argmax(get_voice_model()(tensor_mfcc)).item()

        return jsonify({
            "face_emotion": EMOTION_LABELS[face_pred],
            "voice_emotion": EMOTION_LABELS[voice_pred]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_chat_history", methods=["POST"])
def analyze_chat_history():
    try:
        data = request.get_json()
        text = data.get("text", "")
        model, tokenizer = get_chat_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(scores, 1).item()
        confidence = scores[0][predicted].item() * 100
        return jsonify({"label": CHAT_LABELS[predicted], "score": round(confidence, 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

