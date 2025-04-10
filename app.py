import os
import cv2
import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from torch import nn
import traceback

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Load models once at startup ===
FACE_MODEL = models.resnet18(weights=None)
FACE_MODEL.fc = nn.Linear(FACE_MODEL.fc.in_features, len(EMOTION_LABELS))
try:
    FACE_MODEL.load_state_dict(torch.load(os.path.join("models", "face_emotion_model.pth"), map_location='cpu'))
    FACE_MODEL.eval()
except Exception as e:
    print("❌ Error loading face model:", e)

VOICE_MODEL = nn.Sequential(
    nn.Linear(40, 64),
    nn.ReLU(),
    nn.Linear(64, len(EMOTION_LABELS))
)
try:
    VOICE_MODEL.load_state_dict(torch.load(os.path.join("models", "voice_emotion_model.pth"), map_location='cpu'))
    VOICE_MODEL.eval()
except Exception as e:
    print("❌ Error loading voice model:", e)


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "Server is running ✅"})


@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({"error": "No image file provided"}), 400

        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor_img = transform(img).unsqueeze(0)

        if tensor_img.shape != (1, 3, 224, 224):
            return jsonify({"error": "Incorrect image tensor shape"}), 400

        with torch.no_grad():
            outputs = FACE_MODEL(tensor_img)
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTION_LABELS[predicted.item()]

        return jsonify({"label": emotion})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Face analysis failed", "details": str(e)}), 500


@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    try:
        audio_file = request.files.get('audio')
        if audio_file is None:
            return jsonify({"error": "No audio file provided"}), 400

        try:
            y, sr = librosa.load(audio_file, sr=16000)
        except Exception:
            return jsonify({"error": "Invalid or unreadable audio file"}), 400

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        if tensor_mfcc.shape != (1, 40):
            return jsonify({"error": "Incorrect MFCC shape"}), 400

        with torch.no_grad():
            outputs = VOICE_MODEL(tensor_mfcc)
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTION_LABELS[predicted.item()]

        return jsonify({"label": emotion})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Voice analysis failed", "details": str(e)}), 500


@app.route("/stop_face", methods=["POST"])
def stop_face():
    return jsonify({"status": "Face detection stopped"})


@app.errorhandler(Exception)
def handle_global_exception(e):
    traceback.print_exc()
    return jsonify({"error": "Server error", "details": str(e)}), 500

@app.route("/analyze_combined", methods=["POST"])
def analyze_combined():
    try:
        image_file = request.files.get('image')
        audio_file = request.files.get('audio')

        if not image_file or not audio_file:
            return jsonify({"error": "Image and audio files are required"}), 400

        # ==== FACE ANALYSIS ====
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor_img = transform(img).unsqueeze(0)

        if tensor_img.shape != (1, 3, 224, 224):
            return jsonify({"error": "Incorrect image tensor shape"}), 400

        with torch.no_grad():
            face_output = FACE_MODEL(tensor_img)
            _, face_pred = torch.max(face_output, 1)
            face_emotion = EMOTION_LABELS[face_pred.item()]

        # ==== VOICE ANALYSIS ====
        try:
            y, sr = librosa.load(audio_file, sr=16000)
        except Exception:
            return jsonify({"error": "Invalid or unreadable audio file"}), 400

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        tensor_mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        if tensor_mfcc.shape != (1, 40):
            return jsonify({"error": "Incorrect MFCC shape"}), 400

        with torch.no_grad():
            voice_output = VOICE_MODEL(tensor_mfcc)
            _, voice_pred = torch.max(voice_output, 1)
            voice_emotion = EMOTION_LABELS[voice_pred.item()]

        # === COMBINED RESPONSE ===
        return jsonify({
            "face_emotion": face_emotion,
            "voice_emotion": voice_emotion
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Combined analysis failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

