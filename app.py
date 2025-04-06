import os
import torch
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline
from torchvision import transforms
from torchvision.models import resnet18
import torchaudio

# === Setup ===
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# === Emotion Labels ===
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Face Model Load ===
FACE_MODEL_PATH = "models/face_emotion_model/model.pt"

face_model = resnet18(pretrained=False)
face_model.fc = torch.nn.Linear(face_model.fc.in_features, len(EMOTION_LABELS))
face_model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=device))
face_model.eval()

face_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# === Voice Classifier Load ===
class VoiceClassifier(nn.Module):
    def __init__(self):
        super(VoiceClassifier, self).__init__()
        self.fc1 = nn.Linear(193, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Create and save the correct model structure
print("🎙️ Creating voice model with correct shape...")
voice_model = VoiceClassifier()

# Dummy save (no training)
os.makedirs("models/voice_emotion_model", exist_ok=True)
torch.save(voice_model.state_dict(), "models/voice_emotion_model/model.pt")
print("✅ Voice Emotion Model structure saved.")

# === Chat Classifier Load ===
chat_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
chat_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# === Routes ===

@app.route("/")
def index():
    return jsonify({"message": "Inner Bloom Emotion Detection API"})


@app.route("/analyze_face_frame", methods=["POST"])
def analyze_face():
    image_file = request.files["image"]
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = face_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = face_model(tensor)
        prediction = torch.argmax(output, dim=1).item()
        score = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    return jsonify({
        "label": EMOTION_LABELS[prediction],
        "score": round(score * 100, 2)
    })


@app.route("/analyze_voice_chunk", methods=["POST"])
def analyze_voice():
    audio_file = request.files["audio"]
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    feature_vector = np.hstack((
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1)
    ))

    feature_vector = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = voice_model(feature_vector)
        prediction = torch.argmax(output, dim=1).item()
        score = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    return jsonify({
        "label": EMOTION_LABELS[prediction],
        "score": round(score * 100, 2),
        "transcript": "voice emotion placeholder"
    })


@app.route("/analyze_chat_history", methods=["POST"])
def analyze_chat_history():
    data = request.get_json()
    chat_text = data.get("text", "")

    if not chat_text.strip():
        return jsonify({"error": "Empty chat text"}), 400

    # Step 1: Summarize
    summary_result = chat_summarizer(chat_text, max_length=100, min_length=20, do_sample=False)
    summary = summary_result[0]["summary_text"]

    # Step 2: Emotion
    emotion_scores = chat_classifier(summary)[0]
    emotion_scores.sort(key=lambda x: x["score"], reverse=True)
    top_emotion = emotion_scores[0]

    reason = f"The emotion '{top_emotion['label'].lower()}' was detected because the summary contains themes or language that reflect {top_emotion['label'].lower()} behavior or feelings."

    return jsonify({
        "summary": summary,
        "label": top_emotion["label"].lower(),
        "score": round(top_emotion["score"] * 100, 2),
        "reason": reason
    })


# === Run Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

