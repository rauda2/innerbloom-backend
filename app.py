import os
import io
import torch
import librosa
import numpy as np
from flask import Flask, request, jsonify
from transformers import pipeline
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms, models
from werkzeug.utils import secure_filename

# Init Flask app
app = Flask(__name__)

# Paths to your models
TEXT_MODEL_PATH = "models/text_emotion_model"
FACE_MODEL_PATH = "models/face_emotion_model/face_emotion.pt"

# Load chat (text) emotion classifier
chat_classifier = pipeline("text-classification", model=TEXT_MODEL_PATH, tokenizer=TEXT_MODEL_PATH)

# Load face emotion model
face_model = models.resnet18(pretrained=False)
face_model.fc = torch.nn.Linear(face_model.fc.in_features, 7)  # Assuming 7 emotions
face_model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=torch.device("cpu")))
face_model.eval()

# Define transforms for face
face_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ============================
# ✅ Analyze Face (image)
# ============================
# Make sure your transform matches what the model expects (3 channels)
face_transform = transforms.Compose([
    transforms.Resize((48, 48)),  # or whatever size your model expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/analyze_face_frame", methods=["POST"])
def analyze_face():
    try:
        # Load image and force RGB
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        # Transform and prepare input tensor
        tensor = face_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = face_model(tensor)
            prediction = torch.argmax(output, dim=1).item()
            score = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

        return jsonify({
            "label": EMOTION_LABELS[prediction],
            "score": round(score * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================
# ✅ Analyze Voice (audio chunk)
# ============================
@app.route("/analyze_voice_chunk", methods=["POST"])
def analyze_voice_chunk():
    audio_file = request.files["audio"]
    audio = AudioSegment.from_file(audio_file)
    wav_path = "temp_voice.wav"
    audio.export(wav_path, format="wav")

    y, sr = librosa.load(wav_path, sr=16000)
    transcript = " ".join(["voice emotion placeholder"])  # You can use real ASR later
    os.remove(wav_path)

    result = chat_classifier(transcript)
    if isinstance(result, list):
        result = result[0]
    return jsonify({
        "transcript": transcript,
        "label": result['label'].lower(),
        "score": round(result['score'] * 100, 2)
    })

# ============================
# ✅ Analyze Chat Emotion
# ============================
@app.route("/analyze_chat_history", methods=["POST"])
def analyze_chat():
    data = request.get_json()
    text = data["text"]

    result = chat_classifier(text)
    if isinstance(result, list):
        result = result[0]

    return jsonify({
        "text": text,
        "label": result['label'].lower(),
        "score": round(result['score'] * 100, 2)
    })

# ============================
# ✅ Ping Test
# ============================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Pong from Inner Bloom API"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

