# ✅ Full Flask App with Face, Voice, and Text Emotion Detection
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Load sentiment classifier once
sentiment_classifier = pipeline("sentiment-analysis")

def analyze_emotion(text):
    result = sentiment_classifier(text)[0]
    label = result['label']
    if label == "POSITIVE":
        return "happy"
    elif label == "NEGATIVE":
        return "sad"
    else:
        return "neutral"

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "✅ Inner Bloom server is alive!"})

@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        return jsonify({"label": dominant_emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']
    try:
        # ✅ Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
            temp_3gp.write(audio_file.read())
            temp_3gp_path = temp_3gp.name

        print(f"📦 Saved .3gp to: {temp_3gp_path}")
        wav_path = temp_3gp_path.replace(".3gp", ".wav")

        # ✅ Convert to .wav using pydub
        AudioSegment.from_file(temp_3gp_path, format="3gp").export(wav_path, format="wav")
        print(f"✅ Conversion successful: {wav_path}")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

        print(f"📝 Transcript: {transcript}")
        label = analyze_emotion(transcript)

        os.remove(temp_3gp_path)
        os.remove(wav_path)

        return jsonify({"transcript": transcript, "label": label})

    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_combined", methods=["POST"])
def analyze_combined():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Both image and audio are required"}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']

    try:
        # === FACE ANALYSIS ===
        img_array = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        face_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        face_emotion = face_result[0]['dominant_emotion']

        # === VOICE ANALYSIS ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
            temp_3gp.write(audio_file.read())
            temp_3gp_path = temp_3gp.name

        wav_path = temp_3gp_path.replace(".3gp", ".wav")
        AudioSegment.from_file(temp_3gp_path, format="3gp").export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

        voice_emotion = analyze_emotion(transcript)

        os.remove(temp_3gp_path)
        os.remove(wav_path)

        return jsonify({
            "face_emotion": face_emotion,
            "voice_emotion": voice_emotion,
            "transcript": transcript
        })

    except Exception as e:
        print(f"❌ Combined analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)

