# ✅ Full Flask App for Face & Voice Emotion Detection (Inner Bloom)
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

# ✅ Load text emotion model (once)
sentiment_classifier = pipeline("sentiment-analysis")


def analyze_emotion(text):
    result = sentiment_classifier(text)[0]
    label = result["label"]
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
        emotion = result[0]['dominant_emotion']
        return jsonify({"label": emotion})
    except Exception as e:
        print(f"❌ Face analysis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']
    try:
        # ✅ Save .3gp temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
            temp_3gp.write(audio_file.read())
            temp_3gp_path = temp_3gp.name

        print(f"📦 Saved .3gp to: {temp_3gp_path}")

        # ✅ Convert to .wav
        wav_path = temp_3gp_path.replace(".3gp", ".wav")
        AudioSegment.from_file(temp_3gp_path, format="3gp").export(wav_path, format="wav")
        print(f"✅ Conversion successful: {wav_path}")

        # ✅ Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

        if not transcript.strip():
            raise Exception("Transcript is empty")

        print(f"📝 Transcript: {transcript}")
        label = analyze_emotion(transcript)

        # ✅ Clean up
        os.remove(temp_3gp_path)
        os.remove(wav_path)

        return jsonify({"transcript": transcript, "label": label})

    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return jsonify({"error": "Could not understand audio"}), 500
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return jsonify({"error": str(e)}), 500


# 🔥 Optional: Combined analysis (not yet used)
@app.route("/analyze_combined", methods=["POST"])
def analyze_combined():
    return jsonify({"error": "Combined analysis not implemented yet"}), 501


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)

