from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

app = Flask(__name__)
CORS(app)

# ✅ Load models once (Render memory-optimized)
print("🔄 Loading AI models...")
sentiment_model = pipeline("sentiment-analysis")
recognizer = sr.Recognizer()
print("✅ Models loaded!")

# ✅ Health check (important for Android + Render test)
@app.route('/')
def home():
    return jsonify({"message": "Server is working!", "status": "pong 💖"})

@app.route('/ping')
def ping():
    return jsonify({"message": "Server is working!", "status": "pong 💖"})

# ✅ Face emotion analysis
@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        print(f"✅ Received image: {image.filename}")
        image_bytes = image.read()

        result = DeepFace.analyze(img_path=image_bytes, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        return jsonify({
            "label": emotion,
            "score": round(confidence, 2)
        })

    except Exception as e:
        print("🔥 Face error:", e)
        return jsonify({"error": str(e)}), 500

# ✅ Voice emotion analysis
@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    # Save .3gp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".3gp") as temp_3gp:
        temp_3gp.write(audio_file.read())
        temp_3gp_path = temp_3gp.name

    print(f"📦 Saved .3gp to: {temp_3gp_path}")

    try:
        # Convert to .wav
        wav_path = temp_3gp_path.replace(".3gp", ".wav")
        AudioSegment.from_file(temp_3gp_path, format="3gp").export(wav_path, format="wav")
        print(f"✅ Conversion successful: {wav_path}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return jsonify({"error": "Conversion failed"}), 500
    finally:
        os.remove(temp_3gp_path)

    # Transcribe
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        transcript = "Could not transcribe"

    os.remove(wav_path)

    # Analyze emotion (dummy example)
    label = analyze_emotion(transcript)
    return jsonify({"label": label, "transcript": transcript})

# ✅ Combined face + voice emotion analysis
@app.route('/analyze_combined', methods=['POST'])
def analyze_combined():
    result = {}

    try:
        # Face detection
        if 'image' in request.files:
            image = request.files['image']
            print("✅ Received image for combined analysis:", image.filename)
            image_bytes = image.read()

            face_result = DeepFace.analyze(img_path=image_bytes, actions=['emotion'], enforce_detection=False)
            face_emotion = face_result[0]['dominant_emotion']
            face_confidence = face_result[0]['emotion'][face_emotion]
            result['face_emotion'] = face_emotion
            result['face_score'] = round(face_confidence, 2)

        # Voice detection
        if 'audio' in request.files:
            audio = request.files['audio']
            print("✅ Received audio for combined analysis:", audio.filename)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.3gp') as temp_3gp:
                temp_3gp.write(audio.read())
                temp_3gp_path = temp_3gp.name

            wav_path = temp_3gp_path.replace('.3gp', '.wav')
            AudioSegment.from_file(temp_3gp_path, format="3gp").export(wav_path, format="wav")
            os.remove(temp_3gp_path)

            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
            os.remove(wav_path)

            print("🗣️ Transcript:", transcript)
            result['transcript'] = transcript

            voice_result = sentiment_model(transcript)[0]
            voice_emotion = voice_result['label'].lower()
            voice_confidence = round(voice_result['score'] * 100, 2)
            result['voice_emotion'] = voice_emotion
            result['voice_score'] = voice_confidence

        return jsonify(result)

    except Exception as e:
        print("🔥 Combined error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

