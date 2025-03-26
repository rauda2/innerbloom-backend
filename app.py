from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from pydub import AudioSegment
import tempfile

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Inner Bloom backend is alive!"

@app.route('/ping')
def ping():
    return jsonify({"status": "pong 💓", "message": "Server is working!"})


@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    try:
        from deepface import DeepFace  # Lazy import

        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image']
        image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

        result = DeepFace.analyze(img_path=image_np, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return jsonify({'label': emotion, 'score': 1.0})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    try:
        import speech_recognition as sr
        from transformers import pipeline

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400

        audio = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            audio.save(temp_wav.name)
            wav_path = temp_wav.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "Could not understand audio"

        os.remove(wav_path)

        classifier = pipeline("sentiment-analysis")
        voice_emotion = classifier(transcript)[0]

        return jsonify({
            'label': voice_emotion['label'].lower(),
            'score': voice_emotion['score'],
            'transcript': transcript
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_combined', methods=['POST'])
def analyze_combined():
    result = {}

    # Analyze face
    if 'image' in request.files:
        try:
            from deepface import DeepFace
            image = request.files['image']
            image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            face_result = DeepFace.analyze(img_path=image_np, actions=['emotion'], enforce_detection=False)
            result['face_emotion'] = face_result[0]['dominant_emotion']
        except Exception as e:
            result['face_emotion'] = 'error: ' + str(e)

    # Analyze voice
    if 'audio' in request.files:
        try:
            import speech_recognition as sr
            from transformers import pipeline
            audio = request.files['audio']

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                audio.save(temp_wav.name)
                wav_path = temp_wav.name

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            try:
                transcript = recognizer.recognize_google(audio_data)
                result['transcript'] = transcript
                classifier = pipeline("sentiment-analysis")
                voice_result = classifier(transcript)[0]
                result['voice_emotion'] = voice_result['label'].lower()
            except sr.UnknownValueError:
                result['transcript'] = "Could not understand audio"
                result['voice_emotion'] = "unknown"

            os.remove(wav_path)

        except Exception as e:
            result['voice_emotion'] = 'error: ' + str(e)

    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

