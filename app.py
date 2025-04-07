import os
import cv2
import numpy as np
import torch
import librosa
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from torch import nn
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        model.load_state_dict(torch.load("models/face_emotion_model.pth", map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            outputs = model(tensor_img)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            predicted_label = EMOTION_LABELS[predicted_idx]

        return jsonify({"label": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === CHAT EMOTION ANALYSIS ===
@app.route("/analyze_chat_history", methods=["POST"])
def analyze_chat_history():
    try:
        data = request.json
        user_text = data.get("text")

        prompt = f"""
You are an empathetic emotion detection assistant. 
Your job is to:
1. Summarize the user's message in one line.
2. Determine the user's dominant emotion from the following 7 categories:
   [angry, disgust, fear, happy, neutral, sad, surprise].
3. Provide a reason for your emotion prediction.
4. Return a confidence score as a percentage (0-100%).

Output must be in JSON format like:
{{
  "label": "happy",
  "score": 94,
  "reason": "The message expresses excitement and positivity."
}}

User message: \"{user_text}\"
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You analyze emotions in human messages and reply in JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )

        content = response.choices[0].message["content"]

        # Evaluate the JSON response safely
        import json
        result = json.loads(content.strip())
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

