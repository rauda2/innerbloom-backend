import os
import shutil

# Set your root RAVDESS folder
RAVDESS_ROOT = "dataset/voice_emotions/RAVDESS"
OUTPUT_FOLDER = "dataset/voice_emotions/sorted"

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Create output folders
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for emotion in RAVDESS_EMOTION_MAP.values():
    os.makedirs(os.path.join(OUTPUT_FOLDER, emotion), exist_ok=True)

# Walk through Actor folders
for actor_folder in os.listdir(RAVDESS_ROOT):
    full_actor_path = os.path.join(RAVDESS_ROOT, actor_folder)
    if os.path.isdir(full_actor_path):
        for filename in os.listdir(full_actor_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                parts = filename.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
                    if emotion:
                        src = os.path.join(full_actor_path, filename)
                        dst = os.path.join(OUTPUT_FOLDER, emotion, filename)
                        shutil.copy2(src, dst)

print("✅ All RAVDESS audio files have been sorted into folders by emotion.")

