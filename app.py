from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Inner Bloom backend is alive!"

@app.route('/ping')
def ping():
    return jsonify({"status": "pong 💓", "message": "Server is working!"})

@app.route('/version')
def version():
    return jsonify({
        "app": "Inner Bloom Backend",
        "status": "Running",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT for us
    app.run(host='0.0.0.0', port=port)

