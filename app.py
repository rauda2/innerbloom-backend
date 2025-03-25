from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Inner Bloom backend is alive!"

@app.route('/ping')
def ping():
    return jsonify({"status": "pong 💓", "message": "Server is working!"})

# Optional: version info
@app.route('/version')
def version():
    return jsonify({
        "app": "Inner Bloom Backend",
        "status": "Running",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

