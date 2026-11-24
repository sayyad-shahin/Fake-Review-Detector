"""Web app for the fake review detector.   python app.py   then open http://127.0.0.1:5000"""
import json, os
from flask import Flask, jsonify, render_template, request
from fakereview import load_or_train, DEFAULT_MODEL

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # reject huge requests
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = load_or_train()
    return _detector


def read_metrics():
    path = os.path.join(os.path.dirname(DEFAULT_MODEL), "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


@app.get("/")
def index():
    return render_template("index.html", metrics=read_metrics())


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify(error="'text' must be a string."), 400
    try:
        return jsonify(get_detector().analyze(text))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.get("/api/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    get_detector()  # load (or train) the model before serving
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://127.0.0.1:{port} in your browser (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
