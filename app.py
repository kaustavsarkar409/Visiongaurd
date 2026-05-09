import sys
from pathlib import Path

from flask import Flask, render_template, Response, jsonify

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import AI pipeline
from webcam_detect import generate_frames, history

app = Flask(
    __name__,
    template_folder="templates",
    static_folder=str(PROJECT_ROOT / "static")
)

# Ensure evidence directory exists for screenshots
(PROJECT_ROOT / "static" / "evidence").mkdir(parents=True, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/get_history")
def get_history():

    return jsonify(history[::-1])

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True,
        threaded=True
    )