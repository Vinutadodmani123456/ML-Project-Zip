"""
============================================================
Facial Skin Condition Detection — Flask Web Application
============================================================

Routes
------
GET  /           → Render main UI (index.html)
POST /predict    → Accept uploaded image, return prediction JSON
GET  /health     → Health check endpoint

Run:
    cd app
    python app.py
Then open: http://localhost:5000

Author: MCA Project
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ── Path Setup ────────────────────────────────────────────────────────────────

APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

try:
    from predict import predict_single_image, validate_face
    _MODEL_AVAILABLE = True
except ImportError as e:
    _MODEL_AVAILABLE = False
    _IMPORT_ERROR    = str(e)

# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload
app.config["SECRET_KEY"]         = os.urandom(24)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_available": _MODEL_AVAILABLE})


@app.route("/predict", methods=["POST"])
def predict():
    # ── File validation ────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Please upload JPG, JPEG, PNG, or BMP."
        }), 400

    if not _MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": (
                "Model not available. Please run the training pipeline first:\n"
                "1. python src/feature_extraction.py\n"
                "2. python src/train_models.py"
            )
        }), 503

    # ── Save to temp file ──────────────────────────────────────────────────
    suffix   = "." + secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    tmp_path = Path(tempfile.gettempdir()) / f"skin_{uuid.uuid4().hex}{suffix}"

    try:
        file.save(str(tmp_path))
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save file: {e}"}), 500

    # ── Predict ────────────────────────────────────────────────────────────
    try:
        result = predict_single_image(tmp_path)
    except Exception as e:
        result = {"success": False, "error": str(e)}
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if not result["success"]:
        return jsonify(result), 422

    # ── Serialise and return ───────────────────────────────────────────────
    response = {
        "success":         True,
        "predicted_class": result["predicted_class"],
        "confidence":      round(result["confidence"] * 100, 1),
        "probabilities":   {
            cls: round(prob * 100, 1)
            for cls, prob in result["probabilities"].items()
        },
        "condition_info":  result["condition_info"],
        "disclaimer":      result["disclaimer"],
    }
    return jsonify(response), 200


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Facial Skin Condition Detection — Flask App")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
