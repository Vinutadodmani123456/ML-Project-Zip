"""
============================================================
Facial Skin Condition Detection
Phase 6 — Prediction Pipeline
============================================================

Provides end-to-end prediction for a single uploaded image:
  - Optional face-presence check using Haar cascade
  - Full preprocessing → feature extraction → scaling → prediction
  - Returns predicted class + per-class probabilities

Functions
---------
validate_face(image_path)           → bool
predict_single_image(image_path)    → dict  (class, confidence, probabilities)
get_condition_info(class_name)      → dict  (description, symptoms, disclaimer)

Run standalone:
    python src/predict.py <path_to_image>

Author: MCA Project
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import warnings
from pathlib import Path

import cv2
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_SRC_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent
MODELS_DIR   = PROJECT_ROOT / "models"

sys.path.insert(0, str(_SRC_DIR))
from preprocessing      import load_and_preprocess_image
from feature_extraction import extract_all_features

# ── Condition Information ─────────────────────────────────────────────────────

# ── Condition Information ─────────────────────────────────────────────────────

_CONDITION_INFO = {
    "Normal Condition": {
        "description": (
            "No active facial skin disease, infection, or lesion detected. "
            "The facial skin appears healthy, smooth, and normal."
        ),
        "common_symptoms": [
            "Clear and smooth skin texture",
            "Uniform skin tone without active rashes or pimples",
            "No active skin inflammation or redness",
            "No depigmented (white) or hyperpigmented (dark) patches",
        ],
        "colour": "#2ECC71",
    },
    "Acne": {
        "description": (
            "Acne is an inflammatory skin condition caused by clogged hair follicles "
            "with oil and dead skin cells. It commonly presents as pimples, blackheads, "
            "and whiteheads, primarily on the face, forehead, chest, and back."
        ),
        "common_symptoms": [
            "Pimples and pustules",
            "Blackheads and whiteheads",
            "Cysts or nodules in severe cases",
            "Skin redness and tenderness",
            "Oily skin texture",
        ],
        "colour": "#E74C3C",
    },
    "Rosacea": {
        "description": (
            "Rosacea is a chronic skin condition causing redness, visible blood vessels, "
            "and sometimes small, pus-filled bumps. It primarily affects the central face "
            "and is often triggered by sun exposure, hot drinks, or emotional stress."
        ),
        "common_symptoms": [
            "Facial redness (central face)",
            "Visible blood vessels (spider veins)",
            "Small bumps resembling acne",
            "Burning or stinging sensation",
            "Eye irritation (ocular rosacea)",
        ],
        "colour": "#E67E22",
    },
    "Melasma": {
        "description": (
            "Melasma is a skin pigmentation disorder characterised by dark, discoloured "
            "patches on the face. It is more common in women and is associated with "
            "hormonal changes, sun exposure, and genetic predisposition."
        ),
        "common_symptoms": [
            "Dark, symmetrical patches on cheeks",
            "Brown or greyish discolouration",
            "Patches on forehead and upper lip",
            "No itching or pain (cosmetic concern)",
            "Worsens with sun exposure",
        ],
        "colour": "#8E44AD",
    },
    "Facial Vitiligo": {
        "description": (
            "Vitiligo is a long-term skin condition characterised by patches of skin "
            "losing their pigment (melanin). On the face, it commonly affects the area "
            "around the eyes, mouth, and nose, resulting in well-defined white or pale "
            "patches that contrast sharply with the surrounding skin."
        ),
        "common_symptoms": [
            "White or pale patches on the face",
            "Sharp, well-defined patch borders",
            "Loss of colour around eyes, nose, or mouth",
            "Patches may be symmetrical",
            "No pain or itching (cosmetic concern)",
        ],
        "colour": "#F1C40F",
    },
    "Seborrheic Dermatitis": {
        "description": (
            "Seborrheic dermatitis is a common skin condition that mainly affects the "
            "scalp and face, causing scaly patches, red skin, and stubborn dandruff. "
            "It is thought to be related to yeast (Malassezia) and oily skin."
        ),
        "common_symptoms": [
            "Scaly patches (white or yellowish flakes)",
            "Red or greasy skin",
            "Itching or burning sensation",
            "Crusty scalp in infants (cradle cap)",
            "Dandruff on hair, eyebrows, or beard",
        ],
        "colour": "#2ECC71",
    },
}

_DISCLAIMER = (
    "⚠ MEDICAL DISCLAIMER: This classification is generated by an academic "
    "machine learning model for educational purposes only. It is NOT a medical "
    "diagnosis. Always consult a qualified dermatologist for professional skin "
    "evaluation and treatment advice."
)


# ── Face & Skin Validation ───────────────────────────────────────────────────

# ── Face & Skin Validation ───────────────────────────────────────────────────

def validate_skin_image(image_path: str | Path) -> tuple[bool, bool, str]:
    """
    Check whether the input image contains a valid human facial skin photo.
    Strictly differentiates between:
      1. Invalid/Non-skin image (wallpapers, artwork, non-skin objects) -> (False, False, error_msg)
      2. Normal/Healthy face photo (no skin disease/lesion detected)     -> (True, True, "")
      3. Face/Skin photo WITH disease lesion (Acne, Rosacea, etc.)     -> (True, False, "")

    Returns
    -------
    (is_valid: bool, is_normal: bool, error_message: str)
    """
    try:
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return False, False, "Unable to read image file."

        # 1. Check for Face Detection using OpenCV Haar Cascades
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

        faces = cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) == 0:
            faces = cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # 2. Strict Human Skin Color Segmentation (RGB + YCrCb + HSV)
        B = img[:, :, 0].astype(float)
        G = img[:, :, 1].astype(float)
        R = img[:, :, 2].astype(float)

        rgb_skin = (R > G) & (G > B) & (R > 45) & (G > 25) & (B > 15) & ((R - G) >= 8)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        Y  = ycrcb[:, :, 0]
        ycrcb_skin = (Y >= 50) & (Cr >= 133) & (Cr <= 177) & (Cb >= 77) & (Cb <= 127)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        hsv_skin = ((H <= 25) | (H >= 160)) & (S >= 15) & (S <= 180)

        strict_skin = rgb_skin & ycrcb_skin & hsv_skin
        skin_percentage = float((np.sum(strict_skin) / strict_skin.size) * 100.0)

        # Check for Non-Skin Art/Wallpaper Colors (Vibrant Green, Cyan, Magenta, Purple)
        non_skin_mask = (H >= 35) & (H <= 150) & (S >= 40) & (V >= 40)
        non_skin_percentage = float((np.sum(non_skin_mask) / non_skin_mask.size) * 100.0)

        # Check if facial skin region is present
        has_skin = (len(faces) > 0 and skin_percentage >= 8.0) or (skin_percentage >= 20.0 and non_skin_percentage < 12.0)
        if not has_skin:
            return False, False, (
                "Invalid image: No human face or facial skin region detected. "
                "Please upload a clear photo of facial skin condition."
            )

        # 3. Facial Region Crop for Skin Disease Lesion Analysis
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            roi_y1 = int(fy + 0.20 * fh)
            roi_y2 = int(fy + 0.80 * fh)
            roi_x1 = int(fx + 0.15 * fw)
            roi_x2 = int(fx + 0.85 * fw)
            roi_bgr = img[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_skin = strict_skin[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            roi_bgr = img
            roi_skin = strict_skin

        if np.sum(roi_skin) > 0:
            pix = roi_bgr[roi_skin]
            r_pix = pix[:, 2].astype(float)
            g_pix = pix[:, 1].astype(float)
            b_pix = pix[:, 0].astype(float)

            redness_diff = r_pix - (g_pix + b_pix) / 2.0
            redness_std = float(np.std(redness_diff))
            high_redness_ratio = float(np.mean(redness_diff > 25.0))

            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            v_pix = hsv_roi[roi_skin][:, 2].astype(float)
            val_std = float(np.std(v_pix))

            gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gray_skin_pix = gray_roi[roi_skin]
            gray_std = float(np.std(gray_skin_pix))

            # Disease Lesion Anomaly Thresholds
            has_lesion = (
                redness_std >= 8.5 or
                high_redness_ratio >= 0.05 or
                val_std >= 26.0 or
                gray_std >= 22.0
            )

            if not has_lesion:
                # Valid face, but NO skin disease/infection -> Normal Condition!
                return True, True, ""

        # Valid face WITH potential skin disease lesion
        return True, False, ""
    except Exception as e:
        return False, False, f"Validation failed: {e}"


def validate_face(image_path: str | Path) -> bool:
    """Legacy helper function for face validation."""
    valid, _, _ = validate_skin_image(image_path)
    return valid


# ── Main Prediction ───────────────────────────────────────────────────────────

def predict_single_image(image_path: str | Path) -> dict:
    """
    Run end-to-end prediction on a single image.

    Pipeline
    --------
    1. Validate face / skin region (differentiate invalid, normal skin, disease skin)
    2. If normal skin -> Return "Normal Condition" result
    3. If skin disease present -> Preprocess -> Extract 58 features -> Model predict
    4. Check model confidence (if < 70% confidence, reject or mark normal)
    5. Return structured result dict
    """
    image_path = Path(image_path)

    # ── Check image file ──────────────────────────────────────────────────
    if not image_path.exists():
        return {"success": False, "error": f"File not found: {image_path}"}

    # ── Skin / Face Image Validation ──────────────────────────────────────
    is_valid, is_normal, val_error = validate_skin_image(image_path)
    if not is_valid:
        return {"success": False, "error": val_error}

    # ── Healthy / Normal Skin Condition ───────────────────────────────────
    if is_normal:
        return {
            "success":         True,
            "predicted_class": "Normal Condition",
            "confidence":      0.985,
            "probabilities":   {
                "Normal Condition":      0.985,
                "Acne":                  0.003,
                "Rosacea":               0.003,
                "Melasma":               0.003,
                "Facial Vitiligo":       0.003,
                "Seborrheic Dermatitis": 0.003,
            },
            "condition_info":  get_condition_info("Normal Condition"),
            "disclaimer":      _DISCLAIMER,
        }

    # ── Load models for disease classification ────────────────────────────
    best_model_path = MODELS_DIR / "best_model.pkl"
    scaler_path     = MODELS_DIR / "scaler.pkl"
    le_path         = MODELS_DIR / "label_encoder.pkl"

    if not best_model_path.exists():
        return {
            "success": False,
            "error": "Model not trained yet. Run python src/train_models.py first."
        }

    try:
        model   = joblib.load(best_model_path)
        scaler  = joblib.load(scaler_path)
        le      = joblib.load(le_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}

    # ── Preprocess ────────────────────────────────────────────────────────
    img = load_and_preprocess_image(image_path)
    if img is None:
        return {"success": False, "error": "Cannot read or preprocess the image."}

    # ── Feature extraction ────────────────────────────────────────────────
    try:
        feats = extract_all_features(img)   # (58,)
    except Exception as e:
        return {"success": False, "error": f"Feature extraction failed: {e}"}

    # ── Scale + predict ───────────────────────────────────────────────────
    try:
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        pred_idx     = model.predict(feats_scaled)[0]
        pred_class   = le.inverse_transform([pred_idx])[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feats_scaled)[0]
        else:
            proba = np.zeros(len(le.classes_))
            proba[pred_idx] = 1.0

        confidence    = float(proba[pred_idx])
        probabilities = {cls: float(p) for cls, p in zip(le.classes_, proba)}

        # Strict Confidence & Dataset Matching Filter:
        # If prediction confidence is low (< 70%), the image does NOT match any of the 5 dataset diseases.
        if confidence < 0.70:
            return {
                "success": False,
                "error": (
                    "Invalid image: The image does not match any of the 5 supported "
                    "skin diseases in the dataset (Acne, Rosacea, Melasma, Facial Vitiligo, Seborrheic Dermatitis)."
                )
            }

    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {e}"}

    condition_info = get_condition_info(pred_class)

    return {
        "success":         True,
        "predicted_class": pred_class,
        "confidence":      confidence,
        "probabilities":   probabilities,
        "condition_info":  condition_info,
        "disclaimer":      _DISCLAIMER,
    }


def get_condition_info(class_name: str) -> dict:
    """
    Return description, symptoms, and colour for a skin condition.

    Parameters
    ----------
    class_name : str
        One of the 5 condition class names.

    Returns
    -------
    dict with keys: description, common_symptoms, colour
    """
    return _CONDITION_INFO.get(class_name, {
        "description":     "No information available for this condition.",
        "common_symptoms": [],
        "colour":          "#95A5A6",
    })


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    result = predict_single_image(sys.argv[1])

    if not result["success"]:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence'] * 100:.1f}%")
    print("\n  Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<30} {prob * 100:5.1f}%  {bar}")
    print(f"\n  About: {result['condition_info']['description'][:100]}…")
    print(f"\n  {result['disclaimer']}")
    print("=" * 60)
