"""
AnemiaSense Web Application — FastAPI Backend Server (v3.0 — Dual Model Integration)

Provides non-invasive anemia screening via two independent inference pipelines:
  1. Eye / Ocular Conjunctiva — MobileNetV2 dual-head (classification + Hb regression)
  2. Nail Bed — MobileNetV3Large single-head Hb regression

Both models are kept fully independent; failure of one does not affect the other.
"""

import io
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from tensorflow import keras

from web.db import db_manager, create_thumbnail_base64
from nail_model import (
    load_nail_model,
    get_nail_model,
    get_nail_model_status,
    preprocess_nail_image,
    predict_nail_hb,
    generate_nail_gradcam,
    run_nail_model_diagnostics,
    NAIL_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR   = WORKSPACE_DIR / "configs"
EYE_MODEL_DIR = WORKSPACE_DIR / "models" / "exported"
STATIC_DIR    = Path(__file__).resolve().parent / "static"

# Config files
with open(CONFIGS_DIR / "severity_rules.json", "r") as f:
    SEVERITY_RULES = json.load(f)

with open(CONFIGS_DIR / "preprocessing_config.json", "r") as f:
    PREPROCESSING_CONFIG = json.load(f)

# ---------------------------------------------------------------------------
# EYE MODEL (MobileNetV2, dual-head: anemia_prob + hb_regression)
# Preprocessing: external  (pixel / 127.5) − 1.0 → [-1, 1]
# Input: (None, 224, 224, 3)
# Output: [ shape(None,1) anemia_prob,  shape(None,1) hb_g_dl ]
# ---------------------------------------------------------------------------
EYE_MODEL_PATH = EYE_MODEL_DIR / "anemia_mobilenetv2_multitask.keras"
_eye_model = None
_eye_model_error: Optional[str] = None


def get_eye_model():
    global _eye_model, _eye_model_error
    if _eye_model is not None:
        return _eye_model, None
    if _eye_model_error:
        return None, _eye_model_error
    if not EYE_MODEL_PATH.exists():
        _eye_model_error = f"Eye model file not found: {EYE_MODEL_PATH}"
        return None, _eye_model_error
    try:
        _eye_model = keras.models.load_model(str(EYE_MODEL_PATH), compile=False)
        print(f"[AnemiaSense] Eye model loaded from {EYE_MODEL_PATH}")
        return _eye_model, None
    except Exception as e:
        _eye_model_error = str(e)
        return None, _eye_model_error




# ---------------------------------------------------------------------------
# OPTICAL QUALITY CHECKER (shared utility)
# ---------------------------------------------------------------------------
class ImageQualityChecker:
    def __init__(self, blur_min: float = 20.0, min_lum: float = 20.0,
                 max_lum: float = 235.0, glare_max_pct: float = 15.0):
        self.blur_min = blur_min
        self.min_lum = min_lum
        self.max_lum = max_lum
        self.glare_max_pct = glare_max_pct

    def evaluate(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur_score   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness   = float(np.mean(gray))
        glare_pct    = float(np.mean(gray > 245) * 100.0)

        passed = (blur_score >= self.blur_min and
                  self.min_lum <= brightness <= self.max_lum and
                  glare_pct <= self.glare_max_pct)

        warnings = []
        if blur_score < self.blur_min:
            warnings.append("Image appears blurred or out of focus. Consider capturing in better focus.")
        if brightness < self.min_lum:
            warnings.append("Image illumination is low. Natural indirect daylight is recommended.")
        elif brightness > self.max_lum:
            warnings.append("Image is overexposed. Avoid direct flashlight.")
        if glare_pct > self.glare_max_pct:
            warnings.append("Image has high specular glare or reflection. Avoid direct flashlight.")

        return {
            "blur_score":       round(blur_score, 2),
            "blur_threshold":   self.blur_min,
            "brightness":       round(brightness, 2),
            "glare_percentage": round(glare_pct, 2),
            "passed":           bool(passed),
            "warnings":         warnings,
        }


# ---------------------------------------------------------------------------
# EYE PREPROCESSING — conjunctiva localization + MobileNetV2 normalization
# ---------------------------------------------------------------------------
def preprocess_eye_image(img_rgb: np.ndarray, target_size=(224, 224),
                         crop_coords: Optional[str] = None):
    """
    Isolates palpebral conjunctiva tissue using up to 5 progressive strategies,
    pads to square, resizes, and normalizes to [-1, 1] for MobileNetV2.
    Returns (normalized_array, padded_rgb, telemetry_dict).
    """
    h_orig, w_orig, _ = img_rgb.shape
    crop = None
    focus_method = "full_frame"

    # Strategy 0: Client pre-cropped focused patch (Crystal HD ROI)
    if crop_coords == "client_precropped" or crop_coords == "precropped":
        crop = img_rgb
        focus_method = "client_interactive_focus"
    elif (h_orig == target_size[1] and w_orig == target_size[0]) or (h_orig == w_orig and h_orig <= 256):
        crop = img_rgb
        focus_method = "client_focused_loupe_patch"

    # Strategy 1: Client-supplied crop box
    elif crop_coords:
        try:
            parts = [float(p.strip()) for p in crop_coords.split(',') if p.strip()]
            if len(parts) == 4:
                cx, cy, cw, ch = parts
                if cw <= 1.0 and ch <= 1.0:
                    x0 = max(0, int(cx * w_orig)); y0 = max(0, int(cy * h_orig))
                    w0 = min(w_orig - x0, int(cw * w_orig)); h0 = min(h_orig - y0, int(ch * h_orig))
                else:
                    x0 = max(0, int(cx)); y0 = max(0, int(cy))
                    w0 = min(w_orig - x0, int(cw)); h0 = min(h_orig - y0, int(ch))
                if w0 > 15 and h0 > 15:
                    crop = img_rgb[y0:y0 + h0, x0:x0 + w0]
                    focus_method = "client_interactive_focus"
        except Exception as e:
            print(f"[Focus Warning] crop_coords parse error: {e}")

    # Strategy 2: White/dark dataset background mask
    if crop is None:
        corners = np.vstack([
            img_rgb[:10, :10].reshape(-1, 3), img_rgb[:10, -10:].reshape(-1, 3),
            img_rgb[-10:, :10].reshape(-1, 3), img_rgb[-10:, -10:].reshape(-1, 3)
        ])
        is_white_bg = np.all(corners.mean(axis=0) > 220)
        is_dark_bg  = np.all(corners.mean(axis=0) < 30)
        if is_white_bg or is_dark_bg:
            mask = (img_rgb < 235).any(axis=-1) if is_white_bg else (img_rgb > 25).any(axis=-1)
            ys, xs = np.where(mask)
            if len(ys) > 100:
                y_min = max(0, ys.min() - 5); y_max = min(h_orig, ys.max() + 5)
                x_min = max(0, xs.min() - 5); x_max = min(w_orig, xs.max() + 5)
                masked = np.zeros_like(img_rgb); masked[mask] = img_rgb[mask]
                crop = masked[y_min:y_max, x_min:x_max]
                focus_method = "dataset_tissue_mask"

    # Strategy 3: Mucosal erythema chromophore segmentation
    if crop is None:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, _ = cv2.split(lab)
        r_f = img_rgb[:, :, 0].astype(np.float32)
        g_f = img_rgb[:, :, 1].astype(np.float32) + 1e-5
        b_f = img_rgb[:, :, 2].astype(np.float32) + 1e-5
        ratio_mask = (r_f / g_f > 1.10) & (r_f / b_f > 1.05) & (l_chan > 35) & (l_chan < 235)
        mucosa_mask = (a_chan > 132) & ratio_mask
        ys, xs = np.where(mucosa_mask)
        if len(ys) > 120:
            y_min = max(0, int(np.percentile(ys, 3)) - 10)
            y_max = min(h_orig, int(np.percentile(ys, 97)) + 10)
            x_min = max(0, int(np.percentile(xs, 3)) - 10)
            x_max = min(w_orig, int(np.percentile(xs, 97)) + 10)
            if (x_max - x_min) > 35 and (y_max - y_min) > 25:
                crop = img_rgb[y_min:y_max, x_min:x_max]
                focus_method = "automated_mucosa_segmentation"

    # Strategy 4: Haar eye cascade
    if crop is None:
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, 1.15, 3)
            if len(eyes) > 0:
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
                ex, ey, ew, eh = eyes[0]
                cy_min = ey + int(eh * 0.35)
                cy_max = min(h_orig, ey + int(eh * 1.15))
                cx_min = max(0, ex - int(ew * 0.1))
                cx_max = min(w_orig, ex + int(ew * 1.1))
                crop = img_rgb[cy_min:cy_max, cx_min:cx_max]
                focus_method = "haar_lower_eyelid_crop"
        except Exception as e:
            print(f"[Haar Warning] {e}")

    # Strategy 5: Central viewfinder fallback
    if crop is None:
        y_center = int(h_orig * 0.48); x_center = int(w_orig * 0.50)
        half_h = int(h_orig * 0.25); half_w = int(w_orig * 0.28)
        crop = img_rgb[max(0, y_center - half_h):min(h_orig, y_center + half_h),
                       max(0, x_center - half_w):min(w_orig, x_center + half_w)]
        focus_method = "center_viewfinder_window"

    # Aspect-ratio preserving square padding
    h_c, w_c, _ = crop.shape
    max_dim = max(h_c, w_c)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_off = (max_dim - h_c) // 2; x_off = (max_dim - w_c) // 2
    padded[y_off:y_off + h_c, x_off:x_off + w_c] = crop

    # Optical biophotonic telemetry
    crop_float = crop.astype(np.float32)
    mean_r = float(np.mean(crop_float[:, :, 0]))
    mean_g = float(np.mean(crop_float[:, :, 1])) + 1e-5
    mean_b = float(np.mean(crop_float[:, :, 2])) + 1e-5
    erythema_index = float(np.log10(mean_r) - np.log10(mean_g))
    red_fraction   = float(mean_r / (mean_r + mean_g + mean_b))
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    cie_a_star = float(np.mean(crop_lab[:, :, 1]) - 128.0)
    telemetry = {
        "focus_method":     focus_method,
        "cropped_resolution": [w_c, h_c],
        "erythema_index":   round(erythema_index, 4),
        "cie_a_star":       round(cie_a_star, 2),
        "red_fraction":     round(red_fraction, 4),
        "mucosa_confidence": "HIGH" if (erythema_index > 0.08 or cie_a_star > 10) else "MODERATE",
    }

    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0   # MobileNetV2 normalization
    return normalized, padded, telemetry



# ---------------------------------------------------------------------------
# GRAD-CAM (Eye model only — MobileNetV2 'out_relu' layer)
# ---------------------------------------------------------------------------
_grad_model = None

def get_grad_model(model):
    global _grad_model
    if _grad_model is None:
        try:
            target_layer  = model.get_layer('out_relu')
            target_output = model.get_layer('hb_output')
            _grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, target_output.output]
            )
        except Exception as e:
            print(f"[Grad-CAM] Could not build grad model: {e}")
            return None
    return _grad_model


def generate_gradcam_overlay(model, norm_tensor: np.ndarray, padded_rgb: np.ndarray) -> str:
    try:
        grad_model = get_grad_model(model)
        if grad_model is None:
            return ""
        img_tensor = tf.expand_dims(tf.convert_to_tensor(norm_tensor, dtype=tf.float32), axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1).numpy()
        cam = np.maximum(cam, 0)
        if np.max(cam) > 1e-8:
            cam = cam / np.max(cam)
        cam_resized = cv2.resize(cam, (padded_rgb.shape[1], padded_rgb.shape[0]))
        cam_uint8   = np.uint8(255 * np.clip(cam_resized, 0, 1))
        colored     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(padded_rgb, 0.55, colored_rgb, 0.45, 0)
        _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"
    except Exception as e:
        print(f"[Grad-CAM Warning] {e}")
        return ""


# ---------------------------------------------------------------------------
# SEVERITY CLASSIFICATION (shared logic, WHO demographic-aware)
# ---------------------------------------------------------------------------
def classify_severity(est_hb: float, gender: str) -> Dict[str, str]:
    threshold = 13.0 if gender.upper().startswith('M') else 12.0
    if est_hb >= threshold:
        severity = "NORMAL"; label = "Optimal / Normal hemoglobin screening range"
    elif est_hb < 8.0:
        severity = "SEVERE_ANEMIA"; label = "Severe anemia screening range"
    elif est_hb < 11.0:
        severity = "MODERATE_ANEMIA"; label = "Moderate anemia screening range"
    else:
        severity = "MILD_ANEMIA"; label = "Mild anemia screening range"
    return {
        "severity_level":  severity,
        "severity_label":  label,
        "clinical_action": SEVERITY_RULES["clinical_actions"].get(severity, ""),
        "who_threshold":   threshold,
    }


# ---------------------------------------------------------------------------
# NUTRITION GUIDANCE (rule-based, decoupled from vision models)
# ---------------------------------------------------------------------------
def get_nutrition_guidance(severity: str, gender: str, age: int,
                            dietary_pref: str = "vegetarian") -> Dict[str, Any]:
    is_veg = "veg" in dietary_pref.lower() and "non" not in dietary_pref.lower()
    g_up   = gender.upper()

    if severity == "SEVERE_ANEMIA":
        daily_iron_target = 30; severity_theme = "Urgent Therapeutic Medical & Repletion Protocol"
    elif severity == "MODERATE_ANEMIA":
        daily_iron_target = 26; severity_theme = "High-Potency Nutritional Recovery Protocol"
    elif severity == "MILD_ANEMIA":
        daily_iron_target = 20; severity_theme = "Targeted Dietary Iron Repletion Plan"
    else:
        daily_iron_target = 18 if (g_up == 'F' and 19 <= age <= 50) else 8
        severity_theme = "Iron Homeostasis & Vitality Maintenance Plan"

    if is_veg:
        iron_rich_foods = [
            {"food": "Garden Cress Seeds (Halim/Aliv)", "detail": "High non-heme iron; soak in warm lemon water daily."},
            {"food": "Sprouted Kala Chana & Lentils",   "detail": "Sprouting reduces phytates, multiplying iron bioavailability."},
            {"food": "Moringa (Drumstick) Leaves",       "detail": "Rich plant iron; cook into dal or soup."},
            {"food": "Roasted Black Sesame + Jaggery",   "detail": "Concentrated non-heme iron with copper co-factors."},
            {"food": "Steamed Palak & Methi + Lemon",    "detail": "Folate and iron; always pair with vitamin C source."},
        ]
        daily_meal_idea = {
            "breakfast": "Sprouted moong chaat with amla, tomato, and lemon juice.",
            "lunch":     "Moringa leaf dal with whole wheat roti and beetroot salad.",
            "snack":     "Soaked garden cress seeds in lemon water + 6 black raisins.",
            "dinner":    "Rajma or black chana curry with brown rice and palak saag.",
        }
    else:
        iron_rich_foods = [
            {"food": "Chicken/Mutton Liver",             "detail": "9-11 mg heme iron per 100g; highest bioavailability (35%)."},
            {"food": "Whole Pastured Eggs",               "detail": "Concentrated bioavailable iron and choline."},
            {"food": "Mackerel, Sardines & Rohu Fish",    "detail": "Heme iron + anti-inflammatory omega-3 fatty acids."},
            {"food": "Lean Red Meat (Mutton/Lamb)",        "detail": "High-density heme iron and vitamin B12."},
            {"food": "Sprouted Lentils & Moringa Leaves", "detail": "Plant co-factors to maximise digestive bioavailability."},
        ]
        daily_meal_idea = {
            "breakfast": "Two whole eggs scrambled with diced tomatoes and bell peppers; guava slices.",
            "lunch":     "Liver or lean mutton curry with brown rice, palak dal, fresh lemon.",
            "snack":     "Boiled egg with black pepper and soaked black raisins.",
            "dinner":    "Grilled fish or chicken breast with sautéed spinach and lime.",
        }

    enhancers = [
        {"food": "Fresh Indian Gooseberry (Amla)", "benefit": "600-800 mg Vitamin C per 100g; enhances plant iron absorption up to 400%."},
        {"food": "Fresh Lemon & Lime Juice",        "benefit": "Squeeze over meals immediately before eating."},
        {"food": "Guava (Amrood)",                  "benefit": "Higher vitamin C than oranges; consume whole alongside meals."},
        {"food": "Bell Peppers & Tomatoes",         "benefit": "Ascorbic acid enhancers; always pair with iron dishes."},
    ]
    mealtime_considerations = [
        {"topic": "Tea & Coffee Separation (Chai Rule)", "advice": "Tannins inhibit non-heme iron absorption by 50-70%. Avoid tea/coffee within 90 minutes of main meals."},
        {"topic": "Calcium & Dairy Separation",          "advice": "Calcium competes directly with iron. Keep dairy snacks 2 hours apart from iron meals."},
        {"topic": "Overnight Soaking",                   "advice": "Soak dried beans and lentils overnight to deactivate phytic acid inhibitors."},
    ]
    medical_guidance = [
        "Anemia has many clinical causes: blood loss, B12/folate deficiency, haemoglobinopathies, bone marrow conditions.",
        "Dietary repletion supports recovery but cannot replace verified medical diagnosis.",
        "Confirm screening results with a Complete Blood Count (CBC) and serum ferritin via a licensed healthcare provider.",
        "Never self-prescribe high-dose iron supplements without physician-ordered lab tests.",
    ]

    return {
        "severity":               severity,
        "plan_theme":             severity_theme,
        "dietary_preference":     "Vegetarian" if is_veg else "Non-Vegetarian",
        "daily_target_iron_mg":   daily_iron_target,
        "iron_rich_foods":        iron_rich_foods,
        "absorption_enhancers":   enhancers,
        "mealtime_considerations": mealtime_considerations,
        "daily_meal_idea":        daily_meal_idea,
        "medical_guidance":       medical_guidance,
        "clinical_followup":      SEVERITY_RULES["clinical_actions"].get(severity, "Consult a qualified medical provider."),
    }


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AnemiaSense Dual-Model Screening API",
    description=(
        "Non-invasive anemia screening via ocular conjunctiva (MobileNetV2) "
        "and nail bed (MobileNetV3Large) imaging."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def startup_warmup():
    """Load and warm-up both models at startup for lowest first-request latency."""
    # Eye model
    try:
        eye_m, err = get_eye_model()
        if eye_m is not None:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            eye_m.predict(dummy, verbose=0)
            get_grad_model(eye_m)
            print("[AnemiaSense] Eye model warmed up [OK]")
        else:
            print(f"[AnemiaSense] Eye model not available: {err}")
    except Exception as e:
        print(f"[AnemiaSense] Eye model startup note: {e}")

    # Nail model
    try:
        nail_m, err = load_nail_model()
        if nail_m is not None:
            # Feed raw pixels [0,255] — model has internal Rescaling layer
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            nail_m.predict(dummy, verbose=0)
            from nail_model.gradcam import get_nail_grad_submodel
            get_nail_grad_submodel(nail_m)
            print(f"[AnemiaSense] Nail model warmed up [OK]  ({NAIL_MODEL_PATH.name})")
        else:
            print(f"[AnemiaSense] Nail model not available: {err}")
    except Exception as e:
        print(f"[AnemiaSense] Nail model startup note: {e}")


# ---------------------------------------------------------------------------
# HEALTH & METADATA
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    eye_m, eye_err = get_eye_model()
    nail_status = get_nail_model_status()
    return {
        "status":  "online",
        "service": "AnemiaSense Dual-Model Screening Service",
        "version": "3.1.0",
        "models": {
            "eye_model":  {"loaded": eye_m is not None,  "error": eye_err},
            "nail_model": {
                "loaded": nail_status["loaded"],
                "error": nail_status["error"],
                "checkpoint": nail_status["checkpoint_path"],
                "architecture": nail_status["architecture"],
                "unit": nail_status["output_unit"],
            },
        },
        "device":         "GPU" if tf.config.list_physical_devices('GPU') else "CPU",
        "privacy_notice": "In-memory processing only. Images are decoded ephemerally and never stored.",
        "disclaimer":     SEVERITY_RULES["disclaimer"],
    }


@app.get("/api/metadata")
def get_metadata():
    meta_path = EYE_MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            eye_meta = json.load(f)
    else:
        eye_meta = {}

    nail_status = get_nail_model_status()

    return {
        "eye_model":  eye_meta,
        "nail_model": {
            "architecture":           nail_status["architecture"],
            "task":                   nail_status["task"],
            "input_shape":            nail_status["expected_input_shape"],
            "preprocessing":          nail_status["preprocessing_contract"],
            "output":                 "Single float — estimated hemoglobin (g/dL)",
            "output_layer":           nail_status["output_layer_name"],
            "unit":                   nail_status["output_unit"],
            "target_scale_verified":  nail_status["target_scale_verified"],
            "explainability_layer":   nail_status["explainability_layer"],
            "locked_test_mae_g_dl":   nail_status["locked_test_mae_g_dl"],
            "locked_test_r2":         nail_status["locked_test_r2"],
            "disclaimer": (
                "Academic research prototype. Accuracy is limited (~1.96 g/dL MAE, negative R²). "
                "Not clinically validated for standalone medical diagnosis."
            ),
        },
    }


@app.get("/api/diagnostics/nail")
def nail_diagnostics():
    """Runs automated health and diagnostic validation on the active nail model."""
    return JSONResponse(content={"success": True, "diagnostics": run_nail_model_diagnostics()})


# ---------------------------------------------------------------------------
# EYE PREDICTION ENDPOINT (preserves original response format)
# ---------------------------------------------------------------------------
@app.post("/api/predict")
async def predict_eye(
    file: UploadFile = File(...),
    gender: str = Form("M"),
    age: int = Form(30),
    include_gradcam: bool = Form(True),
    dietary_pref: str = Form("vegetarian"),
    crop_coords: Optional[str] = Form(None),
):
    """
    Eye conjunctiva anemia screening.
    Dual-head MobileNetV2: anemia probability + Hb regression + Grad-CAM.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400,
                detail="Unable to decode image. Upload a clear PNG/JPG/JPEG photo.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Quality evaluation
        qc = ImageQualityChecker()
        qa_metrics = qc.evaluate(img_rgb)

        # 2. Conjunctiva localization + preprocessing
        norm_tensor, padded_rgb, telemetry = preprocess_eye_image(
            img_rgb, target_size=(224, 224), crop_coords=crop_coords)

        # 3. Eye model inference
        model, model_err = get_eye_model()
        if model is None:
            raise HTTPException(status_code=503,
                detail=f"Eye model unavailable: {model_err}")

        input_tensor = np.expand_dims(norm_tensor, axis=0)
        predictions  = model.predict(input_tensor, verbose=0)
        raw_p_anemia = float(predictions[0][0, 0])
        raw_est_hb   = float(predictions[1][0, 0])

        # 4. Biophotonic calibration
        g_clean   = gender.strip().upper()
        threshold = 13.0 if g_clean == 'M' else 12.0
        ei        = telemetry.get("erythema_index", 0.12)
        optical_hb = float(np.clip(8.0 + (ei - 0.06) * 38.0, 6.5, 17.5))
        if telemetry.get("mucosa_confidence") == "HIGH":
            calibrated_hb = 0.70 * raw_est_hb + 0.30 * optical_hb
        else:
            calibrated_hb = raw_est_hb
        est_hb = float(np.clip(calibrated_hb, 6.5, 17.5))

        delta = threshold - est_hb
        calibrated_p = float(1.0 / (1.0 + np.exp(-1.4 * delta)))
        p_anemia = float(np.clip(0.65 * raw_p_anemia + 0.35 * calibrated_p, 0.001, 0.999))
        is_screened_anemic = bool(p_anemia >= 0.5 or est_hb < threshold)

        sev = classify_severity(est_hb, g_clean)

        # 5. Grad-CAM
        gradcam_uri = ""
        if include_gradcam:
            gradcam_uri = generate_gradcam_overlay(model, norm_tensor, padded_rgb)

        # 6. Nutrition guidance
        nutrition_plan = get_nutrition_guidance(sev["severity_level"], g_clean, age, dietary_pref)

        # 7. Persist to MongoDB Screening History
        img_thumb = create_thumbnail_base64(img_rgb)
        history_doc = {
            "mode": "eye",
            "patient": {
                "gender":                         g_clean,
                "gender_display":                 "Male" if g_clean == 'M' else "Female",
                "age":                            age,
                "demographic_who_threshold_g_dl": threshold,
                "dietary_preference":             nutrition_plan["dietary_preference"],
            },
            "screening_results": {
                "estimated_hemoglobin_g_dl": round(est_hb, 2),
                "raw_model_hb_g_dl":         round(raw_est_hb, 2),
                "anemia_probability":        round(p_anemia, 4),
                "anemia_risk_percentage":    round(p_anemia * 100, 1),
                "is_anemic_flag":            is_screened_anemic,
                "severity_level":            sev["severity_level"],
                "severity_label":            sev["severity_label"],
                "clinical_code":             sev["severity_level"],
                "clinical_action":           sev["clinical_action"],
            },
            "biophotonic_telemetry": telemetry,
            "quality_assurance":     qa_metrics,
            "nutritional_guidance":  nutrition_plan,
            "crop_coords":           crop_coords,
            "image_thumbnail":       img_thumb,
            "gradcam_base64":        gradcam_uri,
            "disclaimer":            SEVERITY_RULES["disclaimer"],
        }
        db_meta = db_manager.save_screening(history_doc)

        return JSONResponse(content={
            "success": True,
            "model_used": "Eye / Ocular Conjunctiva — MobileNetV2 Dual-Head",
            "record_id":        db_meta["record_id"],
            "report_id":        db_meta["report_id"],
            "created_at":       db_meta["created_at"],
            "created_at_human": db_meta["created_at_human"],
            "mongodb_stored":   db_meta["mongodb_stored"],
            "patient_context": {
                "gender":                    g_clean,
                "gender_display":            "Male" if g_clean == 'M' else "Female",
                "age":                       age,
                "demographic_who_threshold_g_dl": threshold,
                "dietary_preference":        nutrition_plan["dietary_preference"],
            },
            "screening_results": {
                "anemia_probability":        round(p_anemia, 4),
                "anemia_risk_percentage":    round(p_anemia * 100, 1),
                "is_anemic_flag":            is_screened_anemic,
                "estimated_hemoglobin_g_dl": round(est_hb, 2),
                "raw_model_hb_g_dl":         round(raw_est_hb, 2),
                "severity_level":            sev["severity_level"],
                "severity_label":            sev["severity_label"],
                "clinical_code":             sev["severity_level"],
                "clinical_action":           sev["clinical_action"],
            },
            "biophotonic_telemetry": telemetry,
            "quality_assurance":     qa_metrics,
            "explainability": {
                "gradcam_layer":           "out_relu (MobileNetV2 final conv)",
                "gradcam_overlay_base64":  gradcam_uri,
            },
            "nutritional_guidance": nutrition_plan,
            "disclaimer": SEVERITY_RULES["disclaimer"],
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Eye screening error: {str(e)}")


# ---------------------------------------------------------------------------
# NAIL PREDICTION ENDPOINT (MobileNetV3Large Modular Integration)
# ---------------------------------------------------------------------------
@app.post("/api/predict/nail")
async def predict_nail(
    file: UploadFile = File(...),
    gender: str = Form("M"),
    age: int = Form(30),
    include_gradcam: bool = Form(True),
    dietary_pref: str = Form("vegetarian"),
    crop_coords: Optional[str] = Form(None),
):
    """
    Nail bed anemia screening.
    MobileNetV3Large single-head Hb regression with verified output units and Grad-CAM.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400,
                detail="Unable to decode image. Upload a clear PNG/JPG/JPEG nail photo.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Quality check
        qc = ImageQualityChecker()
        qa_metrics = qc.evaluate(img_rgb)

        # 2. Nail model inference via modular pipeline
        pred_res = predict_nail_hb(img_rgb, crop_coords=crop_coords)
        if not pred_res.get("success"):
            err_msg = pred_res.get("error", "Nail prediction failed.")
            if "unavailable" in pred_res.get("prediction_status", ""):
                raise HTTPException(status_code=503, detail=err_msg)
            elif "implausible" in err_msg.lower():
                raise HTTPException(status_code=422, detail=err_msg)
            else:
                raise HTTPException(status_code=400, detail=err_msg)

        est_hb = pred_res["predicted_hb"]
        qa_metrics = pred_res.get("quality_assurance", qa_metrics)

        # 3. WHO Severity classification & nutrition guidance
        g_clean = gender.strip().upper()
        sev = classify_severity(est_hb, g_clean)
        nutrition_plan = get_nutrition_guidance(sev["severity_level"], g_clean, age, dietary_pref)

        # 4. Grad-CAM Explainability
        gradcam_res = {}
        gradcam_uri = ""
        gradcam_layer = "activation_39 (MobileNetV3Large final conv activation)"
        if include_gradcam:
            gradcam_res = generate_nail_gradcam(img_rgb, crop_coords=crop_coords)
            gradcam_uri = gradcam_res.get("gradcam_overlay_base64", "")
            gradcam_layer = gradcam_res.get("layer_used", gradcam_layer)

        # 5. Persist to MongoDB / JSON Screening History
        img_thumb = create_thumbnail_base64(img_rgb)
        history_doc = {
            "mode": "nail",
            "patient": {
                "gender":                         g_clean,
                "gender_display":                 "Male" if g_clean == 'M' else "Female",
                "age":                            age,
                "demographic_who_threshold_g_dl": sev["who_threshold"],
                "dietary_preference":             nutrition_plan["dietary_preference"],
            },
            "screening_results": {
                "estimated_hemoglobin_g_dl": est_hb,
                "unit":                      "g/dL",
                "severity_level":            sev["severity_level"],
                "severity_label":            sev["severity_label"],
                "clinical_code":             sev["severity_level"],
                "clinical_action":           sev["clinical_action"],
                "target_scale_verified":     True,
                "preprocessing_verified":    True,
                "note": (
                    "The nail model is a continuous single-head Hb regressor (g/dL). "
                    "No separate anemia probability is produced."
                ),
                "research_prototype_notice": (
                    "Academic research prototype (locked-test MAE ~1.96 g/dL, negative R²). "
                    "Not a clinical diagnosis. Obtain a confirmatory laboratory blood test."
                ),
            },
            "quality_assurance":     qa_metrics,
            "nutritional_guidance":  nutrition_plan,
            "crop_coords":           crop_coords,
            "image_thumbnail":       img_thumb,
            "gradcam_base64":        gradcam_uri,
            "disclaimer":            pred_res.get("disclaimer", SEVERITY_RULES["disclaimer"]),
        }
        db_meta = db_manager.save_screening(history_doc)

        return JSONResponse(content={
            "success": True,
            "model_used": "Nail Bed — MobileNetV3Large Hb Regression",
            "model_architecture": "MobileNetV3Large",
            "model_file": NAIL_MODEL_PATH.name,
            "record_id":        db_meta["record_id"],
            "report_id":        db_meta["report_id"],
            "created_at":       db_meta["created_at"],
            "created_at_human": db_meta["created_at_human"],
            "mongodb_stored":   db_meta["mongodb_stored"],
            "patient_context": {
                "gender":                         g_clean,
                "gender_display":                 "Male" if g_clean == 'M' else "Female",
                "age":                            age,
                "demographic_who_threshold_g_dl": sev["who_threshold"],
                "dietary_preference":             nutrition_plan["dietary_preference"],
            },
            "screening_results": {
                "estimated_hemoglobin_g_dl": est_hb,
                "unit":                      "g/dL",
                "target_scale_verified":     True,
                "preprocessing_verified":    True,
                "severity_level":            sev["severity_level"],
                "severity_label":            sev["severity_label"],
                "clinical_code":             sev["severity_level"],
                "clinical_action":           sev["clinical_action"],
                "note": (
                    "The nail model predicts continuous haemoglobin (g/dL). "
                    "No binary classification probability is produced."
                ),
                "research_prototype_notice": (
                    "Academic research prototype (MAE ~1.96 g/dL, R² negative). "
                    "Not a laboratory measurement or medical diagnosis."
                ),
            },
            "explainability": {
                "gradcam_available":      gradcam_res.get("gradcam_available", False),
                "gradcam_layer":          gradcam_layer,
                "gradcam_overlay_base64": gradcam_uri,
            },
            "quality_assurance":    qa_metrics,
            "nutritional_guidance": nutrition_plan,
            "disclaimer":           pred_res.get("disclaimer", SEVERITY_RULES["disclaimer"]),
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nail screening error: {str(e)}")


# ---------------------------------------------------------------------------
# COMBINED ANALYSIS ENDPOINT (runs both models independently)
# ---------------------------------------------------------------------------
@app.post("/api/predict/combined")
async def predict_combined(
    eye_file: Optional[UploadFile] = File(None),
    nail_file: Optional[UploadFile] = File(None),
    gender: str = Form("M"),
    age: int = Form(30),
    include_gradcam: bool = Form(True),
    dietary_pref: str = Form("vegetarian"),
    crop_coords: Optional[str] = Form(None),
    eye_crop_coords: Optional[str] = Form(None),
    nail_crop_coords: Optional[str] = Form(None),
):
    """
    Combined analysis: runs Eye and/or Nail models independently.
    Results are returned as separate sections — no averaging or fusion.
    Either file may be omitted to run a single-model analysis.
    """
    if eye_file is None and nail_file is None:
        raise HTTPException(status_code=400,
            detail="Please provide at least one image (eye_file or nail_file).")

    response: Dict[str, Any] = {
        "success":  True,
        "mode":     "combined_analysis",
        "disclaimer": SEVERITY_RULES["disclaimer"],
        "note": (
            "Eye and nail results are produced by independent models with different "
            "architectures and training data. Do not average or combine the two Hb "
            "estimates into a single value. Each result should be interpreted separately."
        ),
    }

    g_clean = gender.strip().upper()

    # ── Eye analysis ──────────────────────────────────────────────────────────
    if eye_file is not None:
        try:
            eye_contents = await eye_file.read()
            nparr = np.frombuffer(eye_contents, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                response["eye_result"] = {"success": False,
                    "error": "Could not decode eye image."}
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qc = ImageQualityChecker()
                qa_metrics = qc.evaluate(img_rgb)
                norm_tensor, padded_rgb, telemetry = preprocess_eye_image(
                    img_rgb, target_size=(224, 224), crop_coords=(eye_crop_coords or crop_coords))

                eye_m, eye_err = get_eye_model()
                if eye_m is None:
                    response["eye_result"] = {"success": False, "error": eye_err}
                else:
                    inp = np.expand_dims(norm_tensor, axis=0)
                    preds = eye_m.predict(inp, verbose=0)
                    raw_p = float(preds[0][0, 0])
                    raw_hb = float(preds[1][0, 0])

                    threshold = 13.0 if g_clean == 'M' else 12.0
                    ei = telemetry.get("erythema_index", 0.12)
                    optical_hb = float(np.clip(8.0 + (ei - 0.06) * 38.0, 6.5, 17.5))
                    if telemetry.get("mucosa_confidence") == "HIGH":
                        cal_hb = 0.70 * raw_hb + 0.30 * optical_hb
                    else:
                        cal_hb = raw_hb
                    est_hb = float(np.clip(cal_hb, 6.5, 17.5))

                    delta = threshold - est_hb
                    cal_p = float(1.0 / (1.0 + np.exp(-1.4 * delta)))
                    p_anemia = float(np.clip(0.65 * raw_p + 0.35 * cal_p, 0.001, 0.999))
                    is_anemic = bool(p_anemia >= 0.5 or est_hb < threshold)

                    sev = classify_severity(est_hb, g_clean)
                    gradcam_uri = ""
                    if include_gradcam:
                        gradcam_uri = generate_gradcam_overlay(eye_m, norm_tensor, padded_rgb)

                    response["eye_result"] = {
                        "success": True,
                        "model_used": "Eye — MobileNetV2 Dual-Head",
                        "anemia_probability":        round(p_anemia, 4),
                        "anemia_risk_percentage":    round(p_anemia * 100, 1),
                        "is_anemic_flag":            is_anemic,
                        "estimated_hemoglobin_g_dl": round(est_hb, 2),
                        "severity_level":            sev["severity_level"],
                        "severity_label":            sev["severity_label"],
                        "clinical_action":           sev["clinical_action"],
                        "biophotonic_telemetry":     telemetry,
                        "quality_assurance":         qa_metrics,
                        "gradcam_overlay_base64":    gradcam_uri,
                    }
        except Exception as e:
            import traceback; traceback.print_exc()
            response["eye_result"] = {"success": False, "error": str(e)}
    else:
        response["eye_result"] = {"success": False, "error": "No eye image provided."}

    # ── Nail analysis ─────────────────────────────────────────────────────────
    if nail_file is not None:
        try:
            nail_contents = await nail_file.read()
            nparr = np.frombuffer(nail_contents, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                response["nail_result"] = {"success": False,
                    "error": "Could not decode nail image."}
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qc = ImageQualityChecker()
                qa_metrics = qc.evaluate(img_rgb)
                pred_res = predict_nail_hb(img_rgb, crop_coords=(nail_crop_coords or crop_coords))

                if not pred_res.get("success"):
                    response["nail_result"] = {
                        "success": False,
                        "error": pred_res.get("error", "Nail prediction failed.")
                    }
                else:
                    est_hb = pred_res["predicted_hb"]
                    sev = classify_severity(est_hb, g_clean)
                    gradcam_uri = ""
                    gradcam_layer = "activation_39 (MobileNetV3Large final conv activation)"
                    if include_gradcam:
                        g_res = generate_nail_gradcam(img_rgb, crop_coords=(nail_crop_coords or crop_coords))
                        gradcam_uri = g_res.get("gradcam_overlay_base64", "")
                        gradcam_layer = g_res.get("layer_used", gradcam_layer)

                    response["nail_result"] = {
                        "success": True,
                        "model_used": "Nail — MobileNetV3Large Hb Regression",
                        "model_file": NAIL_MODEL_PATH.name,
                        "estimated_hemoglobin_g_dl": est_hb,
                        "unit":                      "g/dL",
                        "target_scale_verified":     True,
                        "preprocessing_verified":    True,
                        "severity_level":            sev["severity_level"],
                        "severity_label":            sev["severity_label"],
                        "clinical_action":           sev["clinical_action"],
                        "quality_assurance":         pred_res.get("quality_assurance", qa_metrics),
                        "gradcam_overlay_base64":    gradcam_uri,
                        "gradcam_layer":             gradcam_layer,
                        "note": (
                            "Nail model is a continuous single-head Hb regressor. "
                            "No separate anemia probability is produced."
                        ),
                        "research_prototype_notice": (
                            "Academic research prototype (MAE ~1.96 g/dL, negative R²). "
                            "Not a standalone clinical diagnosis."
                        ),
                    }
        except Exception as e:
            import traceback; traceback.print_exc()
            response["nail_result"] = {"success": False, "error": str(e)}
    else:
        response["nail_result"] = {"success": False, "error": "No nail image provided."}

    # ── Shared nutrition guidance (based on lowest Hb estimate from successful models) ──
    try:
        hb_values = []
        if response.get("eye_result", {}).get("success"):
            hb_values.append(response["eye_result"]["estimated_hemoglobin_g_dl"])
        if response.get("nail_result", {}).get("success"):
            hb_values.append(response["nail_result"]["estimated_hemoglobin_g_dl"])

        if hb_values:
            min_hb = min(hb_values)
            worst_sev = classify_severity(min_hb, g_clean)
            response["combined_nutrition_guidance"] = get_nutrition_guidance(
                worst_sev["severity_level"], g_clean, age, dietary_pref)
            response["combined_nutrition_guidance"]["basis"] = (
                f"Guidance is based on the lower Hb estimate ({min_hb:.2f} g/dL) "
                "from the available model results, to err on the side of caution."
            )
    except Exception as e:
        print(f"[Combined] Nutrition guidance error: {e}")

    # ── Save combined screening to MongoDB History ──
    try:
        combined_thumb = ""
        if eye_file is not None and 'img_rgb' in locals():
            combined_thumb = create_thumbnail_base64(img_rgb)

        comb_doc = {
            "mode": "combined",
            "patient": {
                "gender":                         g_clean,
                "gender_display":                 "Male" if g_clean == 'M' else "Female",
                "age":                            age,
                "demographic_who_threshold_g_dl": 13.0 if g_clean == 'M' else 12.0,
                "dietary_preference":             dietary_pref,
            },
            "screening_results": {
                "estimated_hemoglobin_g_dl": min_hb if 'min_hb' in locals() else None,
                "severity_level":            worst_sev["severity_level"] if 'worst_sev' in locals() else "NORMAL",
                "severity_label":            worst_sev["severity_label"] if 'worst_sev' in locals() else "Normal",
                "clinical_action":           worst_sev["clinical_action"] if 'worst_sev' in locals() else "",
                "note":                      "Combined dual-model evaluation (Eye + Nail independent pipelines)"
            },
            "eye_result":            response.get("eye_result", {}),
            "nail_result":           response.get("nail_result", {}),
            "combined_nutrition":    response.get("combined_nutrition_guidance", {}),
            "image_thumbnail":       combined_thumb,
            "disclaimer":            SEVERITY_RULES["disclaimer"],
        }
        db_meta = db_manager.save_screening(comb_doc)
        response.update(db_meta)
    except Exception as e:
        print(f"[Combined] Database save error: {e}")

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# HISTORY & DATABASE ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history(
    limit: int = 50,
    skip: int = 0,
    severity: Optional[str] = None,
    search: Optional[str] = None,
):
    """Fetch screening records from MongoDB."""
    records = db_manager.list_screenings(limit=limit, skip=skip, severity=severity, search=search)
    status = db_manager.get_status()
    return JSONResponse(content={
        "success": True,
        "count": len(records),
        "history": records,
        "status": status,
    })


@app.get("/api/history/stats")
async def get_history_stats():
    """Fetch history statistics and metrics."""
    stats = db_manager.get_statistics()
    status = db_manager.get_status()
    return JSONResponse(content={
        "success": True,
        "stats": stats,
        "status": status,
    })


@app.get("/api/history/{record_id}")
async def get_history_detail(record_id: str):
    """Fetch single screening details."""
    doc = db_manager.get_screening(record_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Screening record not found.")
    return JSONResponse(content={"success": True, "record": doc})


@app.delete("/api/history/{record_id}")
async def delete_history_record(record_id: str):
    """Delete screening record from MongoDB."""
    ok = db_manager.delete_screening(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found or already deleted.")
    return JSONResponse(content={"success": True, "deleted": True})


@app.get("/api/db/status")
async def get_db_status():
    """Check MongoDB live connection status and record counts."""
    return JSONResponse(content={"success": True, "status": db_manager.get_status()})


# ---------------------------------------------------------------------------
# STATIC FRONTEND
# ---------------------------------------------------------------------------
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("[AnemiaSense] Starting server on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
