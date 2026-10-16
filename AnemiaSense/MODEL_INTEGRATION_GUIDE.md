# AnemiaSense: Backend Model Integration Guide

This guide specifies how the **AnemiaSense Python Web Backend** integrates the trained deep learning model artifacts, validates eye image quality, reproduces the exact image preprocessing pipeline, executes multi-task inference, classifies WHO severity, and feeds results to the web application.

---

## 1. Directory Structure and Artifact Inventory

Place the exported artifacts inside your web application backend:

```
AnemiaSense/
├── model_artifacts/
│   ├── anemia_mobilenetv2_multitask.keras   # Trained dual-head model
│   ├── model_metadata.json                  # Model specs, architecture, metrics
│   ├── preprocessing_config.json            # Target size, normalization parameters
│   ├── severity_rules.json                  # WHO clinical cutoff rules
│   ├── evaluation_results.csv               # Internal test set benchmark
│   ├── training_history.csv                 # Complete training/val learning history
│   ├── train_metadata.csv                   # Patient-level training split
│   ├── validation_metadata.csv              # Patient-level validation split
│   ├── test_metadata.csv                    # Patient-level test split
│   ├── README.md                            # Artifacts documentation
│   ├── gradcam_visualizations/              # Visual attention heatmaps
│   └── AnemiaSense_Model_Package.zip        # Consolidated deployment archive
└── MODEL_INTEGRATION_GUIDE.md               # This integration reference
```

---

## 2. Image Preprocessing & Quality Assurance Service

To guarantee consistency between model training and live web application inference, the backend must execute the exact preprocessing steps defined in `preprocessing_config.json`:

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class AnemiaImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def check_quality(self, image_bgr: np.ndarray) -> dict:
        \"\"\"
        Evaluates optical quality: blur, mean luminance, and glare percentage.
        \"\"\"
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        glare_pct = float(np.mean(gray > 245) * 100)

        # Quality criteria thresholds
        blur_ok = blur_score >= 20.0
        brightness_ok = 20.0 <= brightness <= 235.0
        glare_ok = glare_pct <= 15.0
        passed = blur_ok and brightness_ok and glare_ok

        issues = []
        if not blur_ok:
            issues.append("Image is too blurry. Please hold camera steady.")
        if not brightness_ok:
            issues.append("Lighting too dark or overexposed. Ensure balanced illumination.")
        if not glare_ok:
            issues.append("Excessive glare or specular reflection detected.")

        return {
            "passed": passed,
            "blur_score": blur_score,
            "brightness": brightness,
            "glare_percentage": glare_pct,
            "issues": issues
        }

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        \"\"\"
        Isolates conjunctiva ROI, standardizes background to neutral black,
        pads to square to preserve aspect ratio, resizes to 224x224,
        and normalizes to [-1.0, 1.0] expected by MobileNetV2.
        \"\"\"
        im_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect background color by sampling image corners
        corners = np.vstack([
            image_bgr[:10, :10].reshape(-1, 3), image_bgr[:10, -10:].reshape(-1, 3),
            image_bgr[-10:, :10].reshape(-1, 3), image_bgr[-10:, -10:].reshape(-1, 3)
        ])
        is_white_bg = np.all(corners.mean(axis=0) > 220)

        mask = (image_bgr < 235).any(axis=-1) if is_white_bg else (image_bgr > 20).any(axis=-1)
        ys, xs = np.where(mask)

        if len(ys) == 0:
            crop = im_rgb
        else:
            y_min, y_max = max(0, ys.min() - 5), min(im_rgb.shape[0], ys.max() + 5)
            x_min, x_max = max(0, xs.min() - 5), min(im_rgb.shape[1], xs.max() + 5)
            
            # Neutralize background
            masked = np.zeros_like(im_rgb)
            masked[mask] = im_rgb[mask]
            crop = masked[y_min:y_max, x_min:x_max]

        # Aspect-ratio preserving square padding
        h, w, _ = crop.shape
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        padded[y_off:y_off+h, x_off:x_off+w] = crop

        resized = cv2.resize(padded, self.target_size, interpolation=cv2.INTER_AREA)
        # MobileNetV2 normalization: scale [0, 255] -> [-1.0, 1.0]
        tensor = (resized.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(tensor, axis=0)
```

---

## 3. Web Service Inference Pipeline

```python
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class AnemiaSenseService:
    def __init__(self, artifacts_dir: str = "AnemiaSense/model_artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        
        # Load trained multi-task model
        model_path = self.artifacts_dir / "anemia_mobilenetv2_multitask.keras"
        self.model = keras.models.load_model(str(model_path))
        
        # Load severity rules
        with open(self.artifacts_dir / "severity_rules.json") as f:
            self.rules = json.load(f)
            
        self.preprocessor = AnemiaImagePreprocessor()

    def predict(self, image_bytes: bytes, gender: str = "M", age: int = 30) -> dict:
        \"\"\"
        Runs the full non-invasive screening inference:
        1. Decodes and checks image quality.
        2. Applies standard ROI preprocessing.
        3. Predicts anemia probability and continuous Hb (g/dL).
        4. Derives demographic-aware WHO severity.
        5. Formulates clinical recommendations.
        \"\"\"
        # Decode byte buffer to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"status": "error", "message": "Failed to decode image"}

        # Step 1: Quality Check
        quality = self.preprocessor.check_quality(img_bgr)
        if not quality["passed"]:
            return {
                "status": "quality_failed",
                "message": "Image quality check failed. Please retake photo.",
                "quality_details": quality
            }

        # Step 2: Preprocess
        input_tensor = self.preprocessor.process(img_bgr)

        # Step 3: Multi-Task Model Inference
        predictions = self.model.predict(input_tensor, verbose=0)
        anemia_prob = float(predictions[0][0][0])
        hb_estimate = float(predictions[1][0][0])
        is_anemic = bool(anemia_prob >= 0.5)

        # Step 4: WHO Clinical Severity Classification
        gender_code = "adult_male" if gender.upper().startswith("M") else "adult_female_non_pregnant"
        threshold = self.rules["supported_demographics"][gender_code]["anemia_threshold_g_dl"]

        if hb_estimate >= threshold:
            severity = "Normal"
            action_code = "MAINTAIN_HEALTH"
            clinical_message = "Hemoglobin concentration appears within normal physiological limits."
        elif hb_estimate < 8.0:
            severity = "Severe"
            action_code = "URGENT_MEDICAL_EVALUATION"
            clinical_message = "Estimated hemoglobin indicates severe anemia. Immediate clinical evaluation and venous blood laboratory confirmation are strongly advised."
        elif hb_estimate < 11.0:
            severity = "Moderate"
            action_code = "PHYSICIAN_CONSULTATION"
            clinical_message = "Estimated hemoglobin indicates moderate anemia. Clinical consultation, complete blood count (CBC), and serum ferritin testing recommended."
        else:
            severity = "Mild"
            action_code = "DIETARY_INTERVENTION"
            clinical_message = "Estimated hemoglobin indicates mild anemia. Review dietary iron and vitamin C intake; consult healthcare provider."

        return {
            "status": "success",
            "anemia_detected": is_anemic,
            "anemia_probability_percent": round(anemia_prob * 100, 1),
            "estimated_hemoglobin_g_dl": round(hb_estimate, 1),
            "normal_threshold_g_dl": threshold,
            "severity_stage": severity,
            "action_code": action_code,
            "clinical_guidance": clinical_message,
            "quality_metrics": quality
        }
```

---

## 4. Food Recommendation Rule Engine Integration

> [!IMPORTANT]
> **Food recommendations must never be trained using eye images.** 
> Food guidance is decoupled into a dedicated rule-based nutrition knowledge engine that triggers only upon mild or normal screening results. Severe or moderate anemia cases mandate physician referral rather than dietary self-management.

```python
def get_nutrition_guidance(severity: str, dietary_preference: str = "all") -> dict:
    \"\"\"
    Knowledge-based nutrition advice engine decoupled from the vision model.
    \"\"\"
    if severity in ["Severe", "Moderate"]:
        return {
            "allowed_self_management": False,
            "advice": "Dietary modification alone is insufficient for moderate to severe anemia. Immediate medical diagnosis is required to identify underlying etiology (e.g. chronic blood loss, malabsorption, renal disease, hemoglobinopathy)."
        }
    
    # Mild / Normal nutritional advice
    iron_rich_foods = [
        {"item": "Spinach & dark leafy greens", "nutrient": "Non-heme Iron, Folate"},
        {"item": "Lentils, chickpeas & beans", "nutrient": "Iron, Protein, Fiber"},
        {"item": "Pomegranate, amla & citrus", "nutrient": "Vitamin C (enhances iron bioavailability)"},
        {"item": "Pumpkin seeds & sesame seeds", "nutrient": "Iron, Zinc, Magnesium"}
    ]
    if dietary_preference.lower() in ["non-veg", "omnivore"]:
        iron_rich_foods.append({"item": "Lean red meat & poultry", "nutrient": "Heme Iron (high bioavailability)"})

    return {
        "allowed_self_management": True,
        "advice": "Incorporate iron-rich foods combined with vitamin C sources to enhance absorption. Avoid drinking tea or coffee within 1 hour of meals as tannins inhibit iron absorption.",
        "recommended_foods": iron_rich_foods
    }
```

---

## 5. Summary of Clinical Governance

1. **Intended Use**: Screening and triage support, not definitive diagnosis.
2. **Confirmatory Testing**: Any screening positive (anemia detected or Hb below cutoff) should prompt standard confirmatory venipuncture CBC testing.
3. **Transparency**: The web application must display both estimated Hb and confidence metrics to prevent user misinterpretation.
