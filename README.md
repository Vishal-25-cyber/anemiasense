# AnemiaSense: Non-Invasive Anemia Screening Using Eye/Conjunctiva Images

AnemiaSense is a clinical deep learning research and screening pipeline that estimates hemoglobin (Hb) concentrations and predicts anemia risk non-invasively using digital images of the human ocular conjunctiva (palpebral and forniceal regions).

---

## 1. Clinical Motivation & Pathology

Anemia affects over **1.6 billion individuals** worldwide, with acute prevalence in pregnant women, young children, and resource-constrained populations. Traditional diagnostic workflows rely on venous blood draws (phlebotomy) and hematology analyzers, which present significant barriers in rural clinics and decentralized screening camps.

Ocular conjunctival tissue provides a direct optical window into microvascular hemoglobin concentration. The mucous membrane lining the eyelid and anterior sclera has high capillary density under a thin, translucent epithelial layer. As intravascular hemoglobin drops, tissue pallor increases proportionally.

---

## 2. Multi-Task Deep Learning Architecture

Rather than framing anemia screening purely as a binary classification problem or an unconstrained regression task, AnemiaSense utilizes a **Multi-Task Transfer Learning Network** powered by **MobileNetV2**:

```text
                  Conjunctiva Image (RGB)
                             ↓
              Preprocessing & Standardization
              (Aspect-ratio padding, 224x224x3, [-1, 1])
                             ↓
                 MobileNetV2 Backbone
             (Pretrained ImageNet feature maps)
                             ↓
                   GlobalAveragePooling2D
                             ↓
                   Dense(128, ReLU) + Dropout(0.3)
                             ↓
             ┌───────────────────────────────┐
             │                               │
    Head 1: Classification         Head 2: Regression
    Dense(1, Sigmoid)              Dense(1, Linear)
    Binary Cross-Entropy Loss      Huber Loss (delta=1.0)
    P(Anemia) ∈ [0, 1]             Estimated Hb (g/dL)
             │                               │
             └───────────────┬───────────────┘
                             ↓
            Demographic-Aware WHO Severity Rules
            (Normal, Mild, Moderate, Severe)
```

### Why Multi-Task Learning?
1. **Shared Feature Representation**: Simultaneous training on binary classification and continuous regression acts as regularizing multi-task learning, improving generalization on clinical sample sizes.
2. **Mitigating Extreme Severity Imbalance**: In clinical datasets, severe anemia is often statistically rare (< 2% of cohort). Training a direct 4-class classifier yields poor generalization on rare classes. Multi-task regression provides continuous gradient signal across all clinical ranges, allowing deterministic WHO stratification.

---

## 3. Repository Structure

```text
C:\anemia final\
│
├── archive (14).zip                # Pristine original clinical dataset
│
├── notebooks\
│   └── AnemiaSense_Model_Training.ipynb  # Comprehensive 22-section Jupyter Notebook
│
├── data\
│   ├── raw\                        # Untouched raw dataset copies
│   ├── extracted\                  # Extracted cohorts (India & Italy)
│   ├── processed\                  # Normalized cache & preprocessed tensors
│   └── splits\                     # Patient-level stratified split manifests
│
├── src\
│   ├── dataset_inspection.py       # Data cleaning, schema validation & MD5 duplicate check
│   ├── preprocessing.py            # Optical QA, ROI extraction & background standardization
│   ├── model.py                    # Multi-task MobileNetV2 architecture & compilation
│   ├── train.py                    # Two-stage training orchestrator & checkpointing
│   ├── evaluate.py                 # Diagnostic evaluation, metrics & figure generators
│   └── explainability.py           # Grad-CAM attention heatmaps on layer out_relu
│
├── models\
│   ├── checkpoints\                # Best validation checkpoints during training
│   └── exported\                   # Production .keras model, metadata & deployment ZIP
│
├── outputs\
│   ├── figures\                    # Confusion matrices, regression plots & curves
│   │   └── gradcam\                # Visual interpretability heatmaps
│   ├── metrics\                    # Evaluation results CSV, classification report, history
│   └── logs\                       # Training & execution logs
│
├── configs\
│   ├── preprocessing_config.json   # Input sizing, padding, and optical thresholds
│   ├── training_config.json        # Hyperparameters, loss weights, and stage configs
│   └── severity_rules.json         # Demographic-aware WHO hemoglobin cutoffs
│
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive system documentation
└── .gitignore                      # Git exclusion rules for large datasets and caches
```

---

## 4. Dataset Profile & Quality Assurance

- **Cohorts**: Dual international clinical cohorts (Karapakkam, Chennai, India: N=95; Bari, Italy: N=123).
- **Audit & Cleaning**:
  - European comma-decimal sanitization (`15,1` -> `15.1`).
  - Unrecorded lab values handled without imputation (Italy patient 93 removed).
  - Typographical filename correction (`_forniceal_palplebral.png` in Italy patient 95).
  - Cryptographic MD5 hash verification confirming **0 duplicate images**.
  - Total audited clean cohort: **217 patients**.
- **Patient-Level Split**:
  - Stratified 70% Train, 15% Validation, 15% Test.
  - Zero patient leakage across splits (enforced via programmatic assertions).

---

## 5. Training Strategy

1. **Stage 1 (Transfer Learning)**:
   - Backbone: ImageNet-pretrained MobileNetV2 frozen.
   - Head training: Dense(128, ReLU) + Dropout(0.3) + Multi-task heads.
   - Optimizer: Adam ($\text{lr} = 10^{-3}$).
   - Epochs: 12 epochs.
2. **Stage 2 (Fine-Tuning)**:
   - Top 20 layers of MobileNetV2 unfrozen; early layers and batch norm kept stable.
   - Optimizer: Adam ($\text{lr} = 10^{-5}$) with `ReduceLROnPlateau` and `EarlyStopping`.
   - Epochs: 8 epochs.
   - Loss balance: Binary Cross-Entropy (weight: 1.0) + Huber Loss (weight: 0.20).

---

## 6. Web Application Backend Integration

The exported model (`models/exported/anemia_mobilenetv2_multitask.keras`) is optimized for seamless deployment in web backends (FastAPI, Flask, Django):

```python
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. Load model and severity rules
model = keras.models.load_model("models/exported/anemia_mobilenetv2_multitask.keras")
with open("configs/severity_rules.json", "r") as f:
    severity_rules = json.load(f)

def predict_anemia(image_bytes, gender="M", age=30):
    # 2. Decode and preprocess
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Square padding and resize
    h, w = img_rgb.shape[:2]
    max_d = max(h, w)
    pad = np.zeros((max_d, max_d, 3), dtype=np.uint8)
    pad[(max_d - h)//2:(max_d - h)//2 + h, (max_d - w)//2:(max_d - w)//2 + w] = img_rgb
    resized = cv2.resize(pad, (224, 224), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(np.float32) / 127.5) - 1.0
    
    # 4. Multi-task inference
    preds = model.predict(norm[np.newaxis, ...], verbose=0)
    p_anemia = float(preds[0][0, 0])
    est_hb = float(preds[1][0, 0])
    
    # 5. WHO Severity Stratification
    threshold = 13.0 if gender.upper() == 'M' else 12.0
    if est_hb >= threshold:
        severity = "NORMAL"
    elif est_hb < 8.0:
        severity = "SEVERE_ANEMIA"
    elif est_hb < 11.0:
        severity = "MODERATE_ANEMIA"
    else:
        severity = "MILD_ANEMIA"
        
    return {
        "anemia_probability": round(p_anemia, 4),
        "estimated_hb_g_dl": round(est_hb, 2),
        "severity_level": severity,
        "clinical_action": severity_rules["clinical_actions"][severity],
        "disclaimer": severity_rules["disclaimer"]
    }
```

### Medical Disclaimer
*AnemiaSense provides automated non-invasive optical screening estimates for risk stratification. It is not an FDA-approved medical diagnostic device. All screening flags must be verified through certified laboratory phlebotomy (Complete Blood Count / automated hematology analyzer).*
