# AnemiaSense — Nail Model Integration & Technical Reference Guide

## 1. Executive Summary & Objective

This document specifies the complete replacement of the legacy nail model integration in the **AnemiaSense** non-invasive anemia screening application with the newly trained **MobileNetV3Large** continuous hemoglobin regression model (`mobilenetv3_corrected_preprocessing_best.keras`).

The eye model pipeline (MobileNetV2 dual-head classification and regression) remains **100% independent and unmodified**.

---

## 2. Model Location & Architecture

| Parameter | Specification | Verification Status |
|---|---|:---:|
| **Active Production Checkpoint** | `models/nail/mobilenetv3_corrected_preprocessing_best.keras` | **Verified** |
| **Model Format** | Keras 3 ZIP Archive (`.keras`) | **Verified** |
| **File Size** | 12,683,152 bytes (~12.1 MB) | **Verified** |
| **SHA-256 Checksum** | `1a140c8fba4cfc66770591942a6d03b47426679f5dc492ac3611c852725e015c` | **Verified** |
| **Backbone Network** | MobileNetV3Large (Depthwise-separable convolutions + SE attention) | **Verified** |
| **Input Shape** | `(None, 224, 224, 3)` | **Verified** |
| **Output Shape** | `(None, 1)` | **Verified** |
| **Output Head Layer Name** | `hb_g_dL_output` (`Dense(1, activation='linear')`) | **Verified** |
| **Feature Bottleneck** | `global_average_pooling` (`GlobalAveragePooling2D`) — 960 features | **Verified** |

---

## 3. Verified Preprocessing Pipeline (10 Steps)

The model features an **internal Rescaling layer** (`scale=1/127.5, offset=-1.0`) embedded directly inside the computational graph.

> **CRITICAL PREPROCESSING CONTRACT:**  
> **Input must be fed as raw float32 pixels in the range `[0.0, 255.0]`.**  
> External normalization to `[-1, 1]` must **NOT** be applied to prevent double normalization.

### 10-Step Pipeline:
1. **Accept Image Input**: Accepts image file bytes, buffer, or numpy array.
2. **Readability Validation**: Verifies headers, channels, and decodability via OpenCV.
3. **Rejection of Invalid Inputs**: Empty bytes, corrupted files, and resolutions below 16x16 are rejected with descriptive `ValueError`.
4. **Color Space Standardization**: Converts input strictly to RGB format.
5. **Interactive Region of Interest (ROI)**: If user crop coordinates are provided, slices bounding box (`[y0:y0+h0, x0:x0+w0]`).
6. **Aspect-Ratio Preserved Square Padding**: Centers tissue crop on a square canvas with symmetric reflection padding to prevent capillary distortion.
7. **Spatial Resizing**: Resizes canvas to $224 \times 224$ pixels via `cv2.INTER_AREA`.
8. **Numerical Conversion**: Converts array to `np.float32` preserving raw pixel values `[0.0, 255.0]`.
9. **Batch Dimension**: Expands dimensions to `(1, 224, 224, 3)`.
10. **Tensor Contract Validation**: Validates dimensions, finiteness, and bounds before model evaluation.

---

## 4. Verified Target Scale & Unit Conversion

- **Original Training Dataset Metadata:** Target column labeled `HB_LEVEL_GperL` (grams per liter).
- **Target Scale Verification:**  
  1. The output layer is explicitly named `hb_g_dL_output` in the model graph.
  2. Inference outputs evaluated on sample patient images yield values between $6.8\text{ g/dL}$ and $15.5\text{ g/dL}$ (matching standard physiological human hemoglobin ranges).
  3. Dividing by 10 would yield unphysical values ($0.68 - 1.55\text{ g/dL}$).
- **Confirmed Conversion:** The model directly predicts continuous Hemoglobin in **$\text{g/dL}$**.  
  **Conversion factor:** $1.0$ (no post-hoc division required).

---

## 5. Explainable AI (Grad-CAM) Integration

- **Target Convolutional Layer:** `activation_39` (final hard-swish activation of the nested MobileNetV3Large backbone, shape `[None, 7, 7, 960]`).
- **Alternative Fallback Layer:** `conv_1` (1x1 projection convolution).
- **Gradient Flow:** Gradients of scalar Hb output are backpropagated through `hb_g_dL_output` and `global_average_pooling` to the backbone's feature maps using `tf.GradientTape()`.
- **Overlay Blending:** Heatmap is normalized to `[0.0, 1.0]`, resized to original crop dimensions, colorized via `cv2.COLORMAP_JET`, and alpha-blended ($\alpha=0.45$) onto the input canvas.
- **Fault-Tolerance:** If gradient computation is unavailable or fails, Grad-CAM returns an empty URI while **hemoglobin inference completes unaffected**.
- **Important Checkpoint Notice:**  
  *The exact training checkpoint utilized for historical explainability trials has not been confirmed independently. Grad-CAM visual heatmaps in this version represent exploratory visual activations of the current `mobilenetv3_corrected_preprocessing_best.keras` checkpoint and do not constitute clinical proof of subungual microvascular focus.*

---

## 6. API Usage Reference

### 6.1 Nail Screening Endpoint
`POST /api/predict/nail`

**Form Data Parameters:**
- `file`: Multipart image file (PNG/JPG/JPEG).
- `gender`: `"M"` or `"F"` (default `"M"`).
- `age`: Integer age in years (default `30`).
- `include_gradcam`: Boolean (default `true`).
- `dietary_pref`: `"vegetarian"` or `"non-vegetarian"` (default `"vegetarian"`).
- `crop_coords`: Optional bounding box string `"x,y,w,h"`.

**Sample Response:**
```json
{
  "success": true,
  "model_used": "Nail Bed — MobileNetV3Large Hb Regression",
  "model_architecture": "MobileNetV3Large",
  "model_file": "mobilenetv3_corrected_preprocessing_best.keras",
  "record_id": "67...a1",
  "report_id": "AS-20261016-95FFFF",
  "patient_context": {
    "gender": "M",
    "gender_display": "Male",
    "age": 30,
    "demographic_who_threshold_g_dl": 13.0,
    "dietary_preference": "Vegetarian"
  },
  "screening_results": {
    "estimated_hemoglobin_g_dl": 11.47,
    "unit": "g/dL",
    "target_scale_verified": true,
    "preprocessing_verified": true,
    "severity_level": "MILD_ANEMIA",
    "severity_label": "Mild anemia screening range",
    "clinical_code": "MILD_ANEMIA",
    "clinical_action": "Slightly low. Consider iron-rich foods and consult a doctor if fatigued.",
    "research_prototype_notice": "Academic research prototype (MAE ~1.96 g/dL, R² negative). Not a clinical diagnosis."
  },
  "explainability": {
    "gradcam_available": true,
    "gradcam_layer": "activation_39 (MobileNetV3Large final conv activation)",
    "gradcam_overlay_base64": "data:image/png;base64,..."
  },
  "quality_assurance": {
    "blur_score": 45.2,
    "brightness": 128.4,
    "glare_percentage": 0.0,
    "passed": true,
    "warnings": []
  },
  "nutritional_guidance": { ... },
  "disclaimer": "..."
}
```

### 6.2 Health & Status Endpoint
`GET /api/health`  
Returns operational health for both independent pipelines.

### 6.3 Automated Diagnostics Endpoint
`GET /api/diagnostics/nail`  
Executes 7-point automated verification suite.

---

## 7. Known Limitations & Research Prototype Disclaimer

1. **Academic Research Prototype:** This model is not certified as a medical device or laboratory diagnostic instrument.
2. **Regression Performance Variance:** Documented locked-test performance on held-out test cohort ($N=38$ patients, 114 crops):
   - **MAE:** approximately $1.96\text{ g/dL}$.
   - **$R^2$ Score:** Negative (indicates substantial dispersion around ground truth).
3. **Clinical Action Directive:** Optical estimates cannot rule in or rule out anemia. Venous blood draw (CBC + Ferritin) is mandatory for clinical decision making.

---

## 8. Rollback Procedure

If rollback to the legacy prototype is ever required:
1. A complete backup is preserved in `backups/nail_integration_backup/`.
2. Restore `backups/nail_integration_backup/app.py.bak` to `web/app.py`.
3. Restore `backups/nail_integration_backup/Nail_Model` to the root directory.
4. Restart the FastAPI service.
