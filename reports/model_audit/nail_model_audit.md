# AnemiaSense — Nail Bed Model Audit & Technical Specification

**Auditor:** Senior Medical Machine Learning Engineer  
**Date of Audit:** October 10, 2026  
**Project Workspace:** `C:\anemia final`  
**Model Architecture:** MobileNetV3Large Continuous Hemoglobin Regression Pipeline  
**Production Checkpoint:** `models/nail/mobilenetv3_corrected_preprocessing_best.keras`  
**Historical Prototype Note:** Replaces legacy prototype candidate (`mobilenetv3_corrected_preprocessing_best_compat.keras` in `backups/nail_integration_backup/`).  

---

## 1. Model Summary & Architectural Topology

| Attribute | Specification |
|---|---|
| **Backbone Network** | `MobileNetV3Large` (Depthwise-separable convolutions, Squeeze-and-Excitation attention) |
| **Input Tensor Shape** | `(None, 224, 224, 3)` RGB float32 |
| **Internal Preprocessing Layer** | `Rescaling(scale=1/127.5, offset=-1.0)` built directly into the computational graph |
| **Input Data Contract** | Raw `[0, 255]` uint8 RGB pixels (**Zero external normalization required**) |
| **Bottleneck Feature Layer** | `GlobalAveragePooling2D` generating a 960-dimensional dense feature representation |
| **Regression Output Head** | `hb_g_dL_output` (`Dense(1, activation='linear')`) estimating continuous Hemoglobin ($g/dL$) |
| **Paired Binary Classifier** | `StandardScaler` + `LogisticRegression(class_weight='balanced', max_iter=2000)` on 960 features |
| **Total Parameters** | **2,997,313** parameters (11.43 MB) |
| **Trainable Head Parameters** | 961 parameters |
| **Frozen Backbone Parameters** | 2,996,352 parameters |

---

## 2. Preprocessing & Input Data Pipeline

1. **Aspect-Ratio Preserved Resizing:**  
   Image is centered and resized to $224 \times 224$ pixels using symmetric border reflection/padding to avoid distorting capillary vascular geometry.
2. **Color Channel Integrity:**  
   Standard RGB color ordering is maintained. Unlike the Eye model (which requires external $[-1, 1]$ normalization), the Nail model handles feature normalization internally.
3. **Region of Interest (ROI) Isolation:**  
   The palmar and subungual nail bed is targeted to minimize surrounding skin reflectance variance.

---

## 3. Clinical Stratification & Thresholds

Continuous estimated Hemoglobin concentrations ($g/dL$) are mapped dynamically according to clinical screening rules:

| Estimated $Hb$ Range | Clinical Classification | Recommended Directive |
|---|:---:|---|
| **$< 8.0\text{ g/dL}$** | **Severe Anemia** | Critical microvascular pallor detected. Urgent clinical consultation and venous laboratory CBC required. |
| **$8.0 - 10.0\text{ g/dL}$** | **Moderate Anemia** | Markedly below clinical baseline. Comprehensive diagnostic workup and serum ferritin panel advised. |
| **$10.0 - 12.0\text{ g/dL}$** (Females) / **$< 13.0\text{ g/dL}$** (Males) | **Mild Anemia** | Marginal pallor detected. Dietary iron optimization and confirmatory laboratory screening indicated. |
| **$\ge 12.0\text{ g/dL}$** (Females) / **$\ge 13.0\text{ g/dL}$** (Males) | **Normal / Non-Anemic** | Tissue erythema and vascular absorption index consistent with physiological baseline. |

---

## 4. Multi-Modal Independence Protocol

- **Zero Synthetic Averaging:**  
  The Eye and Nail models operate as **independent, orthogonal diagnostic modalities**. 
- Even when both images are provided in the Combined evaluation interface, each model processes its respective image through its isolated inference graph to provide transparent, multi-perspective clinical evidence.
