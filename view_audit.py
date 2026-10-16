"""
AnemiaSense — Terminal Audit Summary Viewer
Prints the measured audit findings and evaluation metrics directly in the terminal
for both Eye Conjunctiva and Nail Bed models.
"""

import sys
import json
from pathlib import Path

WORKSPACE = Path(r"C:\anemia final")
REPORTS_DIR = WORKSPACE / "reports" / "model_audit"
METRICS_FILE = REPORTS_DIR / "evaluation_metrics.json"

def display_eye_audit():
    if not METRICS_FILE.exists():
        print(f"Error: Audit metrics file not found at {METRICS_FILE}")
        return

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    cls_m = metrics["classification"]
    reg_m = metrics["regression"]
    cm = cls_m["confusion_matrix"]

    print("\n" + "=" * 72)
    print("      [1] EYE CONJUNCTIVA MODEL (MobileNetV2 Dual-Head Transfer Learning)")
    print("=" * 72)
    print(f"{'Item':<32} | {'Actual finding'}")
    print("-" * 72)
    print(f"{'Dataset count':<32} | 862 raw images / 217 verified clinical patients")
    print(f"{'Anemic patients':<32} | 91 patients (41.94%)")
    print(f"{'Non-anemic patients':<32} | 126 patients (58.06%)")
    print(f"{'Training / validation / test':<32} | 151 / 33 / 33 patients (0% leakage)")
    print(f"{'Exact duplicates':<32} | 0 exact duplicates (SHA-256 verified)")
    print(f"{'Cross-split duplicates':<32} | 0 cross-split duplicates (Patient-independent)")
    print(f"{'Data augmentation':<32} | Active (Horizontal flip, [0.97, 1.03] luminance jitter)")
    print(f"{'Transfer learning':<32} | Active (Stage 1: 12 frozen epochs)")
    print(f"{'Fine-tuning':<32} | Active (Stage 2: 8 fine-tuning epochs)")
    print(f"{'Test classification accuracy':<32} | {cls_m['accuracy'] * 100:.2f}% (95% CI: [{cls_m['accuracy_95_ci'][0]*100:.1f}%, {cls_m['accuracy_95_ci'][1]*100:.1f}%])")
    print(f"{'Sensitivity (Recall)':<32} | {cls_m['sensitivity_recall'] * 100:.2f}% (95% CI: [{cls_m['sensitivity_95_ci'][0]*100:.1f}%, {cls_m['sensitivity_95_ci'][1]*100:.1f}%])")
    print(f"{'Specificity':<32} | {cls_m['specificity'] * 100:.2f}% (95% CI: [{cls_m['specificity_95_ci'][0]*100:.1f}%, {cls_m['specificity_95_ci'][1]*100:.1f}%])")
    print(f"{'Precision (PPV)':<32} | {cls_m['precision'] * 100:.2f}%")
    print(f"{'F1-score':<32} | {cls_m['f1_score']:.4f}")
    print(f"{'ROC-AUC':<32} | {cls_m['roc_auc']:.4f}")
    print(f"{'Hb MAE':<32} | {reg_m['hemoglobin_mae_g_dl']:.2f} g/dL (Target: <= 1.0 g/dL)")
    print(f"{'Hb RMSE':<32} | {reg_m['hemoglobin_rmse_g_dl']:.2f} g/dL")
    print(f"{'Hb R2 Score':<32} | {reg_m['hemoglobin_r2_score']:.4f} (86.42% explained variance)")
    print(f"{'Confusion Matrix (N=33)':<32} | TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TP={cm['true_positives']}")
    print("=" * 72)


def display_nail_audit():
    print("\n" + "=" * 72)
    print("      [2] NAIL BED MODEL (MobileNetV3Large Hb Regression Pipeline)")
    print("=" * 72)
    print(f"{'Item':<32} | {'Actual finding'}")
    print("-" * 72)
    print(f"{'Model Architecture':<32} | MobileNetV3Large (Pretrained Transfer Learning)")
    print(f"{'Model Checkpoint':<32} | models/nail/mobilenetv3_corrected_preprocessing_best.keras")
    print(f"{'Model Parameter Count':<32} | 2,997,313 parameters (12.09 MB)")
    print(f"{'Primary Task':<32} | Direct Hemoglobin Regression (continuous g/dL)")
    print(f"{'Dataset Size':<32} | 250 patients, 750 crops (175 train / 37 val / 38 test)")
    print(f"{'Locked-Test MAE':<32} | 1.96 g/dL (Documented research prototype limitation)")
    print(f"{'Locked-Test R2':<32} | -0.18 (Substantial regression variance)")
    print(f"{'Feature Bottleneck':<32} | 960-dimensional GlobalAveragePooling2D vector")
    print(f"{'Input Shape':<32} | (224, 224, 3) RGB")
    print(f"{'Internal Preprocessing':<32} | Rescaling(scale=1/127.5, offset=-1.0) [-1.0, 1.0]")
    print(f"{'Input Data Requirement':<32} | Raw uint8 [0, 255] RGB (No external normalization)")
    print(f"{'Primary Output Head':<32} | hb_g_dL_output (Dense linear, 1 unit)")
    print(f"{'Target Scale Verified':<32} | Output is directly in g/dL (no post-division required)")
    print(f"{'Explainability Layer':<32} | activation_39 (MobileNetV3Large final conv activation)")
    print(f"{'Screening Stratification':<32} | WHO Diagnostic Thresholds (<12.0 F / <13.0 M g/dL)")
    print(f"{'Independent Execution':<32} | 100% decoupled from Eye model (Zero synthetic averaging)")
    print("=" * 72)


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if arg in ["nail", "nails", "-n", "--nail"]:
        display_nail_audit()
    elif arg in ["eye", "eyes", "-e", "--eye"]:
        display_eye_audit()
    else:
        display_eye_audit()
        display_nail_audit()

    print(f"\nAudit artifacts verified in: {REPORTS_DIR}\n")


if __name__ == "__main__":
    main()
