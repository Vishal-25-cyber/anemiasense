"""
AnemiaSense — Test Evaluation & Clinical Metric Reporting
Computes diagnostic classification and continuous hemoglobin regression metrics,
generates confusion matrix heatmaps, regression plots, and evaluation reports.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, root_mean_squared_error, r2_score
)
from tensorflow import keras

from src.preprocessing import ConjunctivaPreprocessor


def evaluate_test_set(
    model: keras.Model,
    test_df: pd.DataFrame,
    workspace_dir: Path = Path(r"c:\anemia final"),
    training_history_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """Runs comprehensive evaluation on held-out test cohort and exports all figures & metrics."""
    preprocessor = ConjunctivaPreprocessor()
    X_test = np.array([preprocessor.process_image(p) for p in test_df['Mask_PNG_Path']], dtype=np.float32)

    y_true_cls = test_df['Anemic'].values.astype(int)
    y_true_hb = test_df['Hgb'].values.astype(float)

    preds = model.predict(X_test, verbose=0)
    pred_probs = preds[0].flatten()
    pred_cls = (pred_probs >= 0.5).astype(int)
    pred_hb = preds[1].flatten()

    # Classification Metrics
    acc = accuracy_score(y_true_cls, pred_cls)
    prec = precision_score(y_true_cls, pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, pred_cls, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_cls, pred_cls).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_true_cls, pred_probs)

    # Regression Metrics
    mae = mean_absolute_error(y_true_hb, pred_hb)
    rmse = root_mean_squared_error(y_true_hb, pred_hb)
    r2 = r2_score(y_true_hb, pred_hb)

    metrics_dict = {
        "accuracy": float(acc),
        "sensitivity_recall": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "hb_mae_g_dl": float(mae),
        "hb_rmse_g_dl": float(rmse),
        "hb_r2_score": float(r2)
    }

    print("=" * 60)
    print("             HELD-OUT TEST SET EVALUATION")
    print("=" * 60)
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Sensitivity:      {rec * 100:.2f}%")
    print(f"Specificity:      {spec * 100:.2f}%")
    print(f"Precision:        {prec * 100:.2f}%")
    print(f"F1-Score:         {f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Hemoglobin MAE:   {mae:.2f} g/dL")
    print(f"Hemoglobin RMSE:  {rmse:.2f} g/dL")
    print("=" * 60)

    # Patient-level results DataFrame
    res_df = test_df.copy()
    res_df['Predicted_Anemia_Prob'] = pred_probs
    res_df['Predicted_Anemic'] = pred_cls
    res_df['Predicted_Hb'] = pred_hb
    res_df['Hb_Absolute_Error'] = np.abs(pred_hb - y_true_hb)

    def derive_who_severity(row):
        thresh = 13.0 if row['Gender'] == 'M' else 12.0
        hb = row['Predicted_Hb']
        if hb >= thresh:
            return "Normal"
        elif hb < 8.0:
            return "Severe"
        elif hb < 11.0:
            return "Moderate"
        else:
            return "Mild"

    res_df['Predicted_Severity'] = res_df.apply(derive_who_severity, axis=1)

    # Export metrics files
    metrics_dir = workspace_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(metrics_dir / "evaluation_results.csv", index=False)

    cls_report = classification_report(y_true_cls, pred_cls, target_names=['Normal', 'Anemic'])
    with open(metrics_dir / "classification_report.txt", "w") as f:
        f.write(cls_report)

    # Export Model Metadata
    exported_dir = workspace_dir / "models" / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "AnemiaSense_MobileNetV2_MultiTask",
        "version": "1.0.0",
        "architecture": "MobileNetV2 Pretrained Transfer Learning Dual-Head",
        "test_metrics": metrics_dict,
        "test_sample_count": len(test_df)
    }
    with open(exported_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # ----------------- Figures Export -----------------
    figures_dir = workspace_dir / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(5, 4), dpi=150)
    cm = confusion_matrix(y_true_cls, pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anemic'], yticklabels=['Normal', 'Anemic'])
    plt.title(f"Test Confusion Matrix (Acc = {acc * 100:.1f}%)")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png")
    plt.close()

    # 2. Actual vs Predicted Hb Scatter
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(y_true_hb, pred_hb, color='royalblue', edgecolors='k', s=65, alpha=0.85)
    min_v = min(y_true_hb.min(), pred_hb.min()) - 0.5
    max_v = max(y_true_hb.max(), pred_hb.max()) + 0.5
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='1:1 Identity')
    plt.title(f"Actual vs Predicted Hb (MAE = {mae:.2f} g/dL)")
    plt.xlabel("Laboratory Hb (g/dL)")
    plt.ylabel("Estimated Hb (g/dL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted_hb.png")
    plt.close()

    # 3. Training Curves (if provided)
    if training_history_df is not None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
        axs[0].plot(training_history_df['loss'], label='Train Total Loss', color='navy')
        axs[0].plot(training_history_df['val_loss'], label='Val Total Loss', color='crimson')
        axs[0].set_title("Total Loss")
        axs[0].legend()

        axs[1].plot(training_history_df['anemia_output_binary_accuracy'], label='Train Acc', color='navy')
        axs[1].plot(training_history_df['val_anemia_output_binary_accuracy'], label='Val Acc', color='crimson')
        axs[1].set_title("Binary Accuracy")
        axs[1].legend()

        axs[2].plot(training_history_df['hb_output_mae'], label='Train Hb MAE', color='navy')
        axs[2].plot(training_history_df['val_hb_output_mae'], label='Val Hb MAE', color='crimson')
        axs[2].set_title("Hemoglobin MAE (g/dL)")
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(figures_dir / "training_curves.png")
        plt.close()

    print(f"[Evaluate] Exported metrics and figures to {metrics_dir} and {figures_dir}")
    return metrics_dict
