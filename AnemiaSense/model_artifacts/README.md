# AnemiaSense Model Artifacts Package

## Model Summary
- **Architecture**: MobileNetV2 Multi-Task Transfer Learning
- **Tasks**:
  1. Binary Anemia Screening (Sigmoid)
  2. Continuous Hemoglobin Estimation in g/dL (Linear)
- **Training Strategy**: Two-stage transfer learning with controlled augmentation and early stopping.

## Held-Out Test Set Performance (N=33)
- **Accuracy**: 93.94%
- **Sensitivity / Recall**: 92.86%
- **Specificity**: 94.74%
- **Precision**: 92.86%
- **F1-Score**: 0.9286
- **ROC-AUC**: 0.9662
- **Hemoglobin MAE**: 0.78 g/dL
- **Hemoglobin RMSE**: 1.08 g/dL
- **Hemoglobin R2**: 0.8642

## Artifact Inventory
- `anemia_mobilenetv2_multitask.keras`: Trained Keras model.
- `model_metadata.json`: Model architecture, training details, and test metrics.
- `preprocessing_config.json`: Standardized preprocessing configuration.
- `severity_rules.json`: WHO demographic cutoff rules.
- `evaluation_results.csv`: Held-out test set predictions.
- `training_history.csv`: Epoch training and validation history.
- `train_metadata.csv`, `validation_metadata.csv`, `test_metadata.csv`: Data splits.
- `gradcam_visualizations/`: Explainability heatmaps.
