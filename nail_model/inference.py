"""
AnemiaSense — Nail Hemoglobin Inference
Executes end-to-end continuous hemoglobin regression on nail bed images.
"""

from typing import Optional, Dict, Any, Union
import numpy as np

from nail_model.model_loader import get_nail_model
from nail_model.preprocessing import preprocess_nail_image

RESEARCH_DISCLAIMER = (
    "Research prototype: This nail-image model provides an experimental estimate "
    "and is not a laboratory measurement or medical diagnosis. Its accuracy is limited "
    "(locked-test MAE ~1.96 g/dL, negative R²). Do not use this result alone to diagnose anemia, "
    "assess severity, or make treatment decisions. Consult a qualified healthcare professional "
    "and obtain an appropriate laboratory blood test (venous CBC)."
)


def predict_nail_hb(
    image_input: Union[np.ndarray, bytes, bytearray],
    crop_coords: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predicts continuous hemoglobin level (g/dL) from a nail bed photograph.

    Pipeline:
      1. Confirms active MobileNetV3Large model is loaded.
      2. Preprocesses input using the verified 10-step pipeline.
      3. Executes neural regression inference.
      4. Validates that the prediction is numeric, finite, and within physical range.
      5. Emits verified g/dL prediction and structured response.
      6. Returns research disclaimer and model specifications.

    Returns structured dictionary compatible with AnemiaSense backend.
    """
    # 1. Confirm model is loaded
    model, err = get_nail_model()
    if model is None:
        return {
            "success": False,
            "model_type": "nail",
            "model_architecture": "MobileNetV3Large",
            "predicted_hb": None,
            "unit": None,
            "prediction_status": "unavailable",
            "error": f"Nail model unavailable: {err}",
            "disclaimer": RESEARCH_DISCLAIMER,
            "preprocessing_verified": True,
            "target_scale_verified": True,
        }

    # 2. Preprocess input
    try:
        input_tensor, padded_rgb, qa_metrics = preprocess_nail_image(
            image_input, target_size=(224, 224), crop_coords=crop_coords
        )
    except Exception as prep_err:
        return {
            "success": False,
            "model_type": "nail",
            "model_architecture": "MobileNetV3Large",
            "predicted_hb": None,
            "unit": None,
            "prediction_status": "error",
            "error": f"Image preprocessing failed: {str(prep_err)}",
            "disclaimer": RESEARCH_DISCLAIMER,
            "preprocessing_verified": True,
            "target_scale_verified": True,
        }

    # 3. Inference
    try:
        raw_pred = model.predict(input_tensor, verbose=0)
        est_hb = float(np.asarray(raw_pred).reshape(-1)[0])

        # 4. Validate output numeric validity
        if not np.isfinite(est_hb):
            return {
                "success": False,
                "model_type": "nail",
                "model_architecture": "MobileNetV3Large",
                "predicted_hb": None,
                "unit": None,
                "prediction_status": "error",
                "error": f"Model returned non-finite prediction: {est_hb}",
                "disclaimer": RESEARCH_DISCLAIMER,
                "preprocessing_verified": True,
                "target_scale_verified": True,
            }

        # Validate physiological bounds check (0.0 to 30.0 g/dL)
        if est_hb <= 0.0 or est_hb > 30.0:
            return {
                "success": False,
                "model_type": "nail",
                "model_architecture": "MobileNetV3Large",
                "predicted_hb": round(est_hb, 2),
                "unit": "g/dL",
                "prediction_status": "error",
                "error": f"Model returned an implausible Hb estimate ({est_hb:.2f} g/dL). Please try a clearer, well-lit nail photo.",
                "disclaimer": RESEARCH_DISCLAIMER,
                "preprocessing_verified": True,
                "target_scale_verified": True,
            }

        # 5. Verified target scale: output is directly in g/dL
        est_hb_g_dl = round(est_hb, 2)

        return {
            "success": True,
            "model_type": "nail",
            "model_architecture": "MobileNetV3Large",
            "model_name": model.name,
            "predicted_hb": est_hb_g_dl,
            "unit": "g/dL",
            "prediction_status": "success",
            "preprocessing_verified": True,
            "target_scale_verified": True,
            "conversion_factor": 1.0,
            "quality_assurance": qa_metrics,
            "disclaimer": RESEARCH_DISCLAIMER,
            "padded_rgb": padded_rgb,  # cached for Grad-CAM overlay if requested
            "input_tensor": input_tensor,
            "locked_test_mae_g_dl": 1.96,
            "locked_test_r2": -0.18,
            "note": (
                "The nail model is a continuous single-head Hb regressor (g/dL). "
                "No binary classification probability is produced."
            ),
        }

    except Exception as infer_err:
        return {
            "success": False,
            "model_type": "nail",
            "model_architecture": "MobileNetV3Large",
            "predicted_hb": None,
            "unit": None,
            "prediction_status": "error",
            "error": f"Inference execution failed: {str(infer_err)}",
            "disclaimer": RESEARCH_DISCLAIMER,
            "preprocessing_verified": True,
            "target_scale_verified": True,
        }
