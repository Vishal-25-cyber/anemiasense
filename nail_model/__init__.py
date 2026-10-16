"""
AnemiaSense — Nail Model Package
Integrates the new MobileNetV3Large continuous hemoglobin regression model
for non-invasive nail-bed anemia screening.
"""

from nail_model.model_loader import (
    load_nail_model,
    get_nail_model,
    get_nail_model_status,
    NAIL_MODEL_PATH,
)
from nail_model.preprocessing import (
    preprocess_nail_image,
    validate_nail_image,
    evaluate_nail_image_quality,
)
from nail_model.inference import predict_nail_hb
from nail_model.gradcam import generate_nail_gradcam
from nail_model.validation import run_nail_model_diagnostics

__all__ = [
    "load_nail_model",
    "get_nail_model",
    "get_nail_model_status",
    "NAIL_MODEL_PATH",
    "preprocess_nail_image",
    "validate_nail_image",
    "evaluate_nail_image_quality",
    "predict_nail_hb",
    "generate_nail_gradcam",
    "run_nail_model_diagnostics",
]
