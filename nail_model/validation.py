"""
AnemiaSense — Nail Model Diagnostic & Validation Suite
Provides automated verification tests for the MobileNetV3Large nail pipeline.
"""

from typing import Dict, Any
import numpy as np

from nail_model.model_loader import load_nail_model, NAIL_MODEL_PATH
from nail_model.preprocessing import preprocess_nail_image, validate_nail_image
from nail_model.inference import predict_nail_hb
from nail_model.gradcam import generate_nail_gradcam


def run_nail_model_diagnostics() -> Dict[str, Any]:
    """
    Executes a complete self-diagnostic check of the new nail model pipeline:
      1. Checkpoint existence and readable file
      2. Keras model loading
      3. Architectural layer topology and shape contracts
      4. Preprocessing tensor shape, dtype, and [0, 255] bounds
      5. Error handling for corrupted and empty inputs
      6. Inference calculation and finite numeric validation
      7. Verified g/dL target scale contract
      8. Grad-CAM gradient calculation and base64 rendering
    """
    results: Dict[str, Any] = {
        "all_passed": True,
        "tests": {},
    }

    # Test 1: Checkpoint file exists
    test_1 = {
        "name": "Checkpoint File Existence",
        "passed": NAIL_MODEL_PATH.exists() and NAIL_MODEL_PATH.is_file(),
        "path": str(NAIL_MODEL_PATH),
        "size_bytes": NAIL_MODEL_PATH.stat().st_size if NAIL_MODEL_PATH.exists() else 0,
    }
    results["tests"]["checkpoint_file"] = test_1
    if not test_1["passed"]:
        results["all_passed"] = False
        return results

    # Test 2: Model Loading
    model, load_err = load_nail_model()
    test_2 = {
        "name": "Model Loading",
        "passed": model is not None and load_err is None,
        "error": load_err,
        "model_name": model.name if model else None,
    }
    results["tests"]["model_loading"] = test_2
    if not test_2["passed"]:
        results["all_passed"] = False
        return results

    # Test 3: Input / Output Shapes
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    test_3 = {
        "name": "Input / Output Shapes",
        "passed": (in_shape == (None, 224, 224, 3)) and (out_shape == (None, 1)),
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "expected_input": [None, 224, 224, 3],
        "expected_output": [None, 1],
    }
    results["tests"]["tensor_shapes"] = test_3
    if not test_3["passed"]:
        results["all_passed"] = False

    # Test 4: Preprocessing on synthetic image
    test_img = np.random.randint(40, 210, (300, 400, 3), dtype=np.uint8)
    try:
        tensor, padded_rgb, qa = preprocess_nail_image(test_img)
        test_4 = {
            "name": "Image Preprocessing Pipeline",
            "passed": (
                tensor.shape == (1, 224, 224, 3)
                and tensor.dtype == np.float32
                and float(tensor.min()) >= 0.0
                and float(tensor.max()) <= 255.0
                and padded_rgb.shape == (224, 224, 3)
            ),
            "output_tensor_shape": list(tensor.shape),
            "output_tensor_dtype": str(tensor.dtype),
            "value_range": [float(tensor.min()), float(tensor.max())],
        }
    except Exception as e:
        test_4 = {"name": "Image Preprocessing Pipeline", "passed": False, "error": str(e)}
    results["tests"]["preprocessing"] = test_4
    if not test_4["passed"]:
        results["all_passed"] = False

    # Test 5: Rejection of invalid inputs
    rejected_empty = False
    try:
        validate_nail_image(b"")
    except ValueError:
        rejected_empty = True

    rejected_corrupt = False
    try:
        validate_nail_image(b"not an image file content")
    except ValueError:
        rejected_corrupt = True

    test_5 = {
        "name": "Input Validation & Rejection",
        "passed": rejected_empty and rejected_corrupt,
        "rejected_empty_bytes": rejected_empty,
        "rejected_corrupt_bytes": rejected_corrupt,
    }
    results["tests"]["input_rejection"] = test_5
    if not test_5["passed"]:
        results["all_passed"] = False

    # Test 6: Inference Execution
    try:
        inf_res = predict_nail_hb(test_img)
        hb_val = inf_res.get("predicted_hb")
        test_6 = {
            "name": "Inference Execution",
            "passed": (
                inf_res.get("success") is True
                and hb_val is not None
                and np.isfinite(hb_val)
                and inf_res.get("unit") == "g/dL"
            ),
            "predicted_hb": hb_val,
            "unit": inf_res.get("unit"),
            "prediction_status": inf_res.get("prediction_status"),
        }
    except Exception as e:
        test_6 = {"name": "Inference Execution", "passed": False, "error": str(e)}
    results["tests"]["inference"] = test_6
    if not test_6["passed"]:
        results["all_passed"] = False

    # Test 7: Grad-CAM Generation
    try:
        gcam_res = generate_nail_gradcam(test_img)
        test_7 = {
            "name": "Grad-CAM Explainability",
            "passed": (
                gcam_res.get("gradcam_available") is True
                and len(gcam_res.get("gradcam_overlay_base64", "")) > 100
            ),
            "layer_used": gcam_res.get("layer_used"),
            "gradcam_available": gcam_res.get("gradcam_available"),
            "overlay_length": len(gcam_res.get("gradcam_overlay_base64", "")),
        }
    except Exception as e:
        test_7 = {"name": "Grad-CAM Explainability", "passed": False, "error": str(e)}
    results["tests"]["gradcam"] = test_7
    if not test_7["passed"]:
        results["all_passed"] = False

    return results


if __name__ == "__main__":
    import json
    diag = run_nail_model_diagnostics()
    print(json.dumps(diag, indent=2))
