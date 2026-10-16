"""
AnemiaSense — Automated Test Suite for Nail Model Integration & System Verification
Tests model architecture, verified preprocessing, Hb inference, Grad-CAM,
eye pipeline preservation, and FastAPI API routes.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from nail_model import (
    load_nail_model,
    get_nail_model,
    get_nail_model_status,
    preprocess_nail_image,
    validate_nail_image,
    predict_nail_hb,
    generate_nail_gradcam,
    run_nail_model_diagnostics,
    NAIL_MODEL_PATH,
)
from web.app import app, get_eye_model, preprocess_eye_image
from fastapi.testclient import TestClient


class TestNailModelIntegrity(unittest.TestCase):
    """Verifies checkpoint, architecture, and loading contracts."""

    def test_checkpoint_exists(self):
        self.assertTrue(NAIL_MODEL_PATH.exists(), f"Checkpoint missing: {NAIL_MODEL_PATH}")
        self.assertTrue(NAIL_MODEL_PATH.is_file(), f"Checkpoint is not a file: {NAIL_MODEL_PATH}")
        self.assertGreater(NAIL_MODEL_PATH.stat().st_size, 10_000_000, "Checkpoint file size too small")

    def test_model_loading(self):
        model, err = load_nail_model()
        self.assertIsNone(err, f"Model failed to load: {err}")
        self.assertIsNotNone(model, "Model is None after loading")
        self.assertEqual(model.name, "Nail_Hb_MobileNetV3_Corrected")

    def test_model_shapes(self):
        model, _ = load_nail_model()
        self.assertEqual(tuple(model.input_shape), (None, 224, 224, 3))
        self.assertEqual(tuple(model.output_shape), (None, 1))

    def test_model_status(self):
        status = get_nail_model_status()
        self.assertTrue(status["loaded"])
        self.assertEqual(status["architecture"], "MobileNetV3Large")
        self.assertEqual(status["output_unit"], "g/dL")
        self.assertTrue(status["target_scale_verified"])
        self.assertTrue(status["preprocessing_verified"])


class TestNailPreprocessing(unittest.TestCase):
    """Verifies the 10-step verified preprocessing pipeline."""

    def setUp(self):
        # Create a synthetic 300x400 RGB test image
        self.test_img = np.random.randint(40, 220, (300, 400, 3), dtype=np.uint8)

    def test_output_tensor_shape_and_range(self):
        tensor, padded_rgb, qa = preprocess_nail_image(self.test_img, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        # Verify raw [0, 255] float32 range (NOT [-1, 1] external pre-normalization)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 255.0)
        self.assertEqual(padded_rgb.shape, (224, 224, 3))
        self.assertEqual(padded_rgb.dtype, np.uint8)
        self.assertIn("blur_score", qa)
        self.assertIn("passed", qa)

    def test_interactive_crop(self):
        crop_coords = "10,10,100,100"
        tensor, padded_rgb, _ = preprocess_nail_image(self.test_img, crop_coords=crop_coords)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_rejection_of_empty_input(self):
        with self.assertRaises(ValueError):
            validate_nail_image(b"")

    def test_rejection_of_corrupt_input(self):
        with self.assertRaises(ValueError):
            validate_nail_image(b"non-image garbage bytes content")

    def test_rejection_of_none_input(self):
        with self.assertRaises(ValueError):
            validate_nail_image(None)


class TestNailInference(unittest.TestCase):
    """Verifies continuous hemoglobin regression inference and output scale."""

    def setUp(self):
        # Use actual sample nail image if available, else synthetic
        sample_path = WORKSPACE_DIR / "web" / "static" / "samples" / "sample_nail_mild_male.png"
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            self.sample_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            self.sample_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

    def test_inference_returns_finite_g_dl(self):
        res = predict_nail_hb(self.sample_img)
        self.assertTrue(res["success"], f"Prediction failed: {res.get('error')}")
        self.assertEqual(res["unit"], "g/dL")
        hb = res["predicted_hb"]
        self.assertIsInstance(hb, float)
        self.assertTrue(np.isfinite(hb))
        # Physiological boundaries check
        self.assertGreater(hb, 2.0)
        self.assertLess(hb, 25.0)
        self.assertTrue(res["target_scale_verified"])
        self.assertTrue(res["preprocessing_verified"])
        self.assertIn("disclaimer", res)


class TestNailGradCAM(unittest.TestCase):
    """Verifies Grad-CAM neural attention heatmap generation."""

    def setUp(self):
        sample_path = WORKSPACE_DIR / "web" / "static" / "samples" / "sample_nail_mild_male.png"
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            self.sample_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            self.sample_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

    def test_gradcam_overlay_generation(self):
        res = generate_nail_gradcam(self.sample_img)
        self.assertTrue(res["gradcam_available"], f"Grad-CAM unavailable: {res.get('error')}")
        overlay = res["gradcam_overlay_base64"]
        self.assertTrue(overlay.startswith("data:image/png;base64,"))
        self.assertGreater(len(overlay), 500)
        self.assertIn("activation_39", res["layer_used"])


class TestEyeModelPreservation(unittest.TestCase):
    """Mandatory verification: Eye model pipeline must be 100% intact and functional."""

    def test_eye_model_loading_and_architecture(self):
        eye_m, err = get_eye_model()
        self.assertIsNone(err, f"Eye model load error: {err}")
        self.assertIsNotNone(eye_m)
        self.assertEqual(eye_m.name, "AnemiaSense_MobileNetV2_MultiTask")
        self.assertEqual(tuple(eye_m.input_shape), (None, 224, 224, 3))
        # Must have 2 output heads: anemia_prob + hb_regression
        self.assertEqual(len(eye_m.outputs), 2)

    def test_eye_preprocessing_normalization(self):
        dummy = np.full((224, 224, 3), 127.5, dtype=np.uint8)
        norm, padded, tel = preprocess_eye_image(dummy)
        # Eye model uses external [-1, 1] normalization
        self.assertAlmostEqual(float(norm[0, 0, 0]), 0.0, places=1)
        self.assertEqual(norm.shape, (224, 224, 3))


class TestFastAPIIntegration(unittest.TestCase):
    """Tests FastAPI backend routes end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["models"]["eye_model"]["loaded"])
        self.assertTrue(data["models"]["nail_model"]["loaded"])
        self.assertEqual(data["models"]["nail_model"]["architecture"], "MobileNetV3Large")
        self.assertEqual(data["models"]["nail_model"]["unit"], "g/dL")

    def test_metadata_endpoint(self):
        res = self.client.get("/api/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("nail_model", data)
        self.assertEqual(data["nail_model"]["architecture"], "MobileNetV3Large")
        self.assertTrue(data["nail_model"]["target_scale_verified"])

    def test_diagnostics_endpoint(self):
        res = self.client.get("/api/diagnostics/nail")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["diagnostics"]["all_passed"])

    def test_nail_prediction_endpoint(self):
        sample_path = WORKSPACE_DIR / "web" / "static" / "samples" / "sample_nail_mild_male.png"
        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        res = self.client.post(
            "/api/predict/nail",
            files={"file": ("sample_nail.png", file_bytes, "image/png")},
            data={"gender": "M", "age": "30", "include_gradcam": "true", "dietary_pref": "vegetarian"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("MobileNetV3Large", data["model_used"])
        sr = data["screening_results"]
        self.assertIn("estimated_hemoglobin_g_dl", sr)
        self.assertEqual(sr["unit"], "g/dL")
        self.assertTrue(sr["target_scale_verified"])
        self.assertIn("severity_level", sr)
        self.assertIn("research_prototype_notice", sr)
        # Explainability
        self.assertTrue(data["explainability"]["gradcam_available"])
        self.assertTrue(data["explainability"]["gradcam_overlay_base64"].startswith("data:image/png;base64,"))
        # Nutritional guidance
        self.assertIn("nutritional_guidance", data)
        self.assertIn("iron_rich_foods", data["nutritional_guidance"])

    def test_combined_prediction_endpoint(self):
        eye_sample = WORKSPACE_DIR / "web" / "static" / "samples" / "sample_normal_male.png"
        nail_sample = WORKSPACE_DIR / "web" / "static" / "samples" / "sample_nail_mild_male.png"

        with open(eye_sample, "rb") as ef, open(nail_sample, "rb") as nf:
            res = self.client.post(
                "/api/predict/combined",
                files={
                    "eye_file": ("eye.png", ef.read(), "image/png"),
                    "nail_file": ("nail.png", nf.read(), "image/png"),
                },
                data={"gender": "M", "age": "35", "include_gradcam": "true"},
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        # Verify independence: separate eye and nail results
        self.assertIn("eye_result", data)
        self.assertIn("nail_result", data)
        self.assertTrue(data["eye_result"]["success"])
        self.assertTrue(data["nail_result"]["success"])
        # Check both have Grad-CAM overlays
        self.assertTrue(data["eye_result"]["gradcam_overlay_base64"].startswith("data:image/png;base64,"))
        self.assertTrue(data["nail_result"]["gradcam_overlay_base64"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
