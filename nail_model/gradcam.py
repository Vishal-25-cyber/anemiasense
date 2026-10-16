"""
AnemiaSense — Nail Explainable AI (Grad-CAM)
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for the MobileNetV3Large continuous regression model.

TECHNICAL ARCHITECTURE NOTE:
The model contains a nested Functional layer named 'MobileNetV3Large'.
Grad-CAM targets 'activation_39' (final 7x7x960 convolutional activation)
or 'conv_1' within the nested backbone. Gradients are propagated from the
scalar Hb output through the top-level Dense and GlobalAveragePooling2D layers.
"""

import base64
import threading
from typing import Optional, Dict, Any, Union
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from nail_model.model_loader import get_nail_model
from nail_model.preprocessing import preprocess_nail_image

_grad_nail_model: Optional[keras.Model] = None
_grad_nail_lock = threading.Lock()
_grad_layer_name: Optional[str] = None


def get_nail_grad_submodel(
    model: keras.Model,
    candidate_layers=("activation_39", "conv_1")
) -> Optional[keras.Model]:
    """
    Builds and caches the intermediate gradient model that outputs both the
    target spatial convolutional feature map and the backbone output.
    Handles the nested Functional MobileNetV3Large backbone graph cleanly.
    """
    global _grad_nail_model, _grad_layer_name

    if _grad_nail_model is not None:
        return _grad_nail_model

    with _grad_nail_lock:
        if _grad_nail_model is not None:
            return _grad_nail_model

        try:
            base_model = model.get_layer("MobileNetV3Large")
            selected_layer = None

            for layer_name in candidate_layers:
                try:
                    selected_layer = base_model.get_layer(layer_name)
                    _grad_layer_name = layer_name
                    break
                except ValueError:
                    continue

            if selected_layer is None:
                print(f"[AnemiaSense Nail Grad-CAM] None of candidate layers {candidate_layers} found in backbone.")
                return None

            # Submodel mapping from backbone input to [conv_activation, backbone_output]
            submodel = keras.Model(
                inputs=base_model.inputs,
                outputs=[selected_layer.output, base_model.output],
                name="Nail_GradCAM_Submodel",
            )
            _grad_nail_model = submodel
            print(f"[AnemiaSense Nail Grad-CAM] Submodel compiled successfully targeting layer '{_grad_layer_name}'.")
            return _grad_nail_model

        except Exception as e:
            print(f"[AnemiaSense Nail Grad-CAM] Failed to initialize Grad-CAM submodel: {e}")
            return None


def generate_nail_gradcam(
    image_input: Union[np.ndarray, bytes, bytearray],
    crop_coords: Optional[str] = None,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Dict[str, Any]:
    """
    Generates a Grad-CAM visualization for the nail Hb prediction.

    Pipeline:
      1. Preprocesses image using verified pipeline.
      2. Runs gradient tape to compute d(Hb)/d(FeatureMap).
      3. Global-average pools the gradients to weight feature channels.
      4. Computes positive ReLU combination of weighted feature maps.
      5. Normalizes heatmap to [0, 1].
      6. Resizes heatmap to image dimensions and blends with padded RGB image.
      7. Returns base64 PNG data URI and explainability metadata.

    Fails gracefully: Returns empty string and diagnostic notice if gradients
    cannot be computed, keeping inference 100% functional.
    """
    model, err = get_nail_model()
    if model is None:
        return {
            "gradcam_available": False,
            "gradcam_overlay_base64": "",
            "error": f"Nail model not loaded: {err}",
            "layer_used": None,
        }

    try:
        # Preprocess using verified pipeline
        input_tensor, padded_rgb, _ = preprocess_nail_image(
            image_input, target_size=(224, 224), crop_coords=crop_coords
        )

        grad_submodel = get_nail_grad_submodel(model)
        if grad_submodel is None:
            return {
                "gradcam_available": False,
                "gradcam_overlay_base64": "",
                "error": "Could not construct Grad-CAM computational submodel.",
                "layer_used": None,
            }

        gap_layer = model.get_layer("global_average_pooling")
        dense_head = model.get_layer("hb_g_dL_output")

        tf_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # GradientTape forward and backward propagation
        with tf.GradientTape() as tape:
            conv_activations, backbone_output = grad_submodel(tf_tensor)
            tape.watch(conv_activations)
            pooled_features = gap_layer(backbone_output)
            predictions = dense_head(pooled_features)
            loss = predictions[0, 0]

        gradients = tape.gradient(loss, conv_activations)
        if gradients is None:
            return {
                "gradcam_available": False,
                "gradcam_overlay_base64": "",
                "error": "Gradient computation returned None (disconnected gradient graph).",
                "layer_used": _grad_layer_name,
            }

        # Global average pooling of gradients along spatial dimensions
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

        # Weight feature channels by pooled gradients
        feature_maps = conv_activations[0]
        cam = tf.reduce_sum(tf.multiply(pooled_gradients, feature_maps), axis=-1).numpy()

        # Apply ReLU to retain only features that positively influence the prediction
        cam = np.maximum(cam, 0)
        max_cam = np.max(cam)
        if max_cam > 1e-8:
            cam = cam / max_cam
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match image dimensions
        h_out, w_out = padded_rgb.shape[:2]
        heatmap_resized = cv2.resize(cam, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))

        # Colorize and blend overlay
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(padded_rgb, 1.0 - alpha, colored_cam_rgb, alpha, 0)

        # Encode to PNG base64
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", overlay_bgr)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay image to PNG.")

        b64_uri = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return {
            "gradcam_available": True,
            "gradcam_overlay_base64": b64_uri,
            "layer_used": f"{_grad_layer_name} (MobileNetV3Large final conv activation)",
            "heatmap_min": float(np.min(cam)),
            "heatmap_max": float(np.max(cam)),
            "spatial_resolution": [w_out, h_out],
            "disclaimer": (
                "Grad-CAM is an exploratory visualization. It highlights visual activations "
                "within the current checkpoint and does not confirm clinical validity."
            ),
        }

    except Exception as e:
        print(f"[AnemiaSense Nail Grad-CAM Warning] Visualization generation failed: {e}")
        return {
            "gradcam_available": False,
            "gradcam_overlay_base64": "",
            "error": str(e),
            "layer_used": _grad_layer_name,
        }
