"""
AnemiaSense — Model Explainability via Grad-CAM Module
Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
for MobileNetV2 multi-task predictions to verify visual attention on conjunctival microvasculature.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.preprocessing import ConjunctivaPreprocessor


def compute_gradcam(
    model: keras.Model,
    img_normalized: np.ndarray,
    target_head: str = 'hb_output',
    conv_layer_name: str = 'out_relu'
) -> np.ndarray:
    """
    Computes 2D Grad-CAM attention heatmap for a specific output head.

    Args:
        model: Unified multi-task Keras model.
        img_normalized: Preprocessed image tensor of shape (224, 224, 3) normalized to [-1.0, 1.0].
        target_head: Head name ('hb_output' or 'anemia_output').
        conv_layer_name: Target convolutional layer (default: 'out_relu').

    Returns:
        heatmap: 2D numpy array (224, 224) normalized between 0.0 and 1.0.
    """
    # Build feature extractor model from input to [conv_layer, target_output]
    target_layer = model.get_layer(conv_layer_name)
    target_out = model.get_layer(target_head)

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, target_out.output]
    )

    img_tensor = tf.expand_dims(tf.convert_to_tensor(img_normalized, dtype=tf.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    # Global average pooling of gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    # ReLU on attention map (retain positive contributions)
    cam = np.maximum(cam, 0)
    max_val = np.max(cam)
    if max_val > 1e-8:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam)

    # Resize to original input dimensions
    heatmap = cv2.resize(cam, (img_normalized.shape[1], img_normalized.shape[0]), interpolation=cv2.INTER_LINEAR)
    return heatmap


def overlay_gradcam(
    img_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlays colorized Grad-CAM heatmap onto RGB original image."""
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_uint8, 1 - alpha, colored_cam_rgb, alpha, 0)
    return overlay


def generate_and_save_gradcam_figures(
    model: keras.Model,
    test_df: pd.DataFrame,
    output_dir: Path,
    num_samples: int = 4
) -> List[Path]:
    """Generates and saves clinical interpretability visualization panels for sample test cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    preprocessor = ConjunctivaPreprocessor()
    samples = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    for idx, (_, row) in enumerate(samples.iterrows()):
        img_path = row['Mask_PNG_Path']
        norm_img = preprocessor.process_image(img_path)
        raw_uint8 = np.uint8(np.clip((norm_img + 1.0) * 127.5, 0, 255))

        preds = model.predict(norm_img[np.newaxis, ...], verbose=0)
        p_anemia = float(preds[0][0, 0])
        p_hb = float(preds[1][0, 0])

        heatmap_hb = compute_gradcam(model, norm_img, target_head='hb_output')
        overlay_hb = overlay_gradcam(raw_uint8, heatmap_hb)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        axs[0].imshow(raw_uint8)
        axs[0].set_title(f"Conjunctiva ROI\nPatient {row['Patient_ID']}")
        axs[0].axis('off')

        im_hm = axs[1].imshow(heatmap_hb, cmap='jet')
        axs[1].set_title("Grad-CAM Attention Heatmap\n(Layer: out_relu)")
        axs[1].axis('off')
        plt.colorbar(im_hm, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].imshow(overlay_hb)
        axs[2].set_title(
            f"Ocular Overlay\nTrue Hb: {row['Hgb']:.1f} | Pred Hb: {p_hb:.1f} g/dL\nP(Anemic): {p_anemia:.2f}"
        )
        axs[2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f"gradcam_{row['Patient_ID']}.png"
        plt.savefig(save_path)
        plt.close()
        saved_paths.append(save_path)

    print(f"[Explainability] Generated {len(saved_paths)} Grad-CAM panels in {output_dir}")
    return saved_paths
