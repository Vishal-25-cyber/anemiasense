"""
AnemiaSense — Multi-Task MobileNetV2 Model Architecture
Defines the dual-head transfer learning architecture combining
binary anemia screening and continuous hemoglobin regression in a unified computational graph.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mobilenetv2_multitask(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    dense_units: int = 128
) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the multi-task MobileNetV2 transfer learning model.

    Returns:
        model: Unified Keras Model with 'anemia_output' and 'hb_output' heads.
        base_backbone: Pretrained MobileNetV2 backbone for layer-wise freeze/unfreeze control.
    """
    # 1. ImageNet Pretrained MobileNetV2 Feature Extractor
    base_backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_backbone.trainable = False

    # 2. Unified Graph Wiring
    inp = base_backbone.input
    feats = base_backbone.output

    # 3. Global Pooling & Shared Latent Representation
    gap = layers.GlobalAveragePooling2D(name='global_pooling')(feats)
    shared = layers.Dense(dense_units, activation='relu', name='shared_dense')(gap)
    drop = layers.Dropout(dropout_rate, name='dropout')(shared)

    # 4. Multi-Task Output Heads
    anemia_out = layers.Dense(1, activation='sigmoid', name='anemia_output')(drop)
    hb_out = layers.Dense(1, activation='linear', name='hb_output')(drop)

    model = keras.Model(inputs=inp, outputs=[anemia_out, hb_out], name='AnemiaSense_MobileNetV2_MultiTask')
    return model, base_backbone


def compile_multitask_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    anemia_weight: float = 1.0,
    hb_weight: float = 0.20
) -> keras.Model:
    """Compiles the multi-task model with balanced classification and Huber regression losses."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'anemia_output': 'binary_crossentropy',
            'hb_output': keras.losses.Huber(delta=1.0)
        },
        loss_weights={
            'anemia_output': anemia_weight,
            'hb_output': hb_weight
        },
        metrics={
            'anemia_output': ['binary_accuracy'],
            'hb_output': ['mae']
        }
    )
    return model
