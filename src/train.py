"""
AnemiaSense — Multi-Task Model Training Orchestrator
Executes patient-level stratified splitting, two-stage transfer learning,
model checkpointing, and history logging.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from src.dataset_inspection import clean_metadata_and_extract
from src.preprocessing import ConjunctivaPreprocessor
from src.model import build_mobilenetv2_multitask, compile_multitask_model


def partition_dataset_patient_level(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions cohort with patient-level stratification on Country and Anemic state."""
    df['Stratum'] = df['Country'] + "_" + df['Anemic'].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['Stratum']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df['Stratum']
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Assert 0% patient leakage
    assert len(set(train_df['Patient_ID']).intersection(set(val_df['Patient_ID']))) == 0
    assert len(set(train_df['Patient_ID']).intersection(set(test_df['Patient_ID']))) == 0
    assert len(set(val_df['Patient_ID']).intersection(set(test_df['Patient_ID']))) == 0

    return train_df, val_df, test_df


def train_pipeline(
    workspace_dir: Path = Path(r"c:\anemia final"),
    config_path: Path = Path(r"c:\anemia final\configs\training_config.json")
) -> Tuple[keras.Model, pd.DataFrame, pd.DataFrame]:
    """Runs the complete training workflow from data loading to artifact export."""
    with open(config_path, "r") as f:
        config = json.load(f)

    seed = config.get("random_seed", 42)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_dir = workspace_dir / "data" / "extracted" / "dataset anemia"
    df = clean_metadata_and_extract(dataset_dir)

    train_df, val_df, test_df = partition_dataset_patient_level(df, seed=seed)

    # Save splits
    splits_dir = workspace_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train_metadata.csv", index=False)
    val_df.to_csv(splits_dir / "validation_metadata.csv", index=False)
    test_df.to_csv(splits_dir / "test_metadata.csv", index=False)

    preprocessor = ConjunctivaPreprocessor()

    # Preprocess & augment training set
    print("[Train] Preprocessing and augmenting training images...")
    X_train_list, y_train_cls_list, y_train_reg_list = [], [], []
    for idx, row in train_df.iterrows():
        im = preprocessor.process_image(row['Mask_PNG_Path'])
        c_val = float(row['Anemic'])
        r_val = float(row['Hgb'])

        # Raw normalized
        X_train_list.append(im)
        y_train_cls_list.append(c_val)
        y_train_reg_list.append(r_val)

        # Subtle flip
        flip = np.fliplr(im)
        X_train_list.append(flip)
        y_train_cls_list.append(c_val)
        y_train_reg_list.append(r_val)

        # Subtle brightness jitter
        jitter = np.clip(im * np.random.uniform(0.97, 1.03), -1.0, 1.0)
        X_train_list.append(jitter)
        y_train_cls_list.append(c_val)
        y_train_reg_list.append(r_val)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train_cls = np.array(y_train_cls_list, dtype=np.float32).reshape(-1, 1)
    y_train_reg = np.array(y_train_reg_list, dtype=np.float32).reshape(-1, 1)

    X_val = np.array([preprocessor.process_image(p) for p in val_df['Mask_PNG_Path']], dtype=np.float32)
    y_val_cls = val_df['Anemic'].values.reshape(-1, 1).astype(np.float32)
    y_val_reg = val_df['Hgb'].values.reshape(-1, 1).astype(np.float32)

    # Build model
    model, base_backbone = build_mobilenetv2_multitask()

    # Stage 1
    s1_cfg = config["stage1_transfer_learning"]
    compile_multitask_model(
        model,
        learning_rate=s1_cfg["learning_rate"],
        anemia_weight=s1_cfg["loss_weights"]["anemia_output"],
        hb_weight=s1_cfg["loss_weights"]["hb_output"]
    )

    checkpoints_dir = workspace_dir / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoints_dir / "anemia_multitask_best.keras"

    print("[Train] Executing Stage 1 Transfer Learning (Frozen Backbone)...")
    history_s1 = model.fit(
        X_train,
        {'anemia_output': y_train_cls, 'hb_output': y_train_reg},
        validation_data=(X_val, {'anemia_output': y_val_cls, 'hb_output': y_val_reg}),
        batch_size=config["batch_size"],
        epochs=s1_cfg["epochs"],
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_checkpoint),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )

    # Stage 2
    s2_cfg = config["stage2_fine_tuning"]
    base_backbone.trainable = True
    for layer in base_backbone.layers[:-s2_cfg["unfreeze_top_layers"]]:
        layer.trainable = False

    compile_multitask_model(
        model,
        learning_rate=s2_cfg["learning_rate"],
        anemia_weight=s2_cfg["loss_weights"]["anemia_output"],
        hb_weight=s2_cfg["loss_weights"]["hb_output"]
    )

    print("[Train] Executing Stage 2 Fine-Tuning (Top Layers Unfrozen)...")
    history_s2 = model.fit(
        X_train,
        {'anemia_output': y_train_cls, 'hb_output': y_train_reg},
        validation_data=(X_val, {'anemia_output': y_val_cls, 'hb_output': y_val_reg}),
        batch_size=config["batch_size"],
        epochs=s2_cfg["epochs"],
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_checkpoint),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=s2_cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=s2_cfg["reduce_lr_factor"],
                patience=s2_cfg["reduce_lr_patience"],
                min_lr=s2_cfg["min_learning_rate"],
                verbose=1
            )
        ],
        verbose=1
    )

    if best_checkpoint.exists():
        try:
            model = keras.models.load_model(str(best_checkpoint))
            print("[Train] Loaded best validation checkpoint weights.")
        except Exception as e:
            print(f"[Train] Checkpoint load note: {e}")

    # Export final model
    exported_dir = workspace_dir / "models" / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = exported_dir / "anemia_mobilenetv2_multitask.keras"
    model.save(final_model_path)
    print(f"[Train] Final production model exported to: {final_model_path}")

    # Log unified history
    unified_history = {}
    for k in history_s1.history.keys():
        unified_history[k] = history_s1.history[k] + history_s2.history[k]

    history_df = pd.DataFrame(unified_history)
    metrics_dir = workspace_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(metrics_dir / "training_history.csv", index_label='epoch')

    return model, history_df, test_df


if __name__ == "__main__":
    model, hist, test_df = train_pipeline()
    print("Training pipeline executed successfully.")
