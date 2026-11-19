from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import ExperimentConfig
from src.data_loader import (
    DataBundle, make_dataset
)

from src.models import (
    build_resnet18,
    set_all_trainable,
    set_tail_trainable,
    custom_objects
)


def load_or_train_original(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    seed: int,
    run_dir: Path,
) -> Tuple[tf.keras.Model, bool, str]:
    checkpoint = Path(cfg.checkpoint_paths[cfg.dataset])
    if checkpoint.exists():
        model = tf.keras.models.load_model(checkpoint, custom_objects=custom_objects(), compile=False, safe_mode=False)
        source = str(checkpoint)
    else:
        model = build_resnet18(bundle.input_shape, bundle.num_classes)
        sample_ds = make_dataset(bundle.train_items[:16], bundle.train_labels[:16], bundle, 16, False, False, False, seed)
        sample_x, _ = next(iter(sample_ds))
        output_is_logits = infer_output_is_logits(model, sample_x, True)
        model = train_supervised(
            model,
            bundle.train_items,
            bundle.train_labels,
            bundle,
            output_is_logits,
            cfg.original_epochs,
            cfg.batch_size,
            cfg.original_lr,
            seed,
            validation=None,
            train_all=True,
        )
        source = str(f"./checkpoints/{cfg.dataset}_trained_original.keras")
        model.save(source)
    sample_ds = make_dataset(bundle.train_items[:16], bundle.train_labels[:16], bundle, 16, False, False, False, seed)
    sample_x, _ = next(iter(sample_ds))
    output_is_logits = infer_output_is_logits(model, sample_x, cfg.output_is_logits)
    return model, output_is_logits, source

def compile_supervised(model: tf.keras.Model, output_is_logits: bool, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=output_is_logits),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

def train_supervised(
    model: tf.keras.Model,
    items: np.ndarray,
    labels: np.ndarray,
    bundle: DataBundle,
    output_is_logits: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    train_all: bool = True,
    tail_layers: int = 0,
) -> tf.keras.Model:
    if train_all:
        set_all_trainable(model, True, freeze_batch_norm=False)
    else:
        set_tail_trainable(model, bundle.num_classes, tail_layers)
    compile_supervised(model, output_is_logits, learning_rate)
    train_ds = make_dataset(items, labels, bundle, batch_size, True, True, False, seed)
    validation_ds = None
    callbacks: List[tf.keras.callbacks.Callback] = []
    if validation is not None and len(validation[0]) > 0:
        validation_ds = make_dataset(validation[0], validation[1], bundle, batch_size, False, False, False, seed)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(3, epochs // 10),
            restore_best_weights=True,
        ))
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs, verbose=2, callbacks=callbacks)
    return model

def infer_output_is_logits(model: tf.keras.Model, sample_batch: tf.Tensor, override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    last_layer = model.layers[-1]
    activation = getattr(last_layer, "activation", None)
    activation_name = getattr(activation, "__name__", "")
    if activation_name == "softmax":
        return False
    if activation_name == "linear":
        return True
    output = tf.cast(model(sample_batch[:8], training=False), tf.float32)
    nonnegative = bool(tf.reduce_all(output >= 0.0).numpy())
    sums_close = bool(tf.reduce_all(tf.abs(tf.reduce_sum(output, axis=-1) - 1.0) < 1e-3).numpy())
    return not (nonnegative and sums_close)
