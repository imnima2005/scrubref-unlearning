import numpy as np
import tensorflow as tf
from typing import Sequence

from src.data_loader import (
    DataBundle,
    UnlearningSplit
)

from src.models import (
    clone_with_weights,
    fresh_clone
)

from src.config import ExperimentConfig
from src.models import (
    classifier_layer
)
from src.training import train_supervised

def logit_clamp_baseline(
    original: tf.keras.Model,
    bundle: DataBundle,
    forget_classes: Sequence[int],
    shift: float,
) -> tf.keras.Model:
    model = clone_with_weights(original)
    classifier = classifier_layer(model, bundle.num_classes)
    weights = classifier.get_weights()
    kernel = weights[0].copy()
    kernel[:, list(forget_classes)] = 0.0
    if len(weights) == 2:
        bias = weights[1].copy()
        bias[list(forget_classes)] -= shift
        classifier.set_weights([kernel, bias])
    else:
        classifier.set_weights([kernel])
    return model

def head_reset_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    model = clone_with_weights(original)
    classifier = classifier_layer(model, bundle.num_classes)
    weights = classifier.get_weights()
    kernel = tf.keras.initializers.GlorotUniform(seed=seed)(weights[0].shape).numpy()
    if len(weights) == 2:
        bias = np.zeros_like(weights[1])
        classifier.set_weights([kernel, bias])
    else:
        classifier.set_weights([kernel])
    return train_supervised(
        model,
        split.retain_train_items,
        split.retain_train_labels,
        bundle,
        output_is_logits,
        cfg.baseline_epochs,
        cfg.batch_size,
        cfg.baseline_lr,
        seed,
        validation=(split.retain_val_items, split.retain_val_labels),
        train_all=False,
        tail_layers=0,
    )

def retain_finetune_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    model = clone_with_weights(original)
    return train_supervised(
        model,
        split.retain_train_items,
        split.retain_train_labels,
        bundle,
        output_is_logits,
        cfg.baseline_epochs,
        cfg.batch_size,
        cfg.baseline_lr,
        seed,
        validation=(split.retain_val_items, split.retain_val_labels),
        train_all=False,
        tail_layers=cfg.representation_tail_layers,
    )

def retrain_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    initial_path = cfg.initial_weights_paths.get(cfg.dataset, "")
    model = fresh_clone(original, initial_path)
    return train_supervised(
        model,
        split.retain_train_items,
        split.retain_train_labels,
        bundle,
        output_is_logits,
        cfg.retrain_epochs,
        cfg.batch_size,
        cfg.retrain_lr,
        seed,
        validation=(split.retain_val_items, split.retain_val_labels),
        train_all=True,
    )