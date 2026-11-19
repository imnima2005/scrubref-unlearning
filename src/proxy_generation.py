import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Sequence
from pathlib import Path
import json

from src.data_loader import (
    DataBundle,
    ExperimentConfig
)

from src.utils import (
    set_global_determinism,
)

from src.models import (
    output_to_logits,
    output_to_probs
)

def balanced_proxy_targets(forget_classes: Sequence[int], count: int) -> np.ndarray:
    classes = np.asarray(forget_classes, dtype=np.int64)
    return np.resize(classes, count)

def generate_class_proxies(
    teacher: tf.keras.Model,
    bundle: DataBundle,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as cached:
            images = cached["images"].astype(np.float32, copy=False)
            labels = cached["labels"].astype(np.int64, copy=False)
            metadata = json.loads(str(cached["metadata"].item()))
        return images, labels, metadata

    set_global_determinism(seed)
    teacher.trainable = False
    all_images: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    target_probabilities: List[float] = []
    remaining = cfg.proxy_count
    target_schedule = balanced_proxy_targets(forget_classes, cfg.proxy_count)
    cursor = 0

    while remaining > 0:
        current = min(cfg.proxy_batch_size, remaining)
        targets = tf.convert_to_tensor(target_schedule[cursor:cursor + current], dtype=tf.int64)
        images = tf.Variable(tf.random.uniform((current,) + bundle.input_shape, 0.0, 255.0, seed=seed + cursor))
        optimizer = tf.keras.optimizers.Adam(cfg.proxy_lr)
        for _ in range(cfg.proxy_steps):
            with tf.GradientTape() as tape:
                output = teacher(images, training=False)
                logits = output_to_logits(output, output_is_logits)
                class_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
                )
                normalized = images / 255.0
                tv_loss = tf.reduce_mean(tf.image.total_variation(normalized))
                l2_loss = tf.reduce_mean(tf.square(normalized - 0.5))
                loss = class_loss + cfg.proxy_tv_weight * tv_loss + cfg.proxy_l2_weight * l2_loss
            gradients = tape.gradient(loss, images)
            optimizer.apply_gradients([(gradients, images)])
            images.assign(tf.clip_by_value(images, 0.0, 255.0))
        probs = output_to_probs(teacher(images, training=False), output_is_logits).numpy()
        target_probabilities.extend(probs[np.arange(current), targets.numpy()].tolist())
        all_images.append(images.numpy().astype(np.float32))
        all_labels.append(targets.numpy())
        cursor += current
        remaining -= current

    proxy_images = np.concatenate(all_images)
    proxy_labels = np.concatenate(all_labels)
    metadata = {
        "mean_target_probability": float(np.mean(target_probabilities)),
        "min_target_probability": float(np.min(target_probabilities)),
        "max_target_probability": float(np.max(target_probabilities)),
        "count": float(len(proxy_images)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, images=proxy_images, labels=proxy_labels, metadata=json.dumps(metadata))
    return proxy_images, proxy_labels, metadata

def generate_unsir_error_maximizing_noise(
    teacher: tf.keras.Model,
    bundle: DataBundle,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    count: int,
    steps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    set_global_determinism(seed)
    targets_np = balanced_proxy_targets(forget_classes, count)
    targets = tf.convert_to_tensor(targets_np, dtype=tf.int64)
    noise = tf.Variable(tf.random.normal((count,) + bundle.input_shape, mean=127.5, stddev=40.0, seed=seed))
    optimizer = tf.keras.optimizers.Adam(1.0)
    teacher.trainable = False
    for _ in range(steps):
        with tf.GradientTape() as tape:
            output = teacher(noise, training=False)
            logits = output_to_logits(output, output_is_logits)
            ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))
            regularizer = 1e-6 * tf.reduce_mean(tf.square(noise - 127.5))
            objective = -ce + regularizer
        gradient = tape.gradient(objective, noise)
        optimizer.apply_gradients([(gradient, noise)])
        noise.assign(tf.clip_by_value(noise, 0.0, 255.0))
    return noise.numpy().astype(np.float32), targets_np