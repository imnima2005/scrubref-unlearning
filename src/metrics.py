import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    log_loss, roc_auc_score, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.models import output_to_probs
from src.data_loader import make_dataset, DataBundle, UnlearningSplit

def predict_probabilities(
    model: tf.keras.Model,
    items: np.ndarray,
    labels: np.ndarray,
    bundle: DataBundle,
    output_is_logits: bool,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = make_dataset(items, labels, bundle, batch_size, False, False, False, seed)
    probabilities: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []
    for xb, yb in ds:
        output = model(xb, training=False)
        probabilities.append(output_to_probs(output, output_is_logits).numpy())
        labels_out.append(yb.numpy())
    return np.concatenate(probabilities), np.concatenate(labels_out)

def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            ece += np.abs(correctness[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)

def brier_multiclass(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        mask = y_true == class_id
        if np.any(mask):
            result[class_id] = np.mean(y_pred[mask] == class_id)
    return result

def evaluate_utility(
    model: tf.keras.Model,
    bundle: DataBundle,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    probs, y_true = predict_probabilities(
        model, bundle.test_items, bundle.test_labels, bundle, output_is_logits, batch_size, seed
    )
    y_pred = probs.argmax(axis=1)
    forget_mask = np.isin(y_true, np.asarray(forget_classes))
    retain_mask = ~forget_mask
    top5 = np.mean([
        label in np.argpartition(row, -min(5, bundle.num_classes))[-min(5, bundle.num_classes):]
        for row, label in zip(probs, y_true)
    ])
    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "retained_accuracy": float(accuracy_score(y_true[retain_mask], y_pred[retain_mask])),
        "forgotten_accuracy": float(accuracy_score(y_true[forget_mask], y_pred[forget_mask])),
        "top5_accuracy": float(top5),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "nll": float(log_loss(y_true, np.clip(probs, 1e-7, 1.0), labels=np.arange(bundle.num_classes))),
        "brier": brier_multiclass(y_true, probs, bundle.num_classes),
        "ece_15": expected_calibration_error(y_true, probs, bins=15),
        "mean_global_forget_mass": float(np.mean(np.sum(probs[:, forget_classes], axis=1))),
        "mean_forget_test_true_class_confidence": float(
            np.mean(probs[np.where(forget_mask)[0], y_true[forget_mask]])
        ),
    }
    return metrics, per_class_accuracy(y_true, y_pred, bundle.num_classes)

def entropy_scores(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return -np.sum(probs * np.log(probs), axis=1)

def modified_entropy_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    rows = np.arange(len(labels))
    true_probs = probs[rows, labels]
    first = -(1.0 - true_probs) * np.log(true_probs)
    mask = np.ones_like(probs, dtype=bool)
    mask[rows, labels] = False
    remaining = -np.sum((probs * np.log(1.0 - probs)) * mask, axis=1)
    return first + remaining

def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    auc = roc_auc_score(y_true, scores)
    return float(max(auc, 1.0 - auc))

def learned_mia(
    member_probs: np.ndarray,
    member_labels: np.ndarray,
    nonmember_probs: np.ndarray,
    nonmember_labels: np.ndarray,
    repeats: int,
    seed: int,
) -> Tuple[float, float]:
    def attack_features(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        rows = np.arange(len(labels))
        true_conf = probs[rows, labels]
        entropy = entropy_scores(probs)
        mod_entropy = modified_entropy_scores(probs, labels)
        sorted_probs = np.sort(probs, axis=1)[:, -3:]
        return np.column_stack([true_conf, entropy, mod_entropy, sorted_probs])

    x = np.concatenate([
        attack_features(member_probs, member_labels),
        attack_features(nonmember_probs, nonmember_labels),
    ])
    y = np.concatenate([
        np.ones(len(member_probs), dtype=np.int64),
        np.zeros(len(nonmember_probs), dtype=np.int64),
    ])
    aucs: List[float] = []
    aps: List[float] = []
    for repeat in range(repeats):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, random_state=seed + repeat, stratify=y
        )
        attack = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        attack.fit(x_train, y_train)
        scores = attack.predict_proba(x_test)[:, 1]
        aucs.append(roc_auc_score(y_test, scores))
        aps.append(average_precision_score(y_test, scores))
    return float(np.mean(aucs)), float(np.mean(aps))

def evaluate_membership_inference(
    model: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    batch_size: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    member_probs, member_labels = predict_probabilities(
        model,
        split.forget_train_items,
        split.forget_train_labels,
        bundle,
        output_is_logits,
        batch_size,
        seed,
    )
    nonmember_probs, nonmember_labels = predict_probabilities(
        model,
        split.forget_test_items,
        split.forget_test_labels,
        bundle,
        output_is_logits,
        batch_size,
        seed,
    )
    n = min(len(member_probs), len(nonmember_probs))
    rng = np.random.default_rng(seed)
    member_idx = rng.choice(len(member_probs), n, replace=False)
    nonmember_idx = rng.choice(len(nonmember_probs), n, replace=False)
    member_probs = member_probs[member_idx]
    member_labels = member_labels[member_idx]
    nonmember_probs = nonmember_probs[nonmember_idx]
    nonmember_labels = nonmember_labels[nonmember_idx]

    probs = np.concatenate([member_probs, nonmember_probs])
    labels = np.concatenate([member_labels, nonmember_labels])
    membership = np.concatenate([np.ones(n), np.zeros(n)])
    rows = np.arange(len(labels))
    true_conf = probs[rows, labels]
    negative_loss = np.log(np.clip(true_conf, 1e-7, 1.0))
    negative_entropy = -entropy_scores(probs)
    negative_modified_entropy = -modified_entropy_scores(probs, labels)
    learned_auc, learned_ap = learned_mia(
        member_probs, member_labels, nonmember_probs, nonmember_labels, repeats, seed
    )
    return {
        "mia_confidence_auc": safe_auc(membership, true_conf),
        "mia_loss_auc": safe_auc(membership, negative_loss),
        "mia_entropy_auc": safe_auc(membership, negative_entropy),
        "mia_modified_entropy_auc": safe_auc(membership, negative_modified_entropy),
        "mia_learned_auc": learned_auc,
        "mia_learned_ap": learned_ap,
        "mia_member_count": float(n),
    }

def parameter_change_statistics(original: tf.keras.Model, changed: tf.keras.Model, tolerance: float = 1e-7) -> Dict[str, float]:
    original_weights = original.get_weights()
    changed_weights = changed.get_weights()
    if len(original_weights) != len(changed_weights):
        return {"changed_parameter_fraction": float("nan"), "relative_l2_parameter_change": float("nan")}
    total = 0
    changed_count = 0
    squared_delta = 0.0
    squared_original = 0.0
    for before, after in zip(original_weights, changed_weights):
        delta = after.astype(np.float64) - before.astype(np.float64)
        total += delta.size
        changed_count += int(np.count_nonzero(np.abs(delta) > tolerance))
        squared_delta += float(np.sum(delta ** 2))
        squared_original += float(np.sum(before.astype(np.float64) ** 2))
    return {
        "changed_parameter_fraction": float(changed_count / max(total, 1)),
        "relative_l2_parameter_change": float(math.sqrt(squared_delta) / max(math.sqrt(squared_original), 1e-12)),
    }

def peak_gpu_memory_mb() -> float:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return float(info["peak"] / (1024 ** 2))
    except Exception:
        return float("nan")