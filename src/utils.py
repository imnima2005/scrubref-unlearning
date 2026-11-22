from pathlib import Path
import hashlib
import ctypes,gc,os,random
import json
from typing import Dict

import tensorflow as tf
import pandas as pd
import numpy as np

from dataclasses import asdict
from datetime import datetime, timezone

from src.models import custom_objects
from src.config import ExperimentConfig

# =============================================================================
# System, Determinism & Memory Utilities
# =============================================================================
def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def set_global_determinism(seed: int) -> None:
    """Configure random seeds and deterministic TensorFlow operations."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def release_memory(clear_session: bool = True) -> None:
    """Release Python, TensorFlow graph, and allocator caches between stages."""
    if clear_session:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass

def reset_tensorflow(seed: int) -> None:
    """Clear the current TensorFlow state and reset its execution settings."""
    release_memory(clear_session=True)
    set_global_determinism(seed)
    tf.keras.mixed_precision.set_global_policy("float32")

def array_memory_mb(*arrays: np.ndarray) -> float:
    """Return the total memory usage of NumPy arrays in megabytes."""
    total_bytes = sum(
        array.nbytes
        for array in arrays
        if isinstance(array, np.ndarray)
    )
    return float(total_bytes / (1024 ** 2))

# =============================================================================
# Model & File Storage Utilities
# =============================================================================
def method_model_path(run_dir: Path, method: str) -> Path:
    return run_dir / "models" / f"{method}.keras"

def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def load_saved_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects=custom_objects(), compile=False, safe_mode=False)

# =============================================================================
# ImageWoof & Advanced Experiment Path Utilities
# =============================================================================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)

def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def config_fingerprint(cfg: ExperimentConfig) -> str:
    payload = asdict(cfg)
    payload.pop("results_root", None)
    payload.pop("data_root", None)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]

def experiment_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.results_root) / cfg.dataset / f"{cfg.run_mode}_{config_fingerprint(cfg)}"

def seed_dir(cfg: ExperimentConfig, seed: int) -> Path:
    return experiment_dir(cfg) / f"seed_{seed}"

def stage_status_path(cfg: ExperimentConfig, seed: int, stage: str) -> Path:
    return seed_dir(cfg, seed) / "status" / f"{stage}.json"

def set_stage_status(cfg: ExperimentConfig, seed: int, stage: str, state: str, **extra: object) -> None:
    payload = {
        "dataset": cfg.dataset,
        "seed": seed,
        "stage": stage,
        "state": state,
        "updated_at": utc_now(),
        "config_fingerprint": config_fingerprint(cfg),
        **extra,
    }
    atomic_write_json(stage_status_path(cfg, seed, stage), payload)