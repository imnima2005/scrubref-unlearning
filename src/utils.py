from pathlib import Path
import hashlib
import ctypes,gc,os,random
import json
import numpy as np
import tensorflow as tf
from typing import Dict

from src.models import custom_objects

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
    """Release TensorFlow and Python memory where possible."""
    if clear_session:
        tf.keras.backend.clear_session()

    gc.collect()

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
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

def method_model_path(run_dir: Path, method: str) -> Path:
    return run_dir / "models" / f"{method}.keras"

def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def load_saved_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects=custom_objects(), compile=False, safe_mode=False)
