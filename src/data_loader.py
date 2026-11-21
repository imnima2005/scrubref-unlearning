import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config import ExperimentConfig


ArrayLike = Union[np.ndarray, Sequence[str]]

@dataclass
class DataBundle:
    name: str
    train_items: np.ndarray
    train_labels: np.ndarray
    test_items: np.ndarray
    test_labels: np.ndarray
    class_names: List[str]
    image_size: Tuple[int, int]
    channels: int = 3
    path_mode: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.image_size + (self.channels,)


@dataclass
class UnlearningSplit:
    retain_train_items: np.ndarray
    retain_train_labels: np.ndarray
    retain_val_items: np.ndarray
    retain_val_labels: np.ndarray
    forget_train_items: np.ndarray
    forget_train_labels: np.ndarray
    retain_test_items: np.ndarray
    retain_test_labels: np.ndarray
    forget_test_items: np.ndarray
    forget_test_labels: np.ndarray


def ensure_imagewoof(root: str) -> Path:
    root_path = Path(root)
    dataset_dir = root_path / "imagewoof2-320"
    if dataset_dir.exists():
        return dataset_dir
    if not CFG.download_imagewoof:
        raise FileNotFoundError(f"ImageWoof not found at {dataset_dir}")
    root_path.mkdir(parents=True, exist_ok=True)
    archive = root_path / "imagewoof2-320.tgz"
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root_path)
    return dataset_dir

def list_directory_split(split_dir: Path, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    paths: List[str] = []
    labels: List[int] = []
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    for class_id, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_suffixes)
        paths.extend(str(p) for p in files)
        labels.extend([class_id] * len(files))
    return np.asarray(paths, dtype=str), np.asarray(labels, dtype=np.int64)

def load_directory_bundle(name: str, root: str, image_size: Tuple[int, int]) -> DataBundle:
    root_path = Path(root)
    train_dir = root_path / "train"
    val_dir = root_path / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected {train_dir} and {val_dir}")
    class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    train_items, train_labels = list_directory_split(train_dir, class_names)
    test_items, test_labels = list_directory_split(val_dir, class_names)
    if len(train_items) == 0 or len(test_items) == 0:
        raise RuntimeError(f"No images were found under {root_path}")
    return DataBundle(
        name=name,
        train_items=train_items,
        train_labels=train_labels,
        test_items=test_items,
        test_labels=test_labels,
        class_names=class_names,
        image_size=image_size,
        path_mode=True,
    )

def load_dataset(cfg: ExperimentConfig) -> DataBundle:
    if cfg.dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return DataBundle(
            name="cifar10",
            train_items=x_train,
            train_labels=y_train.reshape(-1).astype(np.int64),
            test_items=x_test,
            test_labels=y_test.reshape(-1).astype(np.int64),
            class_names=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            image_size=(32, 32),
            path_mode=False,
        )
    if cfg.dataset == "cifar100":
        import pickle

        dataset_dir = Path(
            "./data/cifar100/cifar-100-python" if IN_COLAB else REPO_DIR / "data/cifar100/cifar-100-python"
        )

        train_file = dataset_dir / "train"
        test_file = dataset_dir / "test"
        meta_file = dataset_dir / "meta"

        required_files = [train_file, test_file, meta_file]
        missing_files = [str(path) for path in required_files if not path.exists()]

        if missing_files:
            raise FileNotFoundError(
                "Missing CIFAR-100 files:\n" + "\n".join(missing_files)
            )

        def load_cifar100_pickle(path: Path):
            with path.open("rb") as file:
                payload = pickle.load(file, encoding="bytes")

            data = payload.get(b"data", payload.get("data"))
            labels = payload.get(
                b"fine_labels",
                payload.get("fine_labels"),
            )

            if data is None or labels is None:
                raise ValueError(f"Invalid CIFAR-100 file: {path}")

            images = np.asarray(data, dtype=np.uint8)
            images = images.reshape(-1, 3, 32, 32)
            images = images.transpose(0, 2, 3, 1)

            labels = np.asarray(labels, dtype=np.int64)

            return images, labels

        x_train, y_train = load_cifar100_pickle(train_file)
        x_test, y_test = load_cifar100_pickle(test_file)

        with meta_file.open("rb") as file:
            meta = pickle.load(file, encoding="bytes")

        raw_class_names = meta.get(
            b"fine_label_names",
            meta.get("fine_label_names"),
        )

        class_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in raw_class_names
        ]

        print("Loaded CIFAR-100 directly from Google Drive")
        print("Train:", x_train.shape, y_train.shape)
        print("Test:", x_test.shape, y_test.shape)

        return DataBundle(
            name="cifar100",
            train_items=x_train,
            train_labels=y_train,
            test_items=x_test,
            test_labels=y_test,
            class_names=class_names,
            image_size=(32, 32),
            path_mode=False,
        )
    if cfg.dataset == "imagewoof":
        dataset_dir = ensure_imagewoof(cfg.data_root)
        bundle = load_directory_bundle("imagewoof", str(dataset_dir), (224, 224))
        configured = cfg.forget_classes_by_dataset["imagewoof"]
        if configured == (0,) and "n02086240" in bundle.class_names:
            cfg.forget_classes_by_dataset["imagewoof"] = (bundle.class_names.index("n02086240"),)
        return bundle
    if cfg.dataset == "imagenet100":
        return load_directory_bundle("imagenet100", cfg.imagenet100_root, (224, 224))
    raise ValueError(f"Unsupported dataset: {cfg.dataset}")

def make_unlearning_split(
    bundle: DataBundle,
    forget_classes: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> UnlearningSplit:
    forget_classes_array = np.asarray(forget_classes, dtype=np.int64)
    train_forget_mask = np.isin(bundle.train_labels, forget_classes_array)
    test_forget_mask = np.isin(bundle.test_labels, forget_classes_array)

    retain_train_pool_indices = np.flatnonzero(~train_forget_mask)
    forget_train_indices = np.flatnonzero(train_forget_mask)
    retain_test_indices = np.flatnonzero(~test_forget_mask)
    forget_test_indices = np.flatnonzero(test_forget_mask)

    retain_train_indices, retain_val_indices = train_test_split(
        retain_train_pool_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=bundle.train_labels[retain_train_pool_indices],
    )

    split = UnlearningSplit(
        retain_train_items=bundle.train_items[retain_train_indices],
        retain_train_labels=bundle.train_labels[retain_train_indices],
        retain_val_items=bundle.train_items[retain_val_indices],
        retain_val_labels=bundle.train_labels[retain_val_indices],
        forget_train_items=bundle.train_items[forget_train_indices],
        forget_train_labels=bundle.train_labels[forget_train_indices],
        retain_test_items=bundle.test_items[retain_test_indices],
        retain_test_labels=bundle.test_labels[retain_test_indices],
        forget_test_items=bundle.test_items[forget_test_indices],
        forget_test_labels=bundle.test_labels[forget_test_indices],
    )

    print(
        "RAM-safe split prepared:",
        f"retain_train={len(split.retain_train_labels)},",
        f"retain_val={len(split.retain_val_labels)},",
        f"forget_train={len(split.forget_train_labels)},",
        f"retain_test={len(split.retain_test_labels)},",
        f"forget_test={len(split.forget_test_labels)}",
    )
    return split

def decode_item(item: tf.Tensor, image_size: Tuple[int, int], path_mode: bool) -> tf.Tensor:
    if path_mode:
        raw = tf.io.read_file(item)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size, antialias=True)
    else:
        image = tf.cast(item, tf.float32)
        if tuple(image.shape[:2]) != image_size:
            image = tf.image.resize(image, image_size, antialias=True)
    image.set_shape(image_size + (3,))
    return tf.cast(image, tf.float32)

def augment_image(image: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    if image_size[0] <= 64:
        pad = 4
        image = tf.image.resize_with_crop_or_pad(image, image_size[0] + 2 * pad, image_size[1] + 2 * pad)
        image = tf.image.random_crop(image, image_size + (3,))
    else:
        enlarged = (int(image_size[0] * 1.08), int(image_size[1] * 1.08))
        image = tf.image.resize(image, enlarged, antialias=True)
        image = tf.image.random_crop(image, image_size + (3,))
    image = tf.image.random_brightness(image, max_delta=8.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 255.0)

def make_dataset(
    items: np.ndarray,
    labels: np.ndarray,
    bundle: DataBundle,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    repeat: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    if len(items) != len(labels):
        raise ValueError(f"items/labels length mismatch: {len(items)} != {len(labels)}")

    labels_array = np.asarray(labels, dtype=np.int64)
    ds = tf.data.Dataset.from_tensor_slices((items, labels_array))
    if shuffle:
        ds = ds.shuffle(
            min(len(items), CFG.shuffle_buffer),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    if repeat:
        ds = ds.repeat()

    def mapper(item: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = decode_item(item, bundle.image_size, bundle.path_mode)
        if augment:
            image = augment_image(image, bundle.image_size)
        return image, tf.cast(label, tf.int64)

    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    try:
        options.autotune.enabled = False
        options.threading.private_threadpool_size = CFG.data_private_threadpool_size
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass

    ds = ds.with_options(options)
    ds = ds.map(mapper, num_parallel_calls=CFG.data_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(CFG.data_prefetch_batches)

def materialize_items(
    items: np.ndarray,
    labels: np.ndarray,
    bundle: DataBundle,
    limit: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(items) > limit:
        indices = rng.choice(len(items), size=limit, replace=False)
        items = items[indices]
        labels = labels[indices]
    ds = make_dataset(items, labels, bundle, CFG.eval_batch_size, False, False, False, seed)
    images_out: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []
    for xb, yb in ds:
        images_out.append(xb.numpy())
        labels_out.append(yb.numpy())
    return np.concatenate(images_out), np.concatenate(labels_out)

# Specific to ImageWoof notebook
_RUNTIME_BATCH_MESSAGES: set = set()

def effective_runtime_batch_size(
    bundle: DataBundle,
    requested_batch_size: int,
    training: bool,
) -> int:
    """Return a conservative batch size without changing the experiment fingerprint."""
    requested = max(1, int(requested_batch_size))
    method = globals().get("ACTIVE_METHOD_NAME")
    base_name = str(bundle.name).replace("_materialized", "")

    if base_name == "cifar100":
        if training:
            cap = 4 if method in {"unsir", "bad_teacher", "scrub"} else 8
        else:
            cap = 8
    elif base_name in {"imagewoof", "imagenet100"}:
        if training:
            cap = 2 if method in {"unsir", "bad_teacher", "scrub"} else 4
        else:
            cap = 4
    elif base_name == "cifar10":
        if training:
            cap = 16 if method in {"unsir", "bad_teacher", "scrub"} else 32
        else:
            cap = 64
    else:
        cap = 4 if training else 8

    effective = min(requested, cap)
    message_key = (base_name, method, requested, effective, training)
    if effective != requested and message_key not in _RUNTIME_BATCH_MESSAGES:
        phase = "training" if training else "evaluation"
        method_text = f", method={method}" if method else ""
        print(
            f"Runtime GPU safety cap: {base_name} {phase} batch "
            f"{requested} -> {effective}{method_text}"
        )
        _RUNTIME_BATCH_MESSAGES.add(message_key)
    return effective