from dataclasses import dataclass, field, fields
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    run_mode: str = "publication"
    dataset: str = "cifar100"
    data_root: str = "./data"
    results_root: str = "./results"

    checkpoint_paths: Dict[str, str] = field(default_factory=lambda: {
        "cifar10": "./checkpoints/cifar10_cnn_best.keras",
        "cifar100": "./checkpoints/cifar100_efficientnet_85.keras",
        "imagewoof": "./checkpoints/imagewoof_effb0_fixed.keras",
        "imagenet100": "./checkpoints/imagenet100_model.keras",
    })
    initial_weights_paths: Dict[str, str] = field(default_factory=lambda: {
        "cifar10": "",
        "cifar100": "",
        "imagewoof": "",
        "imagenet100": "",
    })

    imagenet100_root: str = "/content/drive/MyDrive/datasets/imagenet100"
    forget_classes_by_dataset: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "cifar10": (5,),
        "cifar100": (64,),
        "imagewoof": (0,),
        "imagenet100": (0,),
    })

    seeds: Tuple[int, ...] = (42, 52, 62)
    output_is_logits: Optional[bool] = None
    fallback_architecture: str = "resnet18"

    batch_size: int = 64
    eval_batch_size: int = 128
    data_parallel_calls: int = 2
    data_prefetch_batches: int = 1
    data_private_threadpool_size: int = 2
    shuffle_buffer: int = 4096
    original_epochs: int = 100
    retrain_epochs: int = 100
    baseline_epochs: int = 8
    scrub_epochs: int = 6
    refine_epochs: int = 8
    clamp_epochs: int = 1

    original_lr: float = 1e-3
    retrain_lr: float = 1e-3
    baseline_lr: float = 1e-3
    scrub_lr: float = 2e-3
    refine_lr: float = 1e-3

    proxy_count: int = 1024
    proxy_steps: int = 250
    proxy_batch_size: int = 32
    proxy_lr: float = 1.0
    proxy_tv_weight: float = 2e-5
    proxy_l2_weight: float = 1e-6

    mixup_alpha: float = 0.2
    kd_temperature: float = 3.0
    retain_ce_weight: float = 1.0
    retain_kd_weight: float = 0.8
    retain_feature_weight: float = 0.2
    forget_output_weight: float = 1.0
    forget_entropy_weight: float = 0.15
    forget_feature_weight: float = 0.5

    head_tail_layers: int = 0
    representation_tail_layers: int = 12
    gradient_clip_norm: float = 5.0
    clamp_threshold: float = 0.05
    enable_main_method_clamp: bool = False
    enable_main_method_logit_surgery: bool = False
    logit_surgery_shift: float = 6.0

    retrain_validation_fraction: float = 0.1
    probe_train_limit: int = 10000
    probe_test_limit: int = 5000
    cka_sample_limit: int = 1500
    mia_attack_repeats: int = 5

    ssd_fisher_batches: int = 40
    ssd_selection_weighting: float = 10.0
    ssd_dampening_constant: float = 1.0
    unsir_noise_steps: int = 120
    unsir_impair_lr: float = 0.02
    unsir_repair_lr: float = 0.01

    enabled_methods: Tuple[str, ...] = (
        "original",
        "retrained",
        "logit_clamp",
        "head_reset",
        "retain_finetune",
        "scrubref_head",
        "scrubref_rep",
        "unsir",
        "bad_teacher",
        "scrub",
        "ssd",
    )

    representation_methods: Tuple[str, ...] = (
        "original",
        "retrained",
        "logit_clamp",
        "scrubref_head",
        "scrubref_rep",
        "unsir",
        "bad_teacher",
        "scrub",
        "ssd",
    )

    save_models: bool = True
    download_imagewoof: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a complete or partial YAML configuration with strict key checking."""
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("YAML configuration root must be a mapping.")
        known_fields = {item.name for item in fields(cls)}
        unknown_keys = set(data) - known_fields
        if unknown_keys:
            names = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ValueError(f"Unknown configuration key(s): {names}")
        data = dict(data)
        for name in ("seeds", "enabled_methods", "representation_methods"):
            if name in data and isinstance(data[name], list):
                data[name] = tuple(data[name])
        if "forget_classes_by_dataset" in data:
            mapping = data["forget_classes_by_dataset"]
            if not isinstance(mapping, dict):
                raise ValueError("forget_classes_by_dataset must be a mapping.")
            data["forget_classes_by_dataset"] = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in mapping.items()
            }
        return cls(**data).apply_mode()

    def apply_mode(self) -> "ExperimentConfig":
        if self.run_mode == "debug":
            self.seeds = (42,)
            self.original_epochs = min(self.original_epochs, 3)
            self.retrain_epochs = min(self.retrain_epochs, 3)
            self.baseline_epochs = min(self.baseline_epochs, 2)
            self.scrub_epochs = min(self.scrub_epochs, 2)
            self.refine_epochs = min(self.refine_epochs, 2)
            self.proxy_count = min(self.proxy_count, 128)
            self.proxy_steps = min(self.proxy_steps, 20)
            self.probe_train_limit = min(self.probe_train_limit, 3000)
            self.probe_test_limit = min(self.probe_test_limit, 2000)
            self.cka_sample_limit = min(self.cka_sample_limit, 750)
            self.ssd_fisher_batches = min(self.ssd_fisher_batches, 5)
            self.unsir_noise_steps = min(self.unsir_noise_steps, 15)
        elif self.run_mode != "publication":
            raise ValueError("run_mode must be 'publication' or 'debug'.")
        return self


CFG = ExperimentConfig().apply_mode()
