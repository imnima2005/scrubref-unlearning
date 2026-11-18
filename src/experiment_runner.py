def method_model_path(run_dir: Path, method: str) -> Path:
    return run_dir / "models" / f"{method}.keras"


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def build_method(
    method: str,
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    metadata: Dict[str, float] = {}
    if method == "original":
        return clone_with_weights(original), metadata
    if method == "retrained":
        return retrain_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    if method == "logit_clamp":
        return logit_clamp_baseline(original, bundle, forget_classes, shift=20.0), metadata
    if method == "head_reset":
        return head_reset_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    if method == "retain_finetune":
        return retain_finetune_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    if method == "scrubref_head":
        return scrubref_train(
            original, split, bundle, proxies, proxy_labels, forget_classes, output_is_logits, cfg, seed, False
        )
    if method == "scrubref_rep":
        return scrubref_train(
            original, split, bundle, proxies, proxy_labels, forget_classes, output_is_logits, cfg, seed, True
        )
    if method == "unsir":
        return unsir_baseline(original, split, bundle, forget_classes, output_is_logits, cfg, seed), metadata
    if method == "bad_teacher":
        return bad_teacher_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    if method == "scrub":
        return scrub_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    if method == "ssd":
        return ssd_baseline(original, split, bundle, output_is_logits, cfg, seed), metadata
    raise ValueError(f"Unknown method: {method}")


def load_saved_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects=custom_objects(), compile=False, safe_mode=False)


def run_seed(
    cfg: ExperimentConfig,
    seed: int,
    force_rerun: bool = False,
) -> Dict[str, Path]:
    reset_tensorflow(seed)
    bundle = load_dataset(cfg)
    print(
        "Dataset storage:",
        bundle.train_items.dtype,
        f"{array_memory_mb(bundle.train_items, bundle.test_items):.2f} MB",
    )
    if not bundle.path_mode and bundle.train_items.dtype != np.uint8:
        raise RuntimeError(
            "CIFAR arrays must remain uint8 in RAM. Re-run the RAM-safe dataset cell."
        )
    forget_classes = cfg.forget_classes_by_dataset[cfg.dataset]
    split = make_unlearning_split(bundle, forget_classes, cfg.retrain_validation_fraction, seed)
    print(
        "Bundle + split image arrays:",
        f"{array_memory_mb(bundle.train_items, bundle.test_items, split.retain_train_items, split.retain_val_items, split.forget_train_items, split.retain_test_items, split.forget_test_items):.2f} MB",
    )

    run_dir = Path(cfg.results_root) / cfg.dataset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", asdict(cfg))

    original, output_is_logits, original_source = load_or_train_original(cfg, bundle, seed, run_dir)
    original_run_path = method_model_path(run_dir, "original")
    if not original_run_path.exists() or force_rerun:
        original.save(original_run_path)

    proxy_path = run_dir / "proxies.npz"
    proxies, proxy_labels, proxy_metadata = generate_class_proxies(
        original,
        bundle,
        forget_classes,
        output_is_logits,
        cfg,
        seed,
        proxy_path,
    )
    save_json(run_dir / "proxy_metadata.json", proxy_metadata)

    utility_rows: List[Dict[str, object]] = []
    mia_rows: List[Dict[str, object]] = []
    runtime_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    for method in cfg.enabled_methods:
        reset_tensorflow(seed)
        original = load_saved_model(original_run_path)
        model_path = method_model_path(run_dir, method)
        method_metadata_path = run_dir / "models" / f"{method}_metadata.json"
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass
        start = time.perf_counter()
        metadata: Dict[str, float] = {}
        if model_path.exists() and not force_rerun:
            model = load_saved_model(model_path)
            reused = 1.0
            if method_metadata_path.exists():
                stored_metadata = json.loads(method_metadata_path.read_text(encoding="utf-8"))
                metadata = {
                    key: float(value) for key, value in stored_metadata.get("method_metadata", {}).items()
                    if isinstance(value, (int, float))
                }
                runtime_seconds = float(stored_metadata.get("runtime_seconds", float("nan")))
            else:
                runtime_seconds = float("nan")
        else:
            model, metadata = build_method(
                method,
                original,
                split,
                bundle,
                proxies,
                proxy_labels,
                forget_classes,
                output_is_logits,
                cfg,
                seed,
            )
            reused = 0.0
            runtime_seconds = time.perf_counter() - start
            if cfg.save_models:
                model.save(model_path)
            save_json(method_metadata_path, {
                "runtime_seconds": runtime_seconds,
                "method_metadata": metadata,
            })

        utility, class_accuracy = evaluate_utility(
            model, bundle, forget_classes, output_is_logits, cfg.eval_batch_size, seed
        )
        mia = evaluate_membership_inference(
            model,
            split,
            bundle,
            output_is_logits,
            cfg.eval_batch_size,
            cfg.mia_attack_repeats,
            seed,
        )
        parameter_stats = parameter_change_statistics(original, model)

        utility_rows.append({"seed": seed, "method": method, **utility, **metadata})
        mia_rows.append({"seed": seed, "method": method, **mia})
        runtime_rows.append({
            "seed": seed,
            "method": method,
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb(),
            "reused_saved_model": reused,
            "original_source": original_source,
            "output_is_logits": float(output_is_logits),
            **parameter_stats,
        })
        for class_id, accuracy in enumerate(class_accuracy):
            per_class_rows.append({
                "seed": seed,
                "method": method,
                "class_id": class_id,
                "class_name": bundle.class_names[class_id],
                "is_forgotten": int(class_id in forget_classes),
                "accuracy": float(accuracy),
            })
        print("Completed", method, utility)
        del model, original
        release_memory(clear_session=True)

    pd.DataFrame(utility_rows).to_csv(run_dir / "utility_metrics.csv", index=False)
    pd.DataFrame(mia_rows).to_csv(run_dir / "mia_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(run_dir / "runtime_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(run_dir / "per_class_accuracy.csv", index=False)

    run_representation_audit(cfg, run_dir, bundle, split, forget_classes, output_is_logits, seed)
    make_seed_figures(run_dir)
    outputs = {
        "run_dir": run_dir,
        "utility": run_dir / "utility_metrics.csv",
        "mia": run_dir / "mia_metrics.csv",
        "runtime": run_dir / "runtime_metrics.csv",
        "representation": run_dir / "representation_metrics.csv",
    }
    del proxies, proxy_labels, split, bundle
    release_memory(clear_session=True)
    return outputs


def run_representation_audit(
    cfg: ExperimentConfig,
    run_dir: Path,
    bundle: DataBundle,
    split: UnlearningSplit,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    seed: int,
) -> None:
    train_idx = stratified_indices(bundle.train_labels, cfg.probe_train_limit, seed)
    test_idx = stratified_indices(bundle.test_labels, cfg.probe_test_limit, seed + 1)
    cka_idx = stratified_indices(bundle.test_labels, cfg.cka_sample_limit, seed + 2)

    feature_paths: Dict[str, Path] = {}
    probability_paths: Dict[str, Path] = {}
    for method in cfg.representation_methods:
        path = method_model_path(run_dir, method)
        if not path.exists():
            continue
        reset_tensorflow(seed)
        model = load_saved_model(path)
        train_features, train_labels = extract_features(
            model,
            bundle.train_items[train_idx],
            bundle.train_labels[train_idx],
            bundle,
            cfg.eval_batch_size,
            seed,
        )
        test_features, test_labels = extract_features(
            model,
            bundle.test_items[test_idx],
            bundle.test_labels[test_idx],
            bundle,
            cfg.eval_batch_size,
            seed,
        )
        cka_features, cka_labels = extract_features(
            model,
            bundle.test_items[cka_idx],
            bundle.test_labels[cka_idx],
            bundle,
            cfg.eval_batch_size,
            seed,
        )
        probs, prob_labels = predict_probabilities(
            model,
            bundle.test_items[cka_idx],
            bundle.test_labels[cka_idx],
            bundle,
            output_is_logits,
            cfg.eval_batch_size,
            seed,
        )
        feature_path = run_dir / "features" / f"{method}.npz"
        probability_path = run_dir / "features" / f"{method}_probs.npz"
        np.savez_compressed(
            feature_path,
            train_features=train_features.astype(np.float32, copy=False),
            train_labels=train_labels,
            test_features=test_features.astype(np.float32, copy=False),
            test_labels=test_labels,
            cka_features=cka_features.astype(np.float32, copy=False),
            cka_labels=cka_labels,
        )
        np.savez_compressed(
            probability_path,
            probabilities=probs.astype(np.float32, copy=False),
            labels=prob_labels,
        )
        feature_paths[method] = feature_path
        probability_paths[method] = probability_path
        del (
            model,
            train_features,
            train_labels,
            test_features,
            test_labels,
            cka_features,
            cka_labels,
            probs,
            prob_labels,
        )
        release_memory(clear_session=True)

    if "original" not in feature_paths or "original" not in probability_paths:
        raise RuntimeError("Representation audit requires the saved original model.")

    with np.load(feature_paths["original"]) as data:
        original_features = data["cka_features"].astype(np.float32, copy=True)
    retrained_features = None
    retrained_probs = None
    if "retrained" in feature_paths:
        with np.load(feature_paths["retrained"]) as data:
            retrained_features = data["cka_features"].astype(np.float32, copy=True)
    if "retrained" in probability_paths:
        with np.load(probability_paths["retrained"]) as data:
            retrained_probs = data["probabilities"].astype(np.float32, copy=True)
    with np.load(probability_paths["original"]) as data:
        labels = data["labels"].astype(np.int64, copy=True)

    forget_mask = np.isin(labels, np.asarray(forget_classes))
    retain_mask = ~forget_mask

    rows: List[Dict[str, object]] = []
    for method, feature_path in feature_paths.items():
        with np.load(feature_path) as data:
            train_features = data["train_features"].astype(np.float32, copy=True)
            train_labels = data["train_labels"].astype(np.int64, copy=True)
            test_features = data["test_features"].astype(np.float32, copy=True)
            test_labels = data["test_labels"].astype(np.int64, copy=True)
            method_features = data["cka_features"].astype(np.float32, copy=True)

        probe = representation_probe_metrics(
            train_features,
            train_labels,
            test_features,
            test_labels,
            forget_classes,
        )
        row: Dict[str, object] = {
            "seed": seed,
            "method": method,
            "cka_to_original": linear_cka(method_features, original_features),
            **probe,
        }
        if retrained_features is not None:
            row["cka_to_retrained"] = linear_cka(method_features, retrained_features)
        if retrained_probs is not None and method in probability_paths:
            with np.load(probability_paths[method]) as data:
                method_probs = data["probabilities"].astype(np.float32, copy=True)
            row["js_to_retrained_all"] = jensen_shannon_divergence(method_probs, retrained_probs)
            row["js_to_retrained_retain"] = jensen_shannon_divergence(
                method_probs[retain_mask], retrained_probs[retain_mask]
            )
            row["js_to_retrained_forget"] = jensen_shannon_divergence(
                method_probs[forget_mask], retrained_probs[forget_mask]
            )
            row["prediction_disagreement_to_retrained"] = float(
                np.mean(method_probs.argmax(axis=1) != retrained_probs.argmax(axis=1))
            )
            del method_probs
        rows.append(row)
        del train_features, train_labels, test_features, test_labels, method_features, probe
        release_memory(clear_session=False)

    pd.DataFrame(rows).to_csv(run_dir / "representation_metrics.csv", index=False)
    del original_features, retrained_features, retrained_probs, labels, train_idx, test_idx, cka_idx
    release_memory(clear_session=True)

    def make_seed_figures(run_dir: Path) -> None:
    utility = pd.read_csv(run_dir / "utility_metrics.csv")
    mia = pd.read_csv(run_dir / "mia_metrics.csv")
    representation_path = run_dir / "representation_metrics.csv"
    representation = pd.read_csv(representation_path) if representation_path.exists() else pd.DataFrame()

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(utility))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10, len(utility) * 1.1), 5.5))
    ax.bar(positions - width / 2, utility["retained_accuracy"], width, label="Retained accuracy")
    ax.bar(positions + width / 2, utility["forgotten_accuracy"], width, label="Forgotten accuracy")
    ax.set_xticks(positions, utility["method"], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "retained_forgotten_accuracy.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(mia) * 1.1), 5.5))
    ax.bar(positions - width / 2, mia["mia_confidence_auc"], width, label="Confidence MIA")
    ax.bar(positions + width / 2, mia["mia_learned_auc"], width, label="Learned MIA")
    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xticks(positions, mia["method"], rotation=35, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "membership_inference_auc.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(utility) * 1.1), 5.5))
    ax.bar(positions - width / 2, utility["ece_15"], width, label="ECE")
    ax.bar(positions + width / 2, utility["brier"], width, label="Brier")
    ax.set_xticks(positions, utility["method"], rotation=35, ha="right")
    ax.set_ylabel("Calibration error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_metrics.png", dpi=220)
    plt.close(fig)

    if not representation.empty and "cka_to_retrained" in representation.columns:
        positions_rep = np.arange(len(representation))
        fig, ax = plt.subplots(figsize=(max(10, len(representation) * 1.1), 5.5))
        ax.bar(positions_rep - width / 2, representation["cka_to_original"], width, label="CKA to original")
        ax.bar(positions_rep + width / 2, representation["cka_to_retrained"], width, label="CKA to retrained")
        ax.set_xticks(positions_rep, representation["method"], rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Linear CKA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "representation_cka.png", dpi=220)
        plt.close(fig)