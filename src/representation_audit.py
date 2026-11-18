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
# Missing representation audit helper definitions
def stratified_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)
    samples_per_class = max(1, limit // len(unique_classes))
    selected_indices = []
    for class_id in unique_classes:
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > 0:
            chosen = rng.choice(indices, min(len(indices), samples_per_class), replace=False)
            selected_indices.extend(chosen)
    return np.array(selected_indices, dtype=np.int64)

def extract_features(model: tf.keras.Model, items: np.ndarray, labels: np.ndarray, bundle: DataBundle, batch_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    feature_model = build_feature_model(model, bundle.num_classes)
    ds = make_dataset(items, labels, bundle, batch_size, False, False, False, seed)
    features = []
    labels_out = []
    for xb, yb in ds:
        features.append(feature_model(xb, training=False).numpy())
        labels_out.append(yb.numpy())
    return np.concatenate(features), np.concatenate(labels_out)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XTX = np.dot(X, X.T)
    YTY = np.dot(Y, Y.T)
    num = np.sum(XTX * YTY)
    den = np.sqrt(np.sum(XTX * XTX) * np.sum(YTY * YTY))
    return float(num / max(den, 1e-12))

def jensen_shannon_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.clip(P, 1e-7, 1.0)
    Q = np.clip(Q, 1e-7, 1.0)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(P / M), axis=1)
    kl_qm = np.sum(Q * np.log(Q / M), axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))

def representation_probe_metrics(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray, test_labels: np.ndarray, forget_classes: Sequence[int]) -> Dict[str, float]:
    # Simple linear probe audit implementation
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_features)
    X_te = scaler.transform(test_features)

    probe = LogisticRegression(max_iter=150, class_weight='balanced', C=0.1)
    probe.fit(X_tr, train_labels)
    preds = probe.predict(X_te)

    forget_mask = np.isin(test_labels, np.asarray(forget_classes))

    return {
        "linear_probe_overall_accuracy": float(accuracy_score(test_labels, preds)),
        "linear_probe_retained_accuracy": float(accuracy_score(test_labels[~forget_mask], preds[~forget_mask])),
        "linear_probe_forgotten_accuracy": float(accuracy_score(test_labels[forget_mask], preds[forget_mask]))
    }

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