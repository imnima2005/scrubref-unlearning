def unsir_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    model = clone_with_weights(original)
    noise, noise_labels = generate_unsir_error_maximizing_noise(
        original,
        bundle,
        forget_classes,
        output_is_logits,
        count=min(cfg.proxy_count, 512),
        steps=cfg.unsir_noise_steps,
        seed=seed,
    )
    retain_images, retain_labels = materialize_items(
        split.retain_train_items,
        split.retain_train_labels,
        bundle,
        limit=min(len(noise), len(split.retain_train_items)),
        seed=seed,
    )
    impair_images = np.concatenate([retain_images, noise])
    impair_labels = np.concatenate([retain_labels, noise_labels])
    array_bundle = DataBundle(
        name=f"{bundle.name}_materialized",
        train_items=impair_images,
        train_labels=impair_labels,
        test_items=impair_images,
        test_labels=impair_labels,
        class_names=bundle.class_names,
        image_size=bundle.image_size,
        path_mode=False,
    )
    model = train_supervised(
        model,
        impair_images,
        impair_labels,
        array_bundle,
        output_is_logits,
        epochs=1,
        batch_size=cfg.batch_size,
        learning_rate=cfg.unsir_impair_lr,
        seed=seed,
        train_all=True,
    )
    model = train_supervised(
        model,
        retain_images,
        retain_labels,
        array_bundle,
        output_is_logits,
        epochs=max(1, cfg.baseline_epochs // 2),
        batch_size=cfg.batch_size,
        learning_rate=cfg.unsir_repair_lr,
        seed=seed + 1,
        train_all=True,
    )
    return model