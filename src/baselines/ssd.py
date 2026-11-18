def estimate_fisher_diagonal(
    model: tf.keras.Model,
    items: np.ndarray,
    labels: np.ndarray,
    bundle: DataBundle,
    output_is_logits: bool,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> List[tf.Tensor]:
    set_all_trainable(model, True, freeze_batch_norm=True)
    variables = model.trainable_variables
    fisher = [tf.zeros_like(variable, dtype=tf.float32) for variable in variables]
    ds = make_dataset(items, labels, bundle, batch_size, True, False, False, seed)
    batch_count = 0
    for images, targets in ds.take(max_batches):
        with tf.GradientTape() as tape:
            output = model(images, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                targets, output, from_logits=output_is_logits
            ))
        gradients = tape.gradient(loss, variables)
        fisher = [
            accumulated + (tf.square(tf.cast(gradient, tf.float32)) if gradient is not None else 0.0)
            for accumulated, gradient in zip(fisher, gradients)
        ]
        batch_count += 1
    return [value / max(batch_count, 1) for value in fisher]


def ssd_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    model = clone_with_weights(original)
    retain_fisher = estimate_fisher_diagonal(
        model,
        split.retain_train_items,
        split.retain_train_labels,
        bundle,
        output_is_logits,
        cfg.batch_size,
        cfg.ssd_fisher_batches,
        seed,
    )
    forget_fisher = estimate_fisher_diagonal(
        model,
        split.forget_train_items,
        split.forget_train_labels,
        bundle,
        output_is_logits,
        cfg.batch_size,
        cfg.ssd_fisher_batches,
        seed + 1,
    )
    epsilon = 1e-12
    for variable, retain_value, forget_value in zip(model.trainable_variables, retain_fisher, forget_fisher):
        ratio = forget_value / (retain_value + epsilon)
        selected = ratio > cfg.ssd_selection_weighting
        dampening = tf.minimum(
            cfg.ssd_dampening_constant * retain_value / (forget_value + epsilon),
            tf.ones_like(forget_value),
        )
        factor = tf.where(selected, dampening, tf.ones_like(dampening))
        variable.assign(tf.cast(variable, tf.float32) * factor)
    return model
