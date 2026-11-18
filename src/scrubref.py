def categorical_kl(p: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    p = tf.clip_by_value(p, 1e-7, 1.0)
    q = tf.clip_by_value(q, 1e-7, 1.0)
    return tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p / q), axis=-1))

def scrubref_train(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    forget_classes: Sequence[int],
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
    update_representation: bool = False,
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    set_global_determinism(seed)
    student = clone_with_weights(original)
    teacher = clone_with_weights(original)
    teacher.trainable = False

    if update_representation:
        set_tail_trainable(student, bundle.num_classes, cfg.representation_tail_layers)
    else:
        set_tail_trainable(student, bundle.num_classes, 0)

    feature_teacher = build_feature_model(teacher, bundle.num_classes)
    feature_student = build_feature_model(student, bundle.num_classes)

    # Precompute centroids for retain representations
    retain_ds_feat = make_dataset(split.retain_train_items[:1000], split.retain_train_labels[:1000], bundle, cfg.eval_batch_size, False, False, False, seed)
    retain_feats_list = []
    for xb, _ in retain_ds_feat:
        retain_feats_list.append(feature_teacher(xb, training=False))
    retain_centroid = tf.reduce_mean(tf.concat(retain_feats_list, axis=0), axis=0)

    optimizer = tf.keras.optimizers.Adam(cfg.scrub_lr)

    retain_ds = make_dataset(split.retain_train_items, split.retain_train_labels, bundle, cfg.batch_size, True, True, True, seed)
    proxy_ds = make_dataset(proxies, proxy_labels, bundle, cfg.proxy_batch_size, True, False, True, seed + 1)

    retain_iter = iter(retain_ds)
    proxy_iter = iter(proxy_ds)

    steps_per_epoch = max(1, len(split.retain_train_items) // cfg.batch_size)

    @tf.function(reduce_retracing=True)
    def train_step(retain_images: tf.Tensor, retain_labels: tf.Tensor, proxy_images: tf.Tensor, proxy_labels: tf.Tensor):
        with tf.GradientTape() as tape:
            # Forward on retain
            student_retain_out = student(retain_images, training=True)
            teacher_retain_out = teacher(retain_images, training=False)

            # Forward on proxy
            student_proxy_out = student(proxy_images, training=True)

            # Loss terms
            ce_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                retain_labels, student_retain_out, from_logits=output_is_logits
            ))

            kd_loss = categorical_kl(
                output_to_probs(teacher_retain_out, output_is_logits, cfg.kd_temperature),
                output_to_probs(student_retain_out, output_is_logits, cfg.kd_temperature)
            )

            forget_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                proxy_labels, student_proxy_out, from_logits=output_is_logits
            ))

            loss = cfg.retain_ce_weight * ce_loss + cfg.retain_kd_weight * kd_loss + cfg.forget_output_weight * forget_loss

            if update_representation:
                student_retain_feat = feature_student(retain_images, training=True)
                teacher_retain_feat = feature_teacher(retain_images, training=False)
                feat_retain_loss = tf.reduce_mean(tf.square(student_retain_feat - teacher_retain_feat))

                student_proxy_feat = feature_student(proxy_images, training=True)
                feat_forget_loss = tf.reduce_mean(tf.square(student_proxy_feat - retain_centroid))

                loss += cfg.retain_feature_weight * feat_retain_loss + cfg.forget_feature_weight * feat_forget_loss

        gradients = tape.gradient(loss, student.trainable_variables)
        if cfg.gradient_clip_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, cfg.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    # Run scrubbing epochs
    for epoch in range(cfg.scrub_epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            rx, ry = next(retain_iter)
            px, py = next(proxy_iter)
            loss_val = train_step(rx, ry, px, py)
            epoch_losses.append(float(loss_val.numpy()))
        print(f"ScrubRef Epoch {epoch+1}/{cfg.scrub_epochs} Loss: {np.mean(epoch_losses):.4f}")

    return student, {"scrub_final_loss": float(np.mean(epoch_losses))}