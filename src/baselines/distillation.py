def bad_teacher_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    student = clone_with_weights(original)
    competent = clone_with_weights(original)
    incompetent = fresh_clone(original)
    competent.trainable = False
    incompetent.trainable = False
    set_all_trainable(student, True, freeze_batch_norm=True)
    optimizer = tf.keras.optimizers.Adam(cfg.baseline_lr)
    retain_ds = make_dataset(split.retain_train_items, split.retain_train_labels, bundle, cfg.batch_size, True, True, True, seed)
    forget_ds = make_dataset(split.forget_train_items, split.forget_train_labels, bundle, cfg.batch_size, True, True, True, seed + 1)
    retain_it, forget_it = iter(retain_ds), iter(forget_ds)
    steps = max(1, math.ceil(max(len(split.retain_train_items), len(split.forget_train_items)) / cfg.batch_size))

    @tf.function(reduce_retracing=True)
    def step(retain_images: tf.Tensor, retain_labels: tf.Tensor, forget_images: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            student_retain = student(retain_images, training=True)
            student_forget = student(forget_images, training=True)
            competent_retain = competent(retain_images, training=False)
            incompetent_forget = incompetent(forget_images, training=False)
            ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                retain_labels, student_retain, from_logits=output_is_logits
            ))
            kd_retain = categorical_kl(
                output_to_probs(competent_retain, output_is_logits, cfg.kd_temperature),
                output_to_probs(student_retain, output_is_logits, cfg.kd_temperature),
            )
            kd_forget = categorical_kl(
                output_to_probs(incompetent_forget, output_is_logits, cfg.kd_temperature),
                output_to_probs(student_forget, output_is_logits, cfg.kd_temperature),
            )
            loss = ce + kd_retain + kd_forget
        gradients = tape.gradient(loss, student.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, cfg.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    for epoch in range(cfg.baseline_epochs):
        losses = []
        for _ in range(steps):
            retain_images, retain_labels = next(retain_it)
            forget_images, _ = next(forget_it)
            losses.append(float(step(retain_images, retain_labels, forget_images).numpy()))
        print(f"Bad Teacher epoch {epoch + 1}: loss={np.mean(losses):.6f}")
    return student


def scrub_baseline(
    original: tf.keras.Model,
    split: UnlearningSplit,
    bundle: DataBundle,
    output_is_logits: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> tf.keras.Model:
    student = clone_with_weights(original)
    teacher = clone_with_weights(original)
    teacher.trainable = False
    set_all_trainable(student, True, freeze_batch_norm=True)
    optimizer = tf.keras.optimizers.Adam(cfg.baseline_lr)
    retain_ds = make_dataset(split.retain_train_items, split.retain_train_labels, bundle, cfg.batch_size, True, True, True, seed)
    forget_ds = make_dataset(split.forget_train_items, split.forget_train_labels, bundle, cfg.batch_size, True, True, True, seed + 1)
    retain_it, forget_it = iter(retain_ds), iter(forget_ds)
    steps = max(1, math.ceil(max(len(split.retain_train_items), len(split.forget_train_items)) / cfg.batch_size))

    @tf.function(reduce_retracing=True)
    def retain_step(images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            student_output = student(images, training=True)
            teacher_output = teacher(images, training=False)
            ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, student_output, from_logits=output_is_logits
            ))
            kd = categorical_kl(
                output_to_probs(teacher_output, output_is_logits, cfg.kd_temperature),
                output_to_probs(student_output, output_is_logits, cfg.kd_temperature),
            )
            loss = ce + kd
        gradients = tape.gradient(loss, student.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, cfg.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def forget_step(images: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            student_output = student(images, training=True)
            teacher_output = teacher(images, training=False)
            divergence = categorical_kl(
                output_to_probs(teacher_output, output_is_logits, cfg.kd_temperature),
                output_to_probs(student_output, output_is_logits, cfg.kd_temperature),
            )
            loss = -divergence
        gradients = tape.gradient(loss, student.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, cfg.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return divergence

    for epoch in range(cfg.baseline_epochs):
        retain_losses = []
        forget_divergences = []
        for _ in range(steps):
            forget_images, _ = next(forget_it)
            forget_divergences.append(float(forget_step(forget_images).numpy()))
            retain_images, retain_labels = next(retain_it)
            retain_losses.append(float(retain_step(retain_images, retain_labels).numpy()))
        print(
            f"SCRUB epoch {epoch + 1}: retain_loss={np.mean(retain_losses):.6f}, "
            f"forget_divergence={np.mean(forget_divergences):.6f}"
        )
    return student