import tensorflow as tf
from typing import Dict, List, Tuple
from pathlib import Path

def residual_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False, name=f"{name}_proj_conv")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.ReLU(name=f"{name}_out")(x)

def build_resnet18(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    if input_shape[0] <= 64:
        x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="stem_conv")(x)
    else:
        x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    if input_shape[0] > 64:
        x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same", name="stem_pool")(x)
    filters_per_stage = [64, 128, 256, 512]
    for stage, filters in enumerate(filters_per_stage):
        x = residual_block(x, filters, 1 if stage == 0 else 2, f"stage{stage + 1}_block1")
        x = residual_block(x, filters, 1, f"stage{stage + 1}_block2")
    x = tf.keras.layers.GlobalAveragePooling2D(name="penultimate_features")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, dtype="float32", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="resnet18_scrubref")

def custom_objects() -> Dict[str, object]:
    return {
        "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
    }

def clone_architecture(model: tf.keras.Model) -> tf.keras.Model:
    tf.keras.utils.get_custom_objects().update(custom_objects())
    try:
        return tf.keras.models.clone_model(model)
    except Exception as first_error:
        try:
            return tf.keras.models.model_from_json(model.to_json(), custom_objects=custom_objects())
        except Exception as second_error:
            raise RuntimeError(
                "The checkpoint architecture could not be cloned. Save a serialization-safe .keras model "
                "or remove unsupported Lambda layers before running the revision pipeline."
            ) from second_error

def clone_with_weights(model: tf.keras.Model) -> tf.keras.Model:
    cloned = clone_architecture(model)
    cloned.set_weights(model.get_weights())
    return cloned

def fresh_clone(model: tf.keras.Model, initial_weights_path: str = "") -> tf.keras.Model:
    cloned = clone_architecture(model)
    if initial_weights_path and Path(initial_weights_path).exists():
        cloned.load_weights(initial_weights_path)
    return cloned

def output_to_logits(output: tf.Tensor, output_is_logits: bool) -> tf.Tensor:
    output = tf.cast(output, tf.float32)
    if output_is_logits:
        return output
    probabilities = tf.clip_by_value(output, 1e-7, 1.0)
    return tf.math.log(probabilities)

def output_to_probs(output: tf.Tensor, output_is_logits: bool, temperature: float = 1.0) -> tf.Tensor:
    logits = output_to_logits(output, output_is_logits) / temperature
    return tf.nn.softmax(logits, axis=-1)

def leaf_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    result: List[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            result.extend(leaf_layers(layer))
        else:
            result.append(layer)
    return result

def model_containers(model: tf.keras.Model) -> List[tf.keras.Model]:
    containers = [model]
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            containers.extend(model_containers(layer))
    return containers

def classifier_layer(model: tf.keras.Model, num_classes: int) -> tf.keras.layers.Dense:
    candidates = [layer for layer in leaf_layers(model) if isinstance(layer, tf.keras.layers.Dense) and layer.units == num_classes]
    if not candidates:
        raise ValueError(f"No Dense classifier with {num_classes} units was found.")
    return candidates[-1]

def build_feature_model(model: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    classifier = classifier_layer(model, num_classes)
    return tf.keras.Model(model.inputs, classifier.input, name=f"{model.name}_feature_model")

def set_all_trainable(model: tf.keras.Model, trainable: bool = True, freeze_batch_norm: bool = False) -> None:
    for container in model_containers(model):
        container.trainable = True
    for layer in leaf_layers(model):
        if freeze_batch_norm and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = trainable

def set_tail_trainable(model: tf.keras.Model, num_classes: int, tail_weight_layers: int) -> List[str]:
    for container in model_containers(model):
        container.trainable = True
    leaves = leaf_layers(model)
    for layer in leaves:
        layer.trainable = False
    classifier = classifier_layer(model, num_classes)
    classifier.trainable = True
    candidates = [
        layer for layer in leaves
        if layer is not classifier
        and layer.weights
        and not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    selected = candidates[-tail_weight_layers:] if tail_weight_layers > 0 else []
    for layer in selected:
        layer.trainable = True
    return [layer.name for layer in selected] + [classifier.name]


