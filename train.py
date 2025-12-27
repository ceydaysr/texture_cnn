import argparse
import csv
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf


def load_labels(labels_path):
    items = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["path"], float(row["score"])))
    return items


def split_items(items, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items


def make_dataset(items, data_dir, img_size, batch_size):
    def gen():
        for rel_path, score in items:
            path = os.path.join(data_dir, rel_path)
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            yield arr, score

    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ]
    )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--max_score", type=float, default=5.0)
    parser.add_argument("--saved_model_dir", default="")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    labels_path = os.path.abspath(args.labels)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.saved_model_dir:
        args.saved_model_dir = os.path.join(script_dir, "model", "skin_texture_saved_model")

    items = load_labels(labels_path)
    if not items:
        raise SystemExit("No labels found.")

    # Normalize scores to 0-1 for regression
    norm_items = []
    for rel, score in items:
        score = max(args.min_score, min(args.max_score, score))
        norm = (score - args.min_score) / (args.max_score - args.min_score)
        norm_items.append((rel, norm))

    train_items, val_items = split_items(norm_items)

    train_ds = make_dataset(train_items, data_dir, args.img_size, args.batch_size)
    val_ds = make_dataset(val_items, data_dir, args.img_size, args.batch_size)

    model, base = build_model(args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Optional fine-tune
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:100]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    os.makedirs(args.saved_model_dir, exist_ok=True)
    model.export(args.saved_model_dir)
    print(f"Exported SavedModel to: {args.saved_model_dir}")


if __name__ == "__main__":
    main()
