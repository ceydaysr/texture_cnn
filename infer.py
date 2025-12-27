import argparse
import csv
import os
import random
from glob import glob

import numpy as np
from PIL import Image
import tensorflow as tf


def list_images(data_dir):
    patterns = ["*.tif", "*.tiff"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(data_dir, "**", p), recursive=True))
    return sorted(files)


def load_labels(labels_path):
    if not labels_path:
        return {}
    labels = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["path"]] = float(row["score"])
    return labels


def preprocess(path, img_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--max_score", type=float, default=5.0)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    labels = load_labels(args.labels)

    files = list_images(data_dir)
    if not files:
        raise SystemExit("No .tif files found.")

    rng = random.Random(args.seed)
    samples = rng.sample(files, min(args.num_samples, len(files)))

    saved = tf.saved_model.load(args.model_dir)
    infer = saved.signatures["serve"]

    print("Sample predictions:")
    for path in samples:
        arr = preprocess(path, args.img_size)
        batch = np.expand_dims(arr, axis=0)
        output = infer(tf.constant(batch))
        pred_norm = float(next(iter(output.values())).numpy()[0][0])
        pred = pred_norm * (args.max_score - args.min_score) + args.min_score
        smooth_pct = max(0.0, min(100.0, (1.0 - pred_norm) * 100.0))

        rel = os.path.relpath(path, data_dir)
        if rel in labels:
            true = labels[rel]
            print(f"{rel} -> pred: {pred:.2f}, smooth: {smooth_pct:.0f}%, label: {true:.2f}")
        else:
            print(f"{rel} -> pred: {pred:.2f}, smooth: {smooth_pct:.0f}%")


if __name__ == "__main__":
    main()
