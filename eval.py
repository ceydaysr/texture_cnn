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
    parser.add_argument("--labels", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--min_score", type=float, default=1.0)
    parser.add_argument("--max_score", type=float, default=5.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    labels = load_labels(args.labels)
    files = list_images(data_dir)

    saved = tf.saved_model.load(args.model_dir)
    infer = saved.signatures["serve"]

    labeled = []
    for path in files:
        rel = os.path.relpath(path, data_dir)
        if rel not in labels:
            continue
        labeled.append((path, labels[rel]))

    if not labeled:
        raise SystemExit("No labeled samples found.")

    if args.val_split <= 0 or args.val_split >= 1:
        eval_items = labeled
        split_label = "full"
    else:
        rng = random.Random(args.seed)
        rng.shuffle(labeled)
        n_val = max(1, int(len(labeled) * args.val_split))
        eval_items = labeled[:n_val]
        split_label = f"holdout {args.val_split:.2f}"

    preds = []
    trues = []
    for path, true in eval_items:
        arr = preprocess(path, args.img_size)
        batch = np.expand_dims(arr, axis=0)
        output = infer(tf.constant(batch))
        pred_norm = float(next(iter(output.values())).numpy()[0][0])
        pred = pred_norm * (args.max_score - args.min_score) + args.min_score
        preds.append(pred)
        trues.append(true)

    preds = np.asarray(preds, dtype=np.float32)
    trues = np.asarray(trues, dtype=np.float32)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    score_pct = max(0.0, min(100.0, (1.0 - (mae / (args.max_score - args.min_score))) * 100.0))
    print(f"Split: {split_label}")
    print(f"Samples: {len(preds)}")
    print(f"MAE (1-5 scale): {mae:.4f}")
    print(f"RMSE (1-5 scale): {rmse:.4f}")
    print(f"Score (0-100): {score_pct:.1f}%")


if __name__ == "__main__":
    main()
