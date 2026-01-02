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


def preprocess(path, img_size, cropper=None):
    img = Image.open(path).convert("RGB")
    if cropper is not None:
        arr = np.asarray(img)
        return cropper.extract(arr)
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
    parser.add_argument("--use_cheeks", action="store_true")
    parser.add_argument("--cheek_pad", type=float, default=0.15)
    parser.add_argument("--landmark_model", default="")
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

    cropper = None
    if args.use_cheeks:
        from cheek_crop import CheekCropper

        cropper = CheekCropper(
            img_size=args.img_size,
            pad=args.cheek_pad,
            model_path=args.landmark_model,
        )

    print("Sample predictions:")
    try:
        for path in samples:
            arrs = preprocess(path, args.img_size, cropper=cropper)
            rel = os.path.relpath(path, data_dir)
            if arrs is None:
                print(f"{rel} -> no face detected")
                continue

            if isinstance(arrs, list):
                pred_norms = []
                for arr in arrs:
                    batch = np.expand_dims(arr, axis=0)
                    output = infer(tf.constant(batch))
                    pred_norms.append(float(next(iter(output.values())).numpy()[0][0]))
                pred_norm = float(np.mean(pred_norms))
                left_score = pred_norms[0] * (args.max_score - args.min_score) + args.min_score
                right_score = pred_norms[1] * (args.max_score - args.min_score) + args.min_score
            else:
                batch = np.expand_dims(arrs, axis=0)
                output = infer(tf.constant(batch))
                pred_norm = float(next(iter(output.values())).numpy()[0][0])
                left_score = None
                right_score = None

            pred = pred_norm * (args.max_score - args.min_score) + args.min_score
            smooth_pct = max(0.0, min(100.0, (1.0 - pred_norm) * 100.0))

            if rel in labels:
                true = labels[rel]
                if left_score is not None:
                    print(
                        f"{rel} -> pred: {pred:.2f} (L {left_score:.2f}, R {right_score:.2f}), "
                        f"smooth: {smooth_pct:.0f}%, label: {true:.2f}"
                    )
                else:
                    print(f"{rel} -> pred: {pred:.2f}, smooth: {smooth_pct:.0f}%, label: {true:.2f}")
            else:
                if left_score is not None:
                    print(
                        f"{rel} -> pred: {pred:.2f} (L {left_score:.2f}, R {right_score:.2f}), "
                        f"smooth: {smooth_pct:.0f}%"
                    )
                else:
                    print(f"{rel} -> pred: {pred:.2f}, smooth: {smooth_pct:.0f}%")
    finally:
        if cropper is not None:
            cropper.close()


if __name__ == "__main__":
    main()
