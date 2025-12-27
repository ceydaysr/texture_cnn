import argparse
import csv
import os
from glob import glob
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


def load_labels(path):
    labels = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["path"]] = row["score"]
    return labels


def save_labels(path, labels):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score"])
        for k, v in labels.items():
            writer.writerow([k, v])


def list_images(data_dir):
    patterns = ["*.tif", "*.tiff"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(data_dir, "**", p), recursive=True))
    return sorted(files)

def laplacian_variance(img):
    gray = np.asarray(img, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    p = np.pad(gray, 1, mode="edge")
    lap = (
        p[1:-1, 0:-2]
        + p[1:-1, 2:]
        + p[0:-2, 1:-1]
        + p[2:, 1:-1]
        - 4.0 * p[1:-1, 1:-1]
    )
    return float(lap.var())

def presort_by_roughness(rel_paths, data_dir, size):
    scored = []
    total = len(rel_paths)
    for i, rel in enumerate(rel_paths, start=1):
        path = os.path.join(data_dir, rel)
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.BILINEAR)
        score = laplacian_variance(img)
        scored.append((rel, score))
        if i % 200 == 0 or i == total:
            print(f"Pre-sort: {i}/{total}")
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rel for rel, _ in scored], scored

def auto_labels_from_scores(scored, labels, bins=5):
    scores = [s for _, s in scored]
    if not scores:
        return labels
    stats = {
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
    }
    quantiles = np.quantile(scores, [i / bins for i in range(1, bins)])
    counts = {str(i): 0 for i in range(1, bins + 1)}
    for rel, score in scored:
        label = 1
        for i, q in enumerate(quantiles, start=1):
            if score <= q:
                label = i
                break
        else:
            label = bins
        label_str = str(label)
        labels[rel] = label_str
        counts[label_str] += 1
    return labels, counts, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels", default="labels.csv")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--presort", choices=["laplacian", "none"], default="laplacian")
    parser.add_argument("--presort_size", type=int, default=128)
    parser.add_argument("--auto_labels", action="store_true")
    parser.add_argument("--auto_bins", type=int, default=5)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    labels_path = os.path.abspath(args.labels)

    all_files = list_images(data_dir)
    if not all_files:
        raise SystemExit("No .tif files found.")

    labels = load_labels(labels_path)

    rel_paths = []
    for p in all_files:
        rel = os.path.relpath(p, data_dir)
        if rel not in labels:
            rel_paths.append(rel)

    if not rel_paths:
        print("All files already labeled.")
        return

    scored = None
    if args.presort == "laplacian" or args.auto_labels:
        rel_paths, scored = presort_by_roughness(rel_paths, data_dir, args.presort_size)

    if args.auto_labels:
        if not scored:
            raise SystemExit("No files to label.")
        labels, counts, stats = auto_labels_from_scores(scored, labels, bins=args.auto_bins)
        save_labels(labels_path, labels)
        print(f"Auto-labeled {len(scored)} files into {args.auto_bins} bins.")
        print("Histogram (label -> count):")
        for k in sorted(counts.keys(), key=int):
            print(f"  {k} -> {counts[k]}")
        print("Stats (laplacian variance):")
        print(f"  min: {stats['min']}")
        print(f"  median: {stats['median']}")
        print(f"  max: {stats['max']}")
        hist_path = os.path.splitext(labels_path)[0] + "_histogram.txt"
        with open(hist_path, "w", encoding="ascii") as f:
            f.write("Histogram (label -> count)\n")
            for k in sorted(counts.keys(), key=int):
                f.write(f"{k} -> {counts[k]}\n")
            f.write("Stats (laplacian variance)\n")
            f.write(f"min: {stats['min']}\n")
            f.write(f"median: {stats['median']}\n")
            f.write(f"max: {stats['max']}\n")
        print(f"Saved histogram to: {hist_path}")
        return

    root = tk.Tk()
    root.title("Skin texture labeler")

    img_label = tk.Label(root)
    img_label.pack()

    info = tk.Label(root, text="", font=("Arial", 12))
    info.pack()

    idx = {"value": 0}

    def show_image():
        i = idx["value"]
        if i >= len(rel_paths):
            info.config(text="Done. All images labeled.")
            img_label.config(image="")
            return

        rel = rel_paths[i]
        path = os.path.join(data_dir, rel)
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.size, args.size), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        img_label.image = photo
        img_label.config(image=photo)
        info.config(text=f"{i + 1}/{len(rel_paths)}  {rel}")

    def set_score(score):
        i = idx["value"]
        if i >= len(rel_paths):
            return
        rel = rel_paths[i]
        labels[rel] = str(score)
        save_labels(labels_path, labels)
        idx["value"] += 1
        show_image()

    def skip(_event=None):
        idx["value"] += 1
        show_image()

    def quit_app(_event=None):
        root.destroy()

    for k in ["1", "2", "3", "4", "5"]:
        root.bind(k, lambda e, s=int(k): set_score(s))

    root.bind("s", skip)
    root.bind("q", quit_app)

    instructions = tk.Label(
        root,
        text="Keys: 1-5 = score, s = skip, q = quit",
        font=("Arial", 10),
    )
    instructions.pack()

    show_image()
    root.mainloop()


if __name__ == "__main__":
    main()
