import argparse
import os
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", required=True)
    parser.add_argument("--tflite_path", default="model/skin_texture.tflite")
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.tflite_path), exist_ok=True)
    with open(args.tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {args.tflite_path}")


if __name__ == "__main__":
    main()
