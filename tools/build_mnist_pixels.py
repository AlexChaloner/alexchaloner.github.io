#!/usr/bin/env python3
"""Build the self-contained, downsampled MNIST pixel bundle for the browser demo."""

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SIDE = 8
EXAMPLES_PER_DIGIT = 600


def read_idx_images(path):
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError("Unexpected MNIST image header")
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, rows, cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError("Unexpected MNIST label header")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
        if len(labels) != count:
            raise ValueError("Truncated MNIST labels")
        return labels


def b64(values):
    return base64.b64encode(values.tobytes()).decode("ascii")


def preview(images, path):
    columns = 20
    canvas = Image.new("L", (IMAGE_SIDE * columns, IMAGE_SIDE), 0)
    for column in range(columns):
        tile = Image.fromarray(images[column].reshape(IMAGE_SIDE, IMAGE_SIDE))
        canvas.paste(tile, (column * IMAGE_SIDE, 0))
    canvas.resize((IMAGE_SIDE * columns * 6, IMAGE_SIDE * 6), Image.Resampling.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", default="/tmp/mnist-pixel-preview.png")
    args = parser.parse_args()

    raw_images = read_idx_images(args.images)
    labels = read_idx_labels(args.labels)

    # Keep a deterministic balanced subset: 600 examples of every digit.
    chosen = []
    for digit in range(10):
        chosen.extend(np.flatnonzero(labels == digit)[:EXAMPLES_PER_DIGIT].tolist())
    chosen = np.array(chosen, dtype=np.int64)
    rng = np.random.default_rng(20260719)
    rng.shuffle(chosen)

    # Area pooling preserves the average ink in each cell. These 64 values are the
    # model state itself; there is no encoder or decoder in the browser experiment.
    source = torch.from_numpy(raw_images[chosen].copy()).float().unsqueeze(1)
    pixels = torch.nn.functional.adaptive_avg_pool2d(source, (IMAGE_SIDE, IMAGE_SIDE))
    pixels = torch.clamp(torch.round(pixels), 0, 255).byte().squeeze(1).numpy()
    subset_labels = labels[chosen]

    payload = {
        "version": 2,
        "count": int(len(chosen)),
        "imageSide": IMAGE_SIDE,
        "pixelDim": IMAGE_SIDE * IMAGE_SIDE,
        "pixels": b64(pixels),
        "labels": b64(subset_labels.astype(np.uint8)),
    }
    output = "window.MNISTPixelData=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    Path(args.output).write_text(output, encoding="utf-8")
    preview(pixels[:20], args.preview)
    print(f"wrote {args.output} ({len(output) / 1024:.1f} KiB)", flush=True)
    print(f"preview {args.preview}", flush=True)


if __name__ == "__main__":
    main()
