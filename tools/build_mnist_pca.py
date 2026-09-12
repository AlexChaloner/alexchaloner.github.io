#!/usr/bin/env python3
"""Build a reproducible PCA scatter and original-image atlas from MNIST IDX files."""

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="assets")
    args = parser.parse_args()
    with gzip.open(args.images, "rb") as source:
        magic, count, rows, cols = struct.unpack(">IIII", source.read(16))
        assert (magic, rows, cols) == (2051, 28, 28)
        images = np.frombuffer(source.read(), dtype=np.uint8).reshape(count, rows, cols)
    with gzip.open(args.labels, "rb") as source:
        magic, label_count = struct.unpack(">II", source.read(8))
        assert magic == 2049 and label_count == count
        labels = np.frombuffer(source.read(), dtype=np.uint8)
        assert labels.size == count

    rng = np.random.default_rng(20260912)
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == digit), 300, replace=False)
        for digit in range(10)
    ])
    rng.shuffle(indices)
    samples = images[indices]
    # Center raw pixel intensities; do not standardize individual pixels.
    centered = samples.reshape(len(indices), -1).astype(np.float64) / 255
    mean = centered.mean(axis=0)
    centered -= mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    components = eigenvectors[:, -2:][:, ::-1].copy()
    # Fix arbitrary eigenvector signs for reproducible plot orientation.
    for column in range(2):
        pivot = np.argmax(np.abs(components[:, column]))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    points = centered @ components
    explained = eigenvalues[-2:][::-1] / eigenvalues.sum()
    assert np.allclose(components.T @ components, np.eye(2))
    assert np.allclose(points.mean(axis=0), 0, atol=1e-10)

    # Keep the PCA axes fitted to the same 3,000 images; display 30 per digit.
    displayed = np.concatenate([
        np.flatnonzero(labels[indices] == digit)[:30] for digit in range(10)
    ])
    rng.shuffle(displayed)
    points = points[displayed]
    samples = samples[displayed]
    indices = indices[displayed]

    # Match the demos' N(0,1) source in [-1,1] image coordinates. Transform
    # back to raw-pixel coordinates before applying the SAME data PCA.
    # Clip only the image previews, never the values used for projection.
    noise_seed = 20260913
    noise = np.random.default_rng(noise_seed).normal(size=(20, 784))
    noise_pixels = (noise + 1) / 2
    noise_points = (noise_pixels - mean) @ components
    previews = np.rint(np.clip(noise_pixels, 0, 1) * 255).astype(np.uint8).reshape(-1, 28, 28)
    tiles = np.concatenate([samples, previews])
    columns = 50
    padding = (-len(tiles)) % columns
    tiles = np.pad(tiles, ((0, padding), (0, 0), (0, 0)))
    atlas = tiles.reshape(-1, columns, 28, 28).transpose(0, 2, 1, 3).reshape(-1, columns * 28)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(output / "mnist-pca-atlas.png", optimize=True)
    payload = {
        "count": len(indices) + len(noise), "digitCount": len(indices),
        "noiseCount": len(noise), "pcaFitCount": len(centered),
        "noiseSeed": noise_seed, "noiseDistribution": "N(0,1) in [-1,1] image coordinates",
        "imageSide": 28, "atlasColumns": columns,
        "explainedVariance": explained.round(6).tolist(),
        "points": np.concatenate([points, noise_points]).round(5).tolist(),
        "labels": labels[indices].tolist() + [None] * len(noise),
        "sourceIndices": indices.tolist(),
    }
    (output / "mnist-pca-data.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Built {len(indices)} digits + {len(noise)} noise points; explained variance: {explained.sum():.1%}")
    print(f"Digit bounds: {points.min(axis=0)} to {points.max(axis=0)}")
    print(f"Noise bounds: {noise_points.min(axis=0)} to {noise_points.max(axis=0)}")


if __name__ == "__main__":
    main()
