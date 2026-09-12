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
    centered -= centered.mean(axis=0)
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

    columns = 50
    atlas = samples.reshape(-1, columns, 28, 28).transpose(0, 2, 1, 3).reshape(-1, columns * 28)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(output / "mnist-pca-atlas.png", optimize=True)
    payload = {
        "count": len(indices), "imageSide": 28, "atlasColumns": columns,
        "explainedVariance": explained.round(6).tolist(),
        "points": points.round(5).tolist(), "labels": labels[indices].tolist(),
        "sourceIndices": indices.tolist(),
    }
    (output / "mnist-pca-data.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Built {len(indices)} points; explained variance: {explained.sum():.1%}")


if __name__ == "__main__":
    main()
