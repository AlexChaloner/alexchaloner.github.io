#!/usr/bin/env python3
"""Build the small, self-contained MNIST latent bundle used by the browser demo."""

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


LATENT_DIM = 12
HIDDEN_DIM = 128


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(LATENT_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, 784),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def read_idx_images(path):
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError("Unexpected MNIST image header")
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, 784)


def read_idx_labels(path):
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError("Unexpected MNIST label header")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
        if len(labels) != count:
            raise ValueError("Truncated MNIST labels")
        return labels


def quantize(values):
    bound = float(np.max(np.abs(values))) or 1.0
    scale = bound / 127.0
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8), scale


def b64(values):
    return base64.b64encode(values.tobytes()).decode("ascii")


def preview(images, reconstructions, path):
    canvas = Image.new("L", (28 * 10, 28 * 2), 0)
    for column in range(10):
        source = Image.fromarray((images[column].reshape(28, 28) * 255).astype(np.uint8))
        reconstruction = Image.fromarray((reconstructions[column].reshape(28, 28) * 255).astype(np.uint8))
        canvas.paste(source, (column * 28, 0))
        canvas.paste(reconstruction, (column * 28, 28))
    canvas.resize((1120, 224), Image.Resampling.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", default="/tmp/mnist-latent-preview.png")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(20260719)
    np.random.seed(20260719)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    raw_images = read_idx_images(args.images)
    labels = read_idx_labels(args.labels)
    images = torch.from_numpy(raw_images.astype(np.float32) / 255.0)

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(20260719),
    )
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = torch.nn.functional.binary_cross_entropy(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        print(f"epoch {epoch + 1}/{args.epochs} bce={total / len(images):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        all_latents = model.encoder(images).numpy()
        reconstructions = model.decoder(model.encoder(images[:10])).numpy()
    preview(images[:10].numpy(), reconstructions, args.preview)

    # Keep a deterministic balanced subset: 600 examples of every digit.
    chosen = []
    for digit in range(10):
        chosen.extend(np.flatnonzero(labels == digit)[:600].tolist())
    chosen = np.array(chosen, dtype=np.int64)
    rng = np.random.default_rng(20260719)
    rng.shuffle(chosen)
    subset_latents = all_latents[chosen]
    subset_labels = labels[chosen]

    mean = subset_latents.mean(axis=0)
    std = subset_latents.std(axis=0) + 1e-6
    normalized = (subset_latents - mean) / std
    latent_scale = 1.0 / 24.0
    latent_q = np.clip(np.rint(normalized / latent_scale), -127, 127).astype(np.int8)

    # Fold latent normalization into the decoder's first affine layer.
    first = model.decoder[0]
    second = model.decoder[2]
    w1 = first.weight.detach().numpy() * std[np.newaxis, :]
    b1 = first.bias.detach().numpy() + first.weight.detach().numpy() @ mean
    w2 = second.weight.detach().numpy()
    b2 = second.bias.detach().numpy()
    w1_q, w1_scale = quantize(w1)
    w2_q, w2_scale = quantize(w2)

    payload = {
        "version": 1,
        "count": int(len(chosen)),
        "latentDim": LATENT_DIM,
        "latentScale": latent_scale,
        "latents": b64(latent_q),
        "labels": b64(subset_labels.astype(np.uint8)),
        "decoder": {
            "hiddenDim": HIDDEN_DIM,
            "w1": b64(w1_q),
            "w1Scale": w1_scale,
            "b1": np.round(b1, 6).tolist(),
            "w2": b64(w2_q),
            "w2Scale": w2_scale,
            "b2": np.round(b2, 6).tolist(),
        },
    }
    output = "window.MNISTLatentData=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    Path(args.output).write_text(output, encoding="utf-8")
    print(f"wrote {args.output} ({len(output) / 1024:.1f} KiB)", flush=True)
    print(f"preview {args.preview}", flush=True)


if __name__ == "__main__":
    main()
