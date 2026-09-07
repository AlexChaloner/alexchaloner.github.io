"""Render the blog's Mandelbrot metaphor using NumPy and Pillow.

Run: python3 tools/build_fractal_problem.py
The first two panels deliberately omit structure; labels are an illustration,
not a measurement of model capabilities. The last panel is a finite raster
approximation, with 800 escape-time iterations and 2x supersampling.
"""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, PngImagePlugin


ROOT = Path(__file__).resolve().parents[1]
SCALE = 2
WIDTH, HEIGHT = 3000, 1000
PANEL_WIDTH, PANEL_HEIGHT = 880, 720
BLUE = "#93C9EB"


def mandelbrot_masks():
    width, height = PANEL_WIDTH * SCALE, PANEL_HEIGHT * SCALE
    # Equal scale on both axes, shared by all three panels.
    x = -2.12 + (np.arange(width) + 0.5) * 2.8 / width
    y = ((np.arange(height) + 0.5) - height / 2) * 2.8 / width
    real, imag = np.meshgrid(x, y)
    q = (real - 0.25) ** 2 + imag**2
    core = (q * (q + real - 0.25) <= 0.25 * imag**2) | (
        (real + 1) ** 2 + imag**2 <= 0.0625
    )
    inside = core.ravel().copy()
    indices = np.flatnonzero(~inside)
    c = (real + 1j * imag).ravel()[indices]
    z = np.zeros_like(c)
    for _ in range(800):
        z = z * z + c
        keep = z.real**2 + z.imag**2 <= 4
        indices, c, z = indices[keep], c[keep], z[keep]
        if indices.size == 0:
            break
    inside[indices] = True
    full = Image.fromarray(inside.reshape(height, width).astype(np.uint8) * 255)
    # Opening removes the narrow branches and smallest satellite bulbs while
    # retaining larger secondary structure. Union with the core keeps nesting.
    medium = full.filter(ImageFilter.MinFilter(17)).filter(ImageFilter.MaxFilter(17))
    medium = Image.fromarray(np.maximum(np.asarray(medium), core.astype(np.uint8) * 255))
    basic = Image.fromarray(core.astype(np.uint8) * 255)
    return basic, medium, full


def boundary_masks():
    """Show the same seahorse-valley crop at three levels of retained detail."""
    width, height = PANEL_WIDTH * SCALE, PANEL_HEIGHT * SCALE
    span = 0.012
    x = -0.743643887 + ((np.arange(width) + 0.5) - width / 2) * span / width
    y = 0.131825904 - ((np.arange(height) + 0.5) - height / 2) * span / width
    c = (x[None, :] + 1j * y[:, None]).ravel()
    indices = np.arange(c.size)
    z = np.zeros_like(c)
    for _ in range(500):
        z = z * z + c
        keep = z.real**2 + z.imag**2 <= 4
        indices, c, z = indices[keep], c[keep], z[keep]
    inside = np.zeros(width * height, dtype=np.uint8)
    inside[indices] = 255
    full = Image.fromarray(inside.reshape(height, width))
    # Isotropic smoothing avoids the square bumps of a box-shaped filter.
    # A raised threshold suppresses narrow branches and tiny islands.
    masks = []
    for radius in (18 * SCALE, 3 * SCALE):
        blurred = full.filter(ImageFilter.GaussianBlur(radius))
        masks.append(blurred.point(lambda value: 255 if value >= 155 else 0))
    return *masks, full


def colour_panels():
    """Escape-time colour bands make unresolved boundary detail visible."""
    width, height = PANEL_WIDTH * SCALE, PANEL_HEIGHT * SCALE
    span = 3.15
    x = -2.2 + (np.arange(width) + 0.5) * span / width
    y = ((np.arange(height) + 0.5) - height / 2) * span / width
    real, imag = np.meshgrid(x, y)
    q = (real - 0.25) ** 2 + imag**2
    core = (q * (q + real - 0.25) <= 0.25 * imag**2) | (
        (real + 1) ** 2 + imag**2 <= 0.0625
    )
    count = np.full(real.size, 801, dtype=np.int32)
    smooth = np.full(real.size, 801.0)
    indices = np.flatnonzero(~core.ravel())
    c = (real + 1j * imag).ravel()[indices]
    z = np.zeros_like(c)
    for iteration in range(1, 801):
        z = z * z + c
        magnitude = np.abs(z)
        escaped = magnitude > 2
        count[indices[escaped]] = iteration
        smooth[indices[escaped]] = iteration + 1 - np.log2(np.log(magnitude[escaped]))
        keep = ~escaped
        indices, c, z = indices[keep], c[keep], z[keep]
        if not indices.size:
            break
    # White fades into cool blues, with amber and coral picking out the curls.
    stops = np.array([0, 3.5, 5, 7, 10, 14, 20, 28, 40, 60, 100, 180, 400, 801])
    palette = np.array([
        (255, 255, 255), (255, 255, 255), (236, 246, 252),
        (143, 207, 231), (44, 147, 176), (42, 76, 129),
        (241, 182, 97), (223, 110, 92), (73, 70, 126),
        (104, 185, 204), (247, 202, 133), (200, 95, 102),
        (111, 100, 166), (22, 35, 65),
    ])
    rgb = np.stack([np.interp(smooth, stops, palette[:, channel])
                    for channel in range(3)], axis=-1).astype(np.uint8)
    panels = []
    for limit in (7, 18, 800):
        pixels = rgb.copy()
        pixels[count > limit] = (22, 35, 65)
        panels.append(Image.fromarray(pixels.reshape(height, width, 3)))
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", action="store_true", help="Render a close-up of the boundary.")
    parser.add_argument("--colour", action="store_true", help="Render full Mandelbrots with coloured escape-time bands.")
    args = parser.parse_args()
    canvas = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40 * SCALE)
    masks = colour_panels() if args.colour else (boundary_masks() if args.detail else mandelbrot_masks())
    for index, (label, mask) in enumerate(zip(("GPT-4", "GPT-6", "Reality"), masks)):
        left = (index * 1000 + 60) * SCALE
        top = 100 * SCALE
        if args.colour:
            canvas.paste(mask, (left, top))
        else:
            canvas.paste(BLUE, (left, top), mask)
        draw.text(((index * 1000 + 500) * SCALE, 895 * SCALE), label,
                  font=font, fill="#34495A", anchor="mm")
    canvas = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    metadata = PngImagePlugin.PngInfo()
    description = ("Same Mandelbrot boundary crop in all panels. GPT-4 and GPT-6 deliberately omit progressively less fine structure; Reality uses a 500-iteration finite-resolution approximation."
                   if args.detail else "Mandelbrot metaphor: GPT-4 (main cardioid and bulb), GPT-6 (fine branches omitted), Reality (800-iteration finite-resolution approximation).")
    metadata.add_text("Description", description)
    filename = "fractal-problem-detail.png" if args.detail else "fractal-problem.png"
    if args.colour:
        filename = "fractal-problem-colour.png"
        metadata.add_text("Description", "Full Mandelbrot views at the same scale, with 7, 18, and 800 escape-time iterations. Colour reveals progressively finer boundary structure. Model labels illustrate a metaphor, not measured capabilities.")
    output = ROOT / "assets" / filename
    canvas.save(output, pnginfo=metadata, optimize=True)
    print(f"Saved {output} ({WIDTH} x {HEIGHT})")


if __name__ == "__main__":
    main()
