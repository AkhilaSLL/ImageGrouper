"""Generate assets/app.ico and assets/app.png - a 'grouped images' icon."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter

SS = 4                                   # supersampling factor
S = 256 * SS
ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48),
             (64, 64), (128, 128), (256, 256)]

AMBER = (240, 163, 94)                   # matches theme.EXEMPLAR_FG
WHITE = (255, 255, 255)


def lerp(a: tuple[int, int, int], b: tuple[int, int, int],
         t: float) -> tuple[int, ...]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def build() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Background tile: vertical teal -> blue gradient, rounded corners.
    top, bottom = (45, 190, 176), (79, 140, 255)
    gradient = Image.new("RGB", (1, S))
    for y in range(S):
        gradient.putpixel((0, y), lerp(top, bottom, y / S))
    gradient = gradient.resize((S, S))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, S - 1, S - 1], radius=int(58 * SS), fill=255)
    img.paste(gradient, (0, 0), mask)

    # Two columns of cards: one group tinted, the other plain.  Even at 16px
    # that still reads as "these belong together, those don't".
    tile = int(74 * SS)
    gap_small = int(12 * SS)              # inside a group
    gap_large = int(26 * SS)              # between groups
    total_w = tile * 2 + gap_large
    total_h = tile * 2 + gap_small
    x0 = (S - total_w) // 2
    y0 = (S - total_h) // 2

    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shade = ImageDraw.Draw(shadow)
    for column in range(2):
        for row in range(2):
            x = x0 + column * (tile + gap_large)
            y = y0 + row * (tile + gap_small)
            shade.rounded_rectangle([x, y + int(5 * SS), x + tile,
                                     y + tile + int(5 * SS)],
                                    radius=int(16 * SS), fill=(0, 0, 0, 90))
    img = Image.alpha_composite(
        img, shadow.filter(ImageFilter.GaussianBlur(6 * SS)))

    draw = ImageDraw.Draw(img)
    for column in range(2):
        for row in range(2):
            x = x0 + column * (tile + gap_large)
            y = y0 + row * (tile + gap_small)
            draw.rounded_rectangle([x, y, x + tile, y + tile],
                                   radius=int(16 * SS),
                                   fill=AMBER if column == 0 else WHITE)

            # A sun and a hill, so the tiles read as pictures rather than
            # plain swatches.
            ink = (255, 255, 255, 190) if column == 0 else (79, 140, 255, 210)
            radius = int(7 * SS)
            cx, cy = x + int(22 * SS), y + int(24 * SS)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=ink)
            base = y + tile - int(18 * SS)
            draw.polygon([(x + int(12 * SS), base),
                          (x + int(38 * SS), y + int(34 * SS)),
                          (x + tile - int(10 * SS), base)], fill=ink)

    return img.resize((256, 256), Image.Resampling.LANCZOS)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    icon = build()
    ico_path = os.path.join(ASSETS, "app.ico")
    png_path = os.path.join(ASSETS, "app.png")
    icon.save(ico_path, sizes=ICO_SIZES)
    icon.save(png_path)
    print(f"wrote {ico_path} ({os.path.getsize(ico_path)} bytes)")
    print(f"wrote {png_path} ({os.path.getsize(png_path)} bytes)")


if __name__ == "__main__":
    main()
