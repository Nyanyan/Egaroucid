#!/usr/bin/env python3
"""Generate PNG visualizations for the board_demo.py default position.

The images show values on every legal move for:

1. the legal-masked Policy Network output,
2. the raw Egaroucid for Console evaluation,
3. the Policy made by applying softmax to the Console evaluation, and
4. geometric blends for alpha=0.0, 0.2, ..., 1.0, and
5. the empirical human move probabilities from matching WTHOR positions.

Policy values are rendered with exactly three digits after the decimal point.
Raw Egaroucid evaluation values are rendered as integers.
Every image except the raw evaluation uses a white-to-orange heatmap on legal
move cells, with RGB(255, 75, 0) representing the largest value in that image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
BOARD = "------------------XO-O----XXOO-----XOX-----OOX------O-----------"
BOARD_WIDTH = 8
SUPERSAMPLE = 2
BLEND_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
HEATMAP_BASE_COLOR = (255, 75, 0)


# These are the legal-masked values printed by board_demo.py for the default
# position.  Keeping full precision here makes the displayed 3-decimal values
# reproducible rather than dependent on already-rounded text.
POLICY_NETWORK_VALUES = {
    "d8": 0.0002338472841,
    "f7": 0.3091001809,
    "d7": 0.5727080107,
    "c6": 0.1054185107,
    "g4": 0.002095479984,
    "e3": 0.006799378898,
    "g2": 0.000001572135375,
    "f2": 0.0002392599854,
    "e2": 0.0002856075007,
    "d2": 0.002880484331,
    "c2": 0.0002377111377,
}

EGAROUCID_RAW_VALUES = {
    "d8": -8.0,
    "f7": -1.0,
    "d7": -2.0,
    "c6": -2.0,
    "g4": -11.0,
    "e3": -5.0,
    "g2": -25.0,
    "f2": -9.0,
    "e2": -8.0,
    "d2": -5.0,
    "c2": -12.0,
}

EGAROUCID_POLICY_VALUES = {
    "d8": 0.0005138487904,
    "f7": 0.5635036230,
    "d7": 0.2073013932,
    "c6": 0.2073013932,
    "g4": 0.00002558302185,
    "e3": 0.01032092888,
    "g2": 0.0000000000212730198,
    "f2": 0.0001890343992,
    "e2": 0.0005138487904,
    "d2": 0.01032092888,
    "c2": 0.000009411470273,
}

# Empirical probabilities from the 12,270 WTHOR position samples whose board
# and side to move exactly match this position.
WTHOR_HUMAN_POLICY_VALUES = {
    "d8": 0.0001629991850,
    "f7": 0.2286878566,
    "d7": 0.6132029340,
    "c6": 0.1443357783,
    "g4": 0.0004889975550,
    "e3": 0.009453952730,
    "g2": 0.0,
    "f2": 0.00008149959250,
    "e2": 0.0002444987775,
    "d2": 0.003178484108,
    "c2": 0.0001629991850,
}


@dataclass(frozen=True)
class ImageSpec:
    filename: str
    values: Dict[str, float]
    value_format: str = ".3f"
    heatmap: bool = True


def geometric_blend_values(alpha: float) -> Dict[str, float]:
    """Return normalize(network**alpha * egaroucid**(1-alpha))."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    if POLICY_NETWORK_VALUES.keys() != EGAROUCID_POLICY_VALUES.keys():
        raise ValueError("Policy Network and Egaroucid moves do not match")

    if alpha == 0.0:
        unnormalized = dict(EGAROUCID_POLICY_VALUES)
    elif alpha == 1.0:
        unnormalized = dict(POLICY_NETWORK_VALUES)
    else:
        unnormalized = {
            coord: (
                POLICY_NETWORK_VALUES[coord] ** alpha
                * EGAROUCID_POLICY_VALUES[coord] ** (1.0 - alpha)
            )
            for coord in POLICY_NETWORK_VALUES
        }
    total = sum(unnormalized.values())
    if total <= 0.0:
        raise ValueError("blended Policy has no positive probability")
    return {coord: value / total for coord, value in unnormalized.items()}


BASE_IMAGE_SPECS = (
    ImageSpec("01_policy_network.png", POLICY_NETWORK_VALUES),
    ImageSpec(
        "02_egaroucid_raw_evaluation.png",
        EGAROUCID_RAW_VALUES,
        value_format=".0f",
        heatmap=False,
    ),
    ImageSpec("03_egaroucid_policy.png", EGAROUCID_POLICY_VALUES),
)
BLEND_IMAGE_SPECS = tuple(
    ImageSpec(
        f"{image_index:02d}_blend_alpha_{alpha:.1f}.png",
        geometric_blend_values(alpha),
    )
    for image_index, alpha in enumerate(BLEND_ALPHAS, start=4)
)
WTHOR_IMAGE_SPECS = (
    ImageSpec("10_wthor_human_move_probability.png", WTHOR_HUMAN_POLICY_VALUES),
)
IMAGE_SPECS = BASE_IMAGE_SPECS + BLEND_IMAGE_SPECS + WTHOR_IMAGE_SPECS


def coord_to_xy(coord: str) -> Tuple[int, int]:
    if (
        len(coord) != 2
        or not "a" <= coord[0].lower() <= "h"
        or not "1" <= coord[1] <= "8"
    ):
        raise ValueError(f"invalid board coordinate: {coord}")
    return ord(coord[0].lower()) - ord("a"), int(coord[1]) - 1


def iter_discs() -> Iterable[Tuple[int, int, str]]:
    if len(BOARD) != BOARD_WIDTH * BOARD_WIDTH:
        raise ValueError("BOARD must contain exactly 64 squares")
    for position, piece in enumerate(BOARD):
        if piece in ("X", "O"):
            yield position % BOARD_WIDTH, position // BOARD_WIDTH, piece


def font_candidates() -> Sequence[Path]:
    windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
    return (
        windows_dir / "Fonts" / "arialbd.ttf",
        windows_dir / "Fonts" / "segoeuib.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    )


def load_font(size: int, explicit_font: Optional[Path]) -> ImageFont.ImageFont:
    candidates = (explicit_font,) if explicit_font is not None else font_candidates()
    for path in candidates:
        if path is not None and path.is_file():
            return ImageFont.truetype(str(path), size=size)
    if explicit_font is not None:
        raise FileNotFoundError(f"font file not found: {explicit_font}")
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fitted_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    maximum_width: int,
    maximum_height: int,
    initial_size: int,
    explicit_font: Optional[Path],
) -> ImageFont.ImageFont:
    for size in range(initial_size, 7, -2):
        font = load_font(size, explicit_font)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= maximum_width and bottom - top <= maximum_height:
            return font
    return load_font(8, explicit_font)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    box_left, box_top, box_right, box_bottom = box
    x = (box_left + box_right - text_width) / 2 - left
    y = (box_top + box_bottom - text_height) / 2 - top
    draw.text((x, y), text, font=font, fill="black")


def heatmap_color(value: float, maximum_value: float) -> Tuple[int, int, int]:
    """Linearly blend white into the configured orange heatmap color."""
    if value < 0.0:
        raise ValueError("heatmap values must not be negative")
    intensity = 0.0 if maximum_value <= 0.0 else min(1.0, value / maximum_value)
    return tuple(
        round(255 + (channel - 255) * intensity)
        for channel in HEATMAP_BASE_COLOR
    )


def render_image(
    spec: ImageSpec,
    output_path: Path,
    output_size: int,
    explicit_font: Optional[Path],
) -> None:
    if output_size < 512:
        raise ValueError("output size must be at least 512 pixels")

    canvas_size = output_size * SUPERSAMPLE
    desired_margin = canvas_size // 32
    cell_size = (canvas_size - 2 * desired_margin) // BOARD_WIDTH
    board_size = cell_size * BOARD_WIDTH
    margin = (canvas_size - board_size) // 2
    board_left = margin
    board_top = margin
    board_right = board_left + board_size
    board_bottom = board_top + board_size

    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    line_width = max(4, canvas_size // 256)
    outer_line_width = max(line_width + 2, canvas_size // 192)
    corner_radius = max(16, cell_size // 2)

    draw.rounded_rectangle(
        (board_left, board_top, board_right, board_bottom),
        radius=corner_radius,
        fill="#ffffff",
        outline="black",
        width=outer_line_width,
    )
    if spec.heatmap:
        maximum_value = max(spec.values.values(), default=0.0)
        for coord, value in spec.values.items():
            x, y = coord_to_xy(coord)
            draw.rectangle(
                (
                    board_left + x * cell_size,
                    board_top + y * cell_size,
                    board_left + (x + 1) * cell_size,
                    board_top + (y + 1) * cell_size,
                ),
                fill=heatmap_color(value, maximum_value),
            )
    for line in range(1, BOARD_WIDTH):
        x = board_left + line * cell_size
        y = board_top + line * cell_size
        draw.line(
            (x, board_top, x, board_bottom),
            fill="black",
            width=line_width,
        )
        draw.line(
            (board_left, y, board_right, y),
            fill="black",
            width=line_width,
        )

    # Four standard board reference points, matching the supplied example.
    point_radius = max(5, cell_size // 12)
    for grid_x, grid_y in ((2, 2), (6, 2), (2, 6), (6, 6)):
        center_x = board_left + grid_x * cell_size
        center_y = board_top + grid_y * cell_size
        draw.ellipse(
            (
                center_x - point_radius,
                center_y - point_radius,
                center_x + point_radius,
                center_y + point_radius,
            ),
            fill="black",
        )

    disc_radius = int(cell_size * 0.39)
    disc_outline_width = max(3, cell_size // 28)
    for x, y, piece in iter_discs():
        center_x = board_left + x * cell_size + cell_size // 2
        center_y = board_top + y * cell_size + cell_size // 2
        disc_box = (
            center_x - disc_radius,
            center_y - disc_radius,
            center_x + disc_radius,
            center_y + disc_radius,
        )
        if piece == "X":
            draw.ellipse(disc_box, fill="black")
        else:
            draw.ellipse(
                disc_box,
                fill="white",
                outline="black",
                width=disc_outline_width,
            )

    for coord, value in spec.values.items():
        x, y = coord_to_xy(coord)
        text = format(value, spec.value_format)
        box = (
            board_left + x * cell_size,
            board_top + y * cell_size,
            board_left + (x + 1) * cell_size,
            board_top + (y + 1) * cell_size,
        )
        font = fitted_font(
            draw,
            text,
            maximum_width=int(cell_size * 0.86),
            maximum_height=int(cell_size * 0.48),
            initial_size=int(cell_size * 0.38),
            explicit_font=explicit_font,
        )
        draw_centered_text(draw, box, text, font)

    # Redraw the outer edge last so grid endpoints remain inside the border.
    draw.rounded_rectangle(
        (board_left, board_top, board_right, board_bottom),
        radius=corner_radius,
        outline="black",
        width=outer_line_width,
    )
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(
        (output_size, output_size),
        resample=resampling.LANCZOS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Policy Network, Egaroucid, alpha-blended, and WTHOR "
            "human-move PNG board images from the board-demo values."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="square PNG width and height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="optional TrueType/OpenType font file",
    )
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    for spec in IMAGE_SPECS:
        output_path = args.output_dir / spec.filename
        render_image(spec, output_path, args.size, args.font)
        print(output_path.resolve())


if __name__ == "__main__":
    main()
