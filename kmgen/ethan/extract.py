"""
KMGen Step 1: Extract step-down coordinates from a Kaplan-Meier survival plot
and overlay red circles for visual verification.

Usage:
    python extract.py <image_path> [--output <output_path>] [--model <model_id>]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw


def encode_image(path: Path) -> tuple[str, str]:
    """Read and base64-encode an image, returning (data, media_type)."""
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix)
    if not media_type:
        sys.exit(f"Unsupported image format: {suffix}")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


EXTRACTION_PROMPT = """\
You are analyzing a Kaplan-Meier survival curve image.

Your task: identify every coordinate where the survival curve **steps down** \
(i.e., where a vertical drop occurs — an event/death). These are the points \
where the curve transitions from one horizontal segment to a lower horizontal \
segment.

For each step-down, report:
- x: the time value (horizontal axis) where the drop occurs
- y_before: the survival probability just before the drop (top of the vertical segment)
- y_after: the survival probability just after the drop (bottom of the vertical segment)

Also report the axis ranges so we can map to pixel coordinates:
- x_min, x_max: the range of the x-axis (time)
- y_min, y_max: the range of the y-axis (survival probability, usually 0 to 1.0)
- plot_bbox: the bounding box of the plot area in pixels [left, top, right, bottom] \
  (the area inside the axes, not including labels/title)

Return your answer as JSON with this exact schema:
{
  "x_min": <number>,
  "x_max": <number>,
  "y_min": <number>,
  "y_max": <number>,
  "plot_bbox": [<left_px>, <top_px>, <right_px>, <bottom_px>],
  "steps": [
    {"x": <number>, "y_before": <number>, "y_after": <number>},
    ...
  ]
}

Be precise. Examine the image carefully. Every visible step-down must be captured. \
Do not skip small drops. Return ONLY the JSON object, no other text."""


def extract_steps(image_path: Path, model: str) -> dict:
    """Send the image to Claude and extract step-down coordinates."""
    client = anthropic.Anthropic()
    image_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )

    # Extract JSON from the response
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    # Parse JSON — handle possible markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove opening fence line
        text = text.rsplit("```", 1)[0]  # remove closing fence
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Claude's response as JSON:\n{text}")
        raise SystemExit(1) from e


def data_to_pixel(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bbox: list[int],
) -> tuple[int, int]:
    """Convert data coordinates to pixel coordinates within the plot bbox."""
    left, top, right, bottom = bbox
    px = left + (x - x_min) / (x_max - x_min) * (right - left)
    # y-axis is inverted in pixel space (top = high value)
    py = top + (y_max - y) / (y_max - y_min) * (bottom - top)
    return int(round(px)), int(round(py))


def annotate_image(
    image_path: Path, result: dict, output_path: Path, radius: int = 6
) -> None:
    """Draw red circles on the step-down points of the KM curve."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    bbox = result["plot_bbox"]
    x_min, x_max = result["x_min"], result["x_max"]
    y_min, y_max = result["y_min"], result["y_max"]

    for step in result["steps"]:
        # Mark the point at the TOP of the drop (where the curve was before stepping down)
        px, py = data_to_pixel(step["x"], step["y_before"], x_min, x_max, y_min, y_max, bbox)
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            outline="red",
            width=2,
        )
        # Mark the point at the BOTTOM of the drop (where the curve lands)
        px2, py2 = data_to_pixel(step["x"], step["y_after"], x_min, x_max, y_min, y_max, bbox)
        draw.ellipse(
            [px2 - radius, py2 - radius, px2 + radius, py2 + radius],
            outline="red",
            width=2,
        )
        # Draw a thin red line connecting them (the vertical drop)
        draw.line([(px, py), (px2, py2)], fill="red", width=1)

    img.save(output_path)
    print(f"Annotated image saved to: {output_path}")
    print(f"Detected {len(result['steps'])} step-down events")


def main():
    parser = argparse.ArgumentParser(description="Extract KM plot step-down coordinates")
    parser.add_argument("image", type=Path, help="Path to the KM plot image")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output image path")
    parser.add_argument("--model", "-m", type=str, default="claude-opus-4-6", help="Claude model ID")
    parser.add_argument("--json", "-j", action="store_true", help="Also save raw JSON output")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")

    output = args.output or args.image.with_stem(args.image.stem + "_annotated")

    print(f"Analyzing: {args.image}")
    print(f"Model: {args.model}")

    result = extract_steps(args.image, args.model)

    if args.json:
        json_path = args.image.with_suffix(".json")
        json_path.write_text(json.dumps(result, indent=2))
        print(f"JSON saved to: {json_path}")

    print(f"\nAxis ranges: x=[{result['x_min']}, {result['x_max']}], y=[{result['y_min']}, {result['y_max']}]")
    print(f"Plot bbox (px): {result['plot_bbox']}")
    print(f"\nStep-down events ({len(result['steps'])}):")
    for i, s in enumerate(result["steps"], 1):
        print(f"  {i:3d}. t={s['x']:8.2f}  S: {s['y_before']:.4f} → {s['y_after']:.4f}  (Δ={s['y_before'] - s['y_after']:.4f})")

    annotate_image(args.image, result, output)


if __name__ == "__main__":
    main()
