"""
VLM-based realism evaluation using Claude claude-sonnet-4-6.
Requires ANTHROPIC_API_KEY environment variable.

Usage (manual — not in pytest):
    import numpy as np, os
    os.environ['ANTHROPIC_API_KEY'] = 'your_key'
    from model.vlm_eval import score_samples
    dummy = np.zeros((5, 64, 64), dtype=np.float32); dummy[0] = 1.0
    print(score_samples([dummy]))
"""
import base64
import io
import os

import anthropic
import numpy as np
from PIL import Image

EVAL_PROMPT = """You are evaluating a machine-learning-generated road network image.
The image shows a top-down view of roads for a US suburban area, rendered as colored lines on a black background.
Different colors represent different road types (white=highway, red=arterial, orange=collector, gray=residential).

Rate this image 1-10 on REALISM:
- 10: Looks like a real US suburb from OpenStreetMap. Roads connect properly, spacing is realistic, hierarchy makes sense.
- 7-9: Mostly realistic with minor issues (slightly off spacing or a few disconnections).
- 4-6: Recognizable road patterns but notable problems (some psychedelic artifacts, unrealistic density, or poor connectivity).
- 1-3: Clearly machine-generated garbage: swirling artifacts, disconnected fragments, implausible patterns.

Respond with ONLY:
SCORE: <integer 1-10>
ISSUES: <one sentence describing the main problems, or "none" if score >= 8>"""

# Channel index → RGB color
_COLORS = {
    0: (0, 0, 0),          # background: black
    1: (128, 128, 128),    # residential: gray
    2: (255, 165, 0),      # tertiary: orange
    3: (255, 50, 50),      # primary/secondary: red
    4: (255, 255, 255),    # motorway/trunk: white
}


def road_tensor_to_rgb(road: np.ndarray) -> np.ndarray:
    """Convert 5-channel one-hot road array (5, H, W) to RGB (H, W, 3) uint8."""
    argmax = road.argmax(axis=0)  # (H, W)
    rgb = np.zeros((*argmax.shape, 3), dtype=np.uint8)
    for ch, color in _COLORS.items():
        rgb[argmax == ch] = color
    return rgb


def score_samples(road_arrays: list, model: str = "claude-sonnet-4-6") -> list:
    """
    Score a list of generated road layouts for realism using the Claude API.

    Parameters
    ----------
    road_arrays : list of np.ndarray, each shape (5, H, W) one-hot float32
    model       : Claude model ID

    Returns
    -------
    list of {"score": int, "issues": str} dicts, one per input array
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []

    for road in road_arrays:
        rgb = road_tensor_to_rgb(road)
        img = Image.fromarray(rgb).resize((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": EVAL_PROMPT},
                    ],
                }
            ],
        )

        text = response.content[0].text.strip()
        score_line = next(
            (l for l in text.splitlines() if l.startswith("SCORE:")), "SCORE: 0"
        )
        issues_line = next(
            (l for l in text.splitlines() if l.startswith("ISSUES:")), "ISSUES: parse error"
        )
        try:
            score = int(score_line.split(":")[1].strip())
        except ValueError:
            score = 0
        issues = issues_line.split(":", 1)[1].strip()
        results.append({"score": score, "issues": issues})

    return results
