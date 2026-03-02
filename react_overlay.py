#!/usr/bin/env python3
"""Livestream react emoji overlay — outputs transparent WEBM via FFmpeg."""

import math
import random
import sys
from pathlib import Path

import requests
from PIL import Image

# --- Constants (tweak these) ---
WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION_SEC = 8
TOTAL_FRAMES = FPS * DURATION_SEC

EMOJI_COUNT = 20          # simultaneous floating particles
EMOJI_SIZE = 80           # base px size
SPEED_RANGE = (120, 180)  # px/sec, controls travel time (unused directly; see phase)
DRIFT_AMP_RANGE = (20, 60)   # px, lateral wobble amplitude
DRIFT_FREQ_RANGE = (0.3, 0.8)  # Hz, lateral wobble speed
POP_FRAMES = 12           # frames for pop-in animation (~0.4s)
FADE_ZONE = HEIGHT * 0.15  # top fraction where emojis fade out

EMOJI_CODES = ["1f602", "2764", "1f44d", "1f631"]  # 😂 ❤️ 👍 😱
SPRITE_DIR = Path("emoji_sprites")
TWEMOJI_BASE = "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72"


def fetch_sprites() -> list[Image.Image]:
    """Download twemoji PNGs if not cached, return list of RGBA Images."""
    SPRITE_DIR.mkdir(exist_ok=True)
    sprites = []
    for code in EMOJI_CODES:
        path = SPRITE_DIR / f"{code}.png"
        if not path.exists():
            url = f"{TWEMOJI_BASE}/{code}.png"
            print(f"Fetching {url} ...", file=sys.stderr)
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            path.write_bytes(r.content)
        img = Image.open(path).convert("RGBA")
        sprites.append(img)
    return sprites


def pop_scale(age_frames: int) -> float:
    """Spring pop-in: 0 → 1.3 → 1.0 over POP_FRAMES frames."""
    if age_frames >= POP_FRAMES:
        return 1.0
    t = age_frames / POP_FRAMES
    if t < 0.4:
        return (t / 0.4) * 1.3
    else:
        return 1.3 - ((t - 0.4) / 0.6) * 0.3


def build_particles(seed: int = 42) -> list[dict]:
    """Build EMOJI_COUNT particles with deterministic random params."""
    rng = random.Random(seed)
    particles = []
    for i in range(EMOJI_COUNT):
        particles.append({
            "phase": i * DURATION_SEC / EMOJI_COUNT,
            "emoji_idx": rng.randint(0, len(EMOJI_CODES) - 1),
            "base_x": WIDTH * (i + 0.5) / EMOJI_COUNT + rng.uniform(-30, 30),
            "drift_amp": rng.uniform(*DRIFT_AMP_RANGE),
            "drift_freq": rng.uniform(*DRIFT_FREQ_RANGE),
            "drift_phase": rng.uniform(0, 2 * math.pi),
        })
    return particles


def particle_state(p: dict, cycle_t: float) -> dict:
    """Compute x, y, scale, alpha for a particle at cycle_t seconds into its cycle."""
    progress = cycle_t / DURATION_SEC

    travel = HEIGHT + 2 * EMOJI_SIZE
    y = HEIGHT + EMOJI_SIZE - progress * travel

    x = p["base_x"] + p["drift_amp"] * math.sin(
        2 * math.pi * p["drift_freq"] * cycle_t + p["drift_phase"]
    )

    age_frames = int(cycle_t * FPS)
    scale = pop_scale(age_frames)

    alpha = 1.0
    if y < FADE_ZONE:
        alpha = max(0.0, y / FADE_ZONE)
    if age_frames < 3:
        alpha *= age_frames / 3.0

    return {"x": x, "y": y, "scale": scale, "alpha": alpha}
