#!/usr/bin/env python3
"""Livestream react emoji overlay — outputs transparent WEBM via FFmpeg."""

import math
import random
import subprocess
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


def render_frame(frame_idx: int, particles: list[dict], sprites: list[Image.Image]) -> Image.Image:
    """Render one RGBA frame."""
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))

    t = frame_idx / FPS
    for p in particles:
        cycle_t = (t + p["phase"]) % DURATION_SEC
        state = particle_state(p, cycle_t)

        size = max(1, int(EMOJI_SIZE * state["scale"]))
        sprite = sprites[p["emoji_idx"]].resize((size, size), Image.LANCZOS)

        r, g, b, a = sprite.split()
        a = a.point(lambda v: int(v * state["alpha"]))
        sprite = Image.merge("RGBA", (r, g, b, a))

        paste_x = int(state["x"] - size / 2)
        paste_y = int(state["y"] - size / 2)
        canvas.paste(sprite, (paste_x, paste_y), sprite)

    return canvas


def main():
    output = "react_overlay.webm"
    sprites = fetch_sprites()
    particles = build_particles()

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-vcodec", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-b:v", "0",
        "-crf", "20",
        "-an",
        output,
    ]

    print(f"Rendering {TOTAL_FRAMES} frames → {output} ...", file=sys.stderr)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for frame_idx in range(TOTAL_FRAMES):
            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx}/{TOTAL_FRAMES}", file=sys.stderr)
            frame = render_frame(frame_idx, particles, sprites)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        print("FFmpeg pipe broke — check FFmpeg output above.", file=sys.stderr)
        sys.exit(1)

    print(f"Done: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
