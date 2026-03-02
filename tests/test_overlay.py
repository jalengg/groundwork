import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from react_overlay import fetch_sprites, EMOJI_CODES


def test_sprites_load():
    sprites = fetch_sprites()
    assert len(sprites) == len(EMOJI_CODES)
    for s in sprites:
        assert s.mode == "RGBA"
        assert s.size == (72, 72)


from react_overlay import pop_scale, particle_state, build_particles, POP_FRAMES, HEIGHT, DURATION_SEC, FPS, render_frame, WIDTH


def test_pop_scale_starts_at_zero():
    assert pop_scale(0) == 0.0


def test_pop_scale_overshoots():
    peak = max(pop_scale(f) for f in range(POP_FRAMES))
    assert peak > 1.0


def test_pop_scale_settles():
    assert pop_scale(POP_FRAMES) == 1.0
    assert pop_scale(POP_FRAMES + 10) == 1.0


def test_particle_state_y_at_start():
    p = build_particles()[0]
    p["phase"] = 0.0
    state = particle_state(p, cycle_t=0.0)
    assert state["y"] > HEIGHT


def test_particle_state_y_at_end():
    p = build_particles()[0]
    p["phase"] = 0.0
    state = particle_state(p, cycle_t=DURATION_SEC - 0.01)
    assert state["y"] < 0


def test_particle_alpha_fades_near_top():
    p = build_particles()[0]
    p["phase"] = 0.0
    state = particle_state(p, cycle_t=DURATION_SEC * 0.98)
    assert state["alpha"] < 0.5


def test_render_frame_is_rgba_correct_size():
    sprites = fetch_sprites()
    particles = build_particles()
    frame = render_frame(frame_idx=0, particles=particles, sprites=sprites)
    assert frame.mode == "RGBA"
    assert frame.size == (WIDTH, HEIGHT)


def test_render_frame_has_nonzero_alpha():
    sprites = fetch_sprites()
    particles = build_particles()
    frame = render_frame(frame_idx=15, particles=particles, sprites=sprites)
    pixels = list(frame.getdata())
    alpha_values = [p[3] for p in pixels]
    assert max(alpha_values) > 0
