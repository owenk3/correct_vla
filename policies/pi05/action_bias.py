"""
action_bias.py

Build an additive action bias + suppression mask from structured CorrectionParams.

Each axis entry supports two modes:
  - "mode": "add"      (default) — additive bias on top of VLA action
  - "mode": "suppress" — zero out VLA action for this dim during window (direction/magnitude ignored)
  - "mode": "override" — zero out VLA action AND add bias (suppress + add)

The bias follows a piecewise-linear temporal envelope ϕ(t):
  - 0 outside [t_start, t_end]
  - ramps up in 0.2s, holds at peak, ramps down in 0.2s

Returns (bias, suppress_mask):
  bias:          (N, 7) additive correction
  suppress_mask: (N, 7) multiply VLA actions by this before adding bias (1=keep, 0=zero out)

Final action = vla_actions * suppress_mask + bias

Dimension mapping (LIBERO / pi05 action space — WORLD coordinates):
  index 0: x      +X=Forward,   -X=Backward
  index 1: y      +Y=Right,     -Y=Left
  index 2: z      +Z=Up,        -Z=Down
  index 3: roll   rotation around X
  index 4: pitch  rotation around Y
  index 5: yaw    rotation around Z
  index 6: gripper  +1=open, -1=close
  NOTE: Video is rotated 180° (img[::-1,::-1]), so in the SAVED VIDEO:
    video-left = +Y (world right),  video-right = -Y (world left)
    video-forward = +X (world forward), video-backward = -X (world backward)

Time units: t_start/t_end are in VIDEO SECONDS (what a human sees).
Each env.step() = 1 video frame saved at 30 FPS: video_time = step / 30.
"""

import numpy as np
from typing import Optional

_DIM_MAP = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5, "gripper": 6}

# Video FPS: each env.step() = 1 video frame saved at 30 FPS
_DEFAULT_ACTION_HZ = 30.0

_MAGNITUDE_TERM_RATIO = {"little": 0.5, "little_more": 0.75, "slightly": 1.0, "slightly_more": 1.5, "more": 2.0, "much": 4.0}
_RAMP_S = 0.2  # seconds for ramp-up and ramp-down


def _build_envelope(t_steps, t_start, t_end, n_steps, envelope_type="flat"):
    """Temporal envelope. Returns (n_steps,) float32.

    envelope_type:
      "flat" (default): ramp-up 0.2s, hold at 1, ramp-down 0.2s
      "ramp": linearly increases 0→1 from t_start→t_end, 0 outside
      "triangle": ramps 0→1 to midpoint, then 1→0, 0 outside (paper eq.1)
    """
    if t_start is None:
        return np.ones(n_steps, dtype=np.float32)
    t_start = float(t_start)
    t_end = float(t_end) if t_end is not None else float("inf")
    if envelope_type == "ramp":
        duration = max(t_end - t_start, 1e-6)
        return np.where(
            (t_steps < t_start) | (t_steps > t_end), 0.0,
            (t_steps - t_start) / duration,
        ).astype(np.float32)
    if envelope_type == "triangle":
        duration = max(t_end - t_start, 1e-6)
        t_mid = (t_start + t_end) / 2.0
        return np.where(
            (t_steps < t_start) | (t_steps > t_end), 0.0,
            np.where(
                t_steps <= t_mid,
                2.0 * (t_steps - t_start) / duration,
                2.0 * (t_end - t_steps) / duration,
            )
        ).astype(np.float32)
    ramp = _RAMP_S
    return np.where(
        t_steps < t_start, 0.0,
        np.where(
            t_steps > t_end, 0.0,
            np.where(
                t_steps <= t_start + ramp,
                (t_steps - t_start) / ramp,
                np.where(
                    t_steps >= t_end - ramp,
                    (t_end - t_steps) / ramp,
                    1.0,
                )
            )
        )
    ).astype(np.float32)


def _build_single_axis_bias(
    params: dict,
    n_steps: int,
    step_offset: int,
    action_hz: float,
    action_std: Optional[np.ndarray],
) -> tuple:
    """Build bias + suppress_mask for a single axis. Returns ((n_steps,7), (n_steps,7))."""
    bias = np.zeros((n_steps, 7), dtype=np.float32)
    suppress_mask = np.ones((n_steps, 7), dtype=np.float32)

    dim = params.get("dimension", "unknown")
    if dim not in _DIM_MAP or dim == "unknown":
        return bias, suppress_mask

    dim_idx = _DIM_MAP[dim]
    mode = params.get("mode", "add")  # "add" | "suppress" | "override"

    t_steps = (step_offset + np.arange(n_steps, dtype=np.float32)) / action_hz
    envelope = _build_envelope(t_steps, params.get("t_start"), params.get("t_end"), n_steps, params.get("envelope", "flat"))

    if mode in ("suppress", "override"):
        # Zero out VLA action for this dim during window
        suppress_mask[:, dim_idx] = 1.0 - envelope  # 0 where envelope=1 (full suppression)

    if mode in ("add", "override"):
        direction = params.get("direction", "unknown")
        if direction == "unknown":
            return bias, suppress_mask
        sign = 1.0 if direction == "+1" else -1.0
        term = params.get("magnitude_term", "unknown")
        mag_val = abs(float(params.get("magnitude_value") or 0.0))
        if mag_val != 0.0:
            magnitude = sign * mag_val
        elif action_std is not None and term in _MAGNITUDE_TERM_RATIO:
            magnitude = sign * _MAGNITUDE_TERM_RATIO[term] * float(action_std[dim_idx])
        else:
            magnitude = sign * 1.0
        bias[:, dim_idx] = envelope * magnitude

    return bias, suppress_mask


def build_action_bias(
    params,
    n_steps: int,
    step_offset: int = 0,
    action_hz: float = _DEFAULT_ACTION_HZ,
    action_std: Optional[np.ndarray] = None,
) -> tuple:
    """
    Build (bias, suppress_mask) arrays of shape (n_steps, 7).

    Apply as: corrected_actions = vla_actions * suppress_mask + bias

    Args:
        params: CorrectionParams dict with "axes" list, or single-axis dict.
        n_steps: number of action steps in the chunk
        step_offset: current step count (env steps elapsed)
        action_hz: frame rate (30 FPS video time)
        action_std: (7,) array of per-dim action std from postprocessor stats.

    Returns:
        (bias, suppress_mask): both np.ndarray of shape (n_steps, 7)
    """
    axes = params.get("axes") if isinstance(params, dict) else None
    if axes:
        bias = np.zeros((n_steps, 7), dtype=np.float32)
        suppress_mask = np.ones((n_steps, 7), dtype=np.float32)
        for ax in axes:
            b, m = _build_single_axis_bias(ax, n_steps, step_offset, action_hz, action_std)
            bias += b
            suppress_mask *= m  # combine masks multiplicatively
        return bias, suppress_mask
    return _build_single_axis_bias(params, n_steps, step_offset, action_hz, action_std)
