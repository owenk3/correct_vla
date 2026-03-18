"""
policies/correctvla/correction.py

Our method: human-correction-based context injection for VLA failure recovery.

Standalone module — no dependency on eval scripts.
Inputs: standard types (paths, PIL images, strings).
Outputs: correction text, ICA-compatible context dict.

Usage (in any eval pipeline):
    from policies.correctvla.correction import load_corrections, build_ours_context

    corrections = load_corrections("corrections/")
    # corrections[task_id] = {
    #   "base": str,          ← correction.txt
    #   "attempt_1": str,     ← attempt_1.txt (optional)
    #   "version": "v2",
    #   "path": "corrections/task_18/v2",
    # }
    ctx = build_ours_context(mp4_path, task_description, corrections[task_id]["base"])
    # ctx["human_correction"], ctx["initial_img"], ctx["final_img"], ctx["assessment"]
    # → pass ctx into your ICA save_top_level_ica_dir call
"""

import os
import re
from typing import Optional


def _latest_version_dir(task_dir: str) -> Optional[str]:
    """Return path to the highest-numbered v{N}/ subdir, or None."""
    versions = [
        d for d in os.listdir(task_dir)
        if d.startswith("v") and d[1:].isdigit()
        and os.path.isdir(os.path.join(task_dir, d))
    ]
    if not versions:
        return None
    latest = max(versions, key=lambda d: int(d[1:]))
    return os.path.join(task_dir, latest)


def load_corrections(corrections_dir: str, suite: str = "") -> dict:
    """
    Load task-level corrections.

    Search order (later overrides earlier):
      1. {corrections_dir}/task_{id}/...        (legacy, suite-agnostic)
      2. {corrections_dir}/{suite}/task_{id}/... (suite-scoped, priority)

    Returns:
      {task_id (int): { "base": str, "version": str, "path": str, ... }}
    """
    result = {}
    scan_dirs = [corrections_dir]
    if suite:
        suite_dir = os.path.join(corrections_dir, suite)
        if os.path.isdir(suite_dir):
            scan_dirs.append(suite_dir)

    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for entry in os.listdir(scan_dir):
            if not entry.startswith("task_"):
                continue
            entry_path = os.path.join(scan_dir, entry)

            # New versioned layout: task_{id}/ directory
            if os.path.isdir(entry_path):
                try:
                    task_id = int(entry[len("task_"):])
                except ValueError:
                    continue
                ver_dir = _latest_version_dir(entry_path)
                if ver_dir is None:
                    continue
                record = {"version": os.path.basename(ver_dir), "path": ver_dir}
                base_path = os.path.join(ver_dir, "correction.txt")
                if os.path.exists(base_path):
                    with open(base_path) as f:
                        record["base"] = f.read().strip()
                params_path = os.path.join(ver_dir, "correction_params.json")
                if os.path.exists(params_path):
                    import json as _json
                    with open(params_path) as f:
                        record["correction_params"] = _json.load(f)
                for fname in sorted(os.listdir(ver_dir)):
                    if fname.startswith("attempt_") and fname.endswith(".txt"):
                        key = fname[:-len(".txt")]
                        with open(os.path.join(ver_dir, fname)) as f:
                            record[key] = f.read().strip()
                if record.get("base"):
                    result[task_id] = record
                    print(f"  [correctvla] task_id={task_id} ({record['version']}): {record['base'][:60]}...")

            # Legacy flat layout: task_{id}.txt
            elif entry.endswith(".txt"):
                try:
                    task_id = int(entry[len("task_"):-len(".txt")])
                except ValueError:
                    continue
                if task_id in result:
                    continue  # versioned already loaded
                with open(entry_path) as f:
                    text = f.read().strip()
                if text:
                    result[task_id] = {"base": text, "version": "legacy", "path": entry_path}
                    print(f"  [correctvla] task_id={task_id} (legacy): {text[:60]}...")

    return result


# ---------------------------------------------------------------------------
# Timed-feedback parsing (AdaptVLA format → correction_params dict)
# ---------------------------------------------------------------------------

_DIR_MAP = {
    "left": ("y", "+1"), "right": ("y", "-1"),
    "forward": ("x", "+1"), "backward": ("x", "-1"), "back": ("x", "-1"),
    "up": ("z", "+1"), "down": ("z", "-1"),
}

_MAG_TERMS = {"little", "little_more", "slightly", "slightly_more", "more", "much"}

def parse_timed_feedback(feedback: str) -> dict:
    """
    Parse timed_feedback string into correction_params dict.

    Supports two magnitude formats:
      "down slightly 2.2~2.6s"      → magnitude_term (resolved at runtime via action_std)
      "down +0.2 2.2~2.6s"          → magnitude_value (used directly)
    Axes separated by ';'

    Returns: {"axes": [{"dimension": ..., "direction": ..., ...}, ...]}
    """
    axes = []
    for part in feedback.split(";"):
        part = part.strip().rstrip(";")
        if not part:
            continue
        tokens = part.split()
        if len(tokens) < 2:
            continue
        direction_word = tokens[0].lower()
        if direction_word not in _DIR_MAP:
            continue
        dim, sign = _DIR_MAP[direction_word]
        mag_tok = tokens[1].lower().lstrip("+")
        t_start, t_end = None, None
        for tok in tokens[2:]:
            m = re.match(r'([\d.]+)~([\d.]+)s?', tok)
            if m:
                t_start, t_end = float(m.group(1)), float(m.group(2))
        entry = {
            "dimension": dim, "direction": sign,
            "t_start": t_start, "t_end": t_end,
            "mode": "add",
        }
        if mag_tok in _MAG_TERMS:
            entry["magnitude_term"] = mag_tok
            entry["magnitude_value"] = 0.0
        else:
            entry["magnitude_value"] = abs(float(mag_tok))
        axes.append(entry)
    return {"axes": axes}


def load_episode_corrections(filepath: str) -> dict:
    """
    Load per-episode corrections from human_corrections txt file.

    Returns: {(suite, episode): {"base": str, "correction_params": dict}}
    """
    result = {}
    if not os.path.isfile(filepath):
        return result
    current_suite = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and "]" in line:
                current_suite = line.split("]")[0].lstrip("[").strip()
                continue
            m = re.match(r'episode=(\d+)\s*\|\s*timed_feedback:\s*"(.+)"', line)
            if m and current_suite:
                ep = int(m.group(1))
                feedback = m.group(2)
                params = parse_timed_feedback(feedback)
                result[(current_suite, ep)] = {
                    "base": feedback,
                    "correction_params": params,
                }
    print(f"  [correctvla] Loaded {len(result)} episode corrections from {os.path.basename(filepath)}")
    return result


def load_episode_corrections_from_dirs(corrections_dir: str, model_family: str) -> dict:
    """
    Load per-episode corrections from dir structure:
      corrections/{model_family}/{suite}/ep_{N}/v{latest}/correction_params.json

    Returns: {(suite, episode): {"base": str, "correction_params": dict, "version": str, "path": str}}
    """
    import json as _json
    result = {}
    model_dir = os.path.join(corrections_dir, model_family)
    if not os.path.isdir(model_dir):
        return result
    for suite in sorted(os.listdir(model_dir)):
        suite_dir = os.path.join(model_dir, suite)
        if not os.path.isdir(suite_dir):
            continue
        for ep_entry in sorted(os.listdir(suite_dir)):
            if not ep_entry.startswith("ep_"):
                continue
            ep = int(ep_entry[3:])
            ep_dir = os.path.join(suite_dir, ep_entry)
            ver_dir = _latest_version_dir(ep_dir)
            if ver_dir is None:
                continue
            record = {"version": os.path.basename(ver_dir), "path": ver_dir}
            params_path = os.path.join(ver_dir, "correction_params.json")
            if os.path.exists(params_path):
                with open(params_path) as f:
                    record["correction_params"] = _json.load(f)
            base_path = os.path.join(ver_dir, "correction.txt")
            if os.path.exists(base_path):
                with open(base_path) as f:
                    record["base"] = f.read().strip()
            if record.get("correction_params") or record.get("base"):
                result[(suite, ep)] = record
    if result:
        print(f"  [correctvla] Loaded {len(result)} episode corrections from {model_dir}/")
    return result


def build_ours_context(
    mp4_path: str,
    task_description: str,
    correction: str,
) -> dict:
    """
    Build failure context for 'ours' method from a failure mp4 + human correction.

    Uses the human correction as a lens to analyze the failure video (1 GPT call),
    producing a grounded diagnosis of what went wrong and how to fix it.

    Returns a dict with keys:
        initial_img      : PIL.Image
        final_img        : PIL.Image
        human_correction : str
        whathappened     : str  — correction-guided diagnosis from video
        reasoning        : str  — what behavioral change is needed
        assessment       : str  — same as whathappened (ICA compat)
    """
    import cv2
    from PIL import Image as PILImage
    from policies.vlm.vlm_hl.vlm_methods import critique_vla_video_with_correction, extract_correction_params

    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Too few frames in {os.path.basename(mp4_path)}")

    print(f"  [correctvla] Extracting structured correction parameters...")
    params = extract_correction_params(correction, task_description)
    print(f"  [correctvla] Params: dim={params.dimension} dir={params.direction} "
          f"mag={params.magnitude_term}({params.magnitude_value:.2f}) "
          f"t=[{params.t_start},{params.t_end}] reason={params.reason}")

    print(f"  [correctvla] Analyzing failure video through human correction lens...")
    whathappened = critique_vla_video_with_correction(frames, task_description, correction)
    reasoning = f"Human correction: {correction}\nVideo diagnosis: {whathappened}"

    return {
        "initial_img": PILImage.fromarray(frames[0]),
        "final_img": PILImage.fromarray(frames[-1]),
        "human_correction": correction,
        "correction_params": params,
        "whathappened": whathappened,
        "reasoning": reasoning,
        "assessment": whathappened,
    }
