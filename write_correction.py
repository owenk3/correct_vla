#!/usr/bin/env python3
"""
write_correction.py

Save human corrections into versioned per-episode structure:
  corrections/{model}/{suite}/ep_{N}/v{V}/
    correction.txt           ← timed feedback string
    correction_params.json   ← structured action bias params
    meta.json                ← metadata

CLI usage:
  python write_correction.py --list --model openvla --suite libero_spatial
  python write_correction.py --model openvla --suite libero_spatial --episode 96 \
    --feedback "down +0.2 2.2~2.6s"
  python write_correction.py --model openvla --suite libero_spatial --episode 96 \
    --feedback "down +0.2 2.2~2.6s; left +0.3 3.0~3.5s" --version v2
  python write_correction.py --model openvla --suite libero_spatial --episode 96 --rewrite \
    --feedback "down +0.3 2.0~2.8s"
"""

import argparse
import json
import os
import re
import sys

from policies.correctvla.correction import parse_timed_feedback

CORRECTIONS_DIR = "./corrections"
FAILURE_DIR = "./rollouts"

SUITE_TRIALS = {
    "libero_spatial": 50, "libero_object": 50,
    "libero_goal": 50, "libero_10": 50,
    "libero_90": 5,
}

_FNAME_RE = re.compile(
    r"^(?P<ts>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})--(?P<model>[^-]+)--episode=(?P<ep>\d+)--success=(?P<ok>\w+)--task=(?P<task>.+)\.mp4$"
)


def _latest_version(ep_dir: str) -> str | None:
    if not os.path.isdir(ep_dir):
        return None
    versions = [d for d in os.listdir(ep_dir) if d.startswith("v") and d[1:].isdigit()
                and os.path.isdir(os.path.join(ep_dir, d))]
    return max(versions, key=lambda d: int(d[1:])) if versions else None


def _next_version(ep_dir: str) -> str:
    cur = _latest_version(ep_dir)
    n = int(cur[1:]) + 1 if cur else 1
    return f"v{n}"


def _list_failures(model: str, suite: str | None):
    """List failure episodes from rollout dirs."""
    patterns = [
        os.path.join(FAILURE_DIR, f"{model}-in/failures"),
        os.path.join(FAILURE_DIR, f"{model}-ood/failures"),
        os.path.join(FAILURE_DIR, f"*{model}*/failures"),
    ]
    found_dirs = []
    for p in patterns:
        if os.path.isdir(p):
            found_dirs.append(p)
    if not found_dirs:
        print(f"No failure dirs found for model={model}")
        return

    for fdir in found_dirs:
        print(f"\n=== {fdir} ===")
        rows = []
        for fname in sorted(os.listdir(fdir)):
            m = _FNAME_RE.match(fname)
            if not m or m.group("ok").lower() != "false":
                continue
            ep = int(m.group("ep"))
            task_desc = m.group("task").replace("-", " ")
            rows.append((ep, task_desc[:50]))
        for ep, desc in rows:
            ep_dir = os.path.join(CORRECTIONS_DIR, model, suite or "unknown", f"ep_{ep}")
            ver = _latest_version(ep_dir)
            tag = f"  [{ver}]" if ver else ""
            print(f"  ep={ep:>4}  {desc}{tag}")


def write_episode_correction(
    model: str, suite: str, episode: int, feedback: str,
    version: str | None = None, rewrite: bool = False,
    task_description: str | None = None,
) -> str:
    ep_dir = os.path.join(CORRECTIONS_DIR, model, suite, f"ep_{episode}")
    os.makedirs(ep_dir, exist_ok=True)

    if rewrite:
        ver = version or _latest_version(ep_dir)
        if not ver:
            print(f"[error] No existing version to rewrite for ep_{episode}")
            sys.exit(1)
    else:
        ver = version or _next_version(ep_dir)

    ver_dir = os.path.join(ep_dir, ver)
    os.makedirs(ver_dir, exist_ok=True)

    # correction.txt
    with open(os.path.join(ver_dir, "correction.txt"), "w") as f:
        f.write(feedback.strip() + "\n")

    # correction_params.json
    params = parse_timed_feedback(feedback)
    with open(os.path.join(ver_dir, "correction_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # meta.json
    trials = SUITE_TRIALS.get(suite, 50)
    task_id = (episode - 1) // trials
    meta = {
        "model_family": model, "task_suite": suite,
        "task_id": task_id, "episode": episode,
        "num_trials_per_task": trials, "version": ver,
        "feedback": feedback.strip(),
    }
    if task_description:
        meta["task"] = task_description
    with open(os.path.join(ver_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return ver_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write human corrections for correctvla.")
    parser.add_argument("--list", action="store_true", help="List failure episodes")
    parser.add_argument("--model", type=str, default="openvla", help="Model family (openvla/pi05)")
    parser.add_argument("--suite", type=str, help="Task suite (libero_spatial, libero_90, etc.)")
    parser.add_argument("--episode", type=int, help="Episode number")
    parser.add_argument("--feedback", type=str, help="Timed feedback string, e.g. 'down +0.2 2.2~2.6s'")
    parser.add_argument("--version", type=str, help="Force version (e.g. v2); default: auto-increment")
    parser.add_argument("--rewrite", action="store_true", help="Overwrite latest version instead of creating new")
    parser.add_argument("--task", type=str, help="Task description (optional, for meta.json)")
    args = parser.parse_args()

    if args.list:
        _list_failures(args.model, args.suite)
        sys.exit(0)

    if not (args.suite and args.episode and args.feedback):
        print("[error] Required: --suite, --episode, --feedback")
        print("  Example: python write_correction.py --model openvla --suite libero_spatial --episode 96 --feedback 'down +0.2 2.2~2.6s'")
        sys.exit(1)

    path = write_episode_correction(
        args.model, args.suite, args.episode, args.feedback,
        version=args.version, rewrite=args.rewrite, task_description=args.task,
    )
    params = parse_timed_feedback(args.feedback)
    print(f"Saved → {path}")
    print(f"  feedback: {args.feedback}")
    print(f"  axes: {len(params['axes'])}")
    for ax in params["axes"]:
        print(f"    {ax['dimension']} {ax['direction']} mag={ax['magnitude_value']} t={ax['t_start']}~{ax['t_end']}s")
