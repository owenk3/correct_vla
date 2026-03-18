"""
vla_eval_failures.py

Re-runs failed episodes with corrective action bias.

Eval modes:
  --eval vanilla    Base VLA policy (pi0.5 / OpenVLA-OFT), no correction
  --eval ours       Human correction → structured params → action bias
  --eval llm_baseline   VLM watches failure video → generates correction → action bias

Usage:
    # Vanilla:
    python experiments/robot/libero/vla_eval_failures.py \
        --pretrained_checkpoint lerobot/pi05_libero_finetuned \
        --model_family pi05 --task_suite_name libero_90 --episode 127

    # Ours (human correction):
    python experiments/robot/libero/vla_eval_failures.py \
        --pretrained_checkpoint lerobot/pi05_libero_finetuned \
        --model_family pi05 --task_suite_name libero_90 --episode 127 \
        --eval ours

    # LLM-Baseline (VLM-generated correction):
    python experiments/robot/libero/vla_eval_failures.py \
        --pretrained_checkpoint lerobot/pi05_libero_finetuned \
        --model_family pi05 --task_suite_name libero_90 --episode 127 \
        --eval llm_baseline
"""

import os
import re
import sys
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import json
import time

import draccus
import numpy as np
import torch

torch_load_orig = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch_load_orig(f, *args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
import experiments.robot.robot_utils as _robot_utils
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.vla_eval import (
    TASK_MAX_STEPS,
    TaskSuite,
    _resize_image,
    initialize_model,
    log_message,
    process_action,
)

# Load config sections
from policies.vlm.config import get_vlm_config as _get_vlm_cfg, get_eval_config as _get_eval_cfg
from policies.correctvla.correction import (
    load_corrections as _load_task_corrections,
    load_episode_corrections as _load_ep_corrections,
    load_episode_corrections_from_dirs as _load_ep_corrections_dirs,
    parse_timed_feedback,
    build_ours_context,
)
_VLM_CFG = _get_vlm_cfg()
_EVAL_CFG = _get_eval_cfg()
TIMESTAMP_TO_SUITE: dict = _VLM_CFG["timestamp_to_suite"]

# Filename pattern: {timestamp}--{model}--episode={N}--success={bool}--task={desc}.mp4
_FNAME_RE = re.compile(
    r"^(?P<ts>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})--(?P<model>[^-]+)--episode=(?P<ep>\d+)--success=(?P<ok>\w+)--task=(?P<task>.+)\.mp4$"
)


def parse_failure_filename(fname: str):
    """Returns (timestamp, episode_int) or None if not a failure file."""
    m = _FNAME_RE.match(fname)
    if m is None or m.group("ok").lower() != "false":
        return None
    return m.group("ts"), int(m.group("ep"))


def _decode_mp4_frames(mp4_path: str) -> list:
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _load_human_correction(mp4_path: str) -> Optional[str]:
    txt_path = os.path.splitext(mp4_path)[0] + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            c = f.read().strip()
        return c or None
    return None


def build_attempt0_from_mp4(mp4_path: str, task_description: str, ep_dir: str,
                             vlmi, run_utils, eval_mode: str,
                             correction_data: Optional[dict] = None,
                             video_dir: str = ""):
    """
    Build ep_dir/attempt_0/ from a failure .mp4 on-the-fly.
    - ours mode with correction: skip GPT, save images + correction only (cheap)
    - ours without correction: run full GPT analysis
    """
    from PIL import Image as PILImage

    attempt_dir = os.path.join(ep_dir, "attempt_0")
    if os.path.exists(attempt_dir):
        return  # already built

    frames = _decode_mp4_frames(mp4_path)
    if len(frames) < 2:
        print(f"  [warn] too few frames in {os.path.basename(mp4_path)}, skipping attempt_0 build")
        return

    base_correction = correction_data.get("base") if correction_data else None
    human_correction = base_correction or _load_human_correction(mp4_path)
    os.makedirs(attempt_dir, exist_ok=True)

    correction_params = None
    if eval_mode == "ours" and human_correction:
        ctx = build_ours_context(mp4_path, task_description, human_correction)
        initial_img = ctx["initial_img"]
        final_img = ctx["final_img"]
        whathappened = ctx["whathappened"]
        reasoning = ctx["reasoning"]
        assessment = ctx["assessment"]
        # Pre-computed params from corrections/ take priority over on-the-fly extraction
        correction_params = (correction_data or {}).get("correction_params") or ctx.get("correction_params")
    else:
        initial_img = PILImage.fromarray(frames[0])
        final_img = PILImage.fromarray(frames[-1])
        # Full GPT analysis
        print(f"  Building attempt_0 context (GPT) for {os.path.basename(mp4_path)}...")
        whathappened = vlmi.critique_vla_video_failure(frames, task_description)
        reasoning = vlmi.reason_about_vla_failure(initial_img, task_description, whathappened)
        assessment = vlmi.assess_hl_failure(
            initial_img, final_img, task_description,
            [(task_description, False, whathappened, reasoning)]
        )

    subtask_dir = os.path.join(attempt_dir, "subtask_0")
    run_utils.save_reasoning_ica_dir(
        subtask_dir, initial_img, task_description,
        success=False, whathappened=whathappened, reasoning=reasoning
    )
    run_utils.save_top_level_ica_dir(
        attempt_dir, initial_img, final_img,
        task=task_description, success=False, assessment=assessment,
        human_correction=human_correction, correction_params=correction_params,
    )
    # Copy original failure video into video_dir as attempt_0 (if not already there)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        dest_name = os.path.basename(mp4_path).replace("--success=False--", "--a0--success=False--")
        dest_path = os.path.join(video_dir, dest_name)
        if not os.path.exists(dest_path):
            shutil.copy2(mp4_path, dest_path)
    print(f"  Built attempt_0 → {attempt_dir}")


def episode_to_task(episode: int, num_trials_per_task: int = 50):
    """Map 1-indexed global episode number → (task_id, episode_idx)."""
    idx = episode - 1
    return idx // num_trials_per_task, idx % num_trials_per_task


def _load_openai_key():
    """Load OPENAI_API_KEY from correctvla/.env into environment."""
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    env_path = os.path.abspath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["OPENAI_API_KEY"] = key
                    return key
    return os.environ.get("OPENAI_API_KEY", "")


@dataclass
class FailureEvalConfig:
    # fmt: off
    pretrained_checkpoint: Union[str, Path] = ""
    model_family: str = "pi05"

    # Eval method: "vanilla" | "ours" (human correction → action bias) | "llm_baseline" (VLM-generated correction)
    eval: str = "vanilla"

    # For llm_baseline: VLM refines its own correction on attempt 2+ (a18-style)
    use_vlm_refinement: bool = True

    # Single-episode mode
    task_suite_name: str = ""
    episode: int = -1                   # 1-indexed global episode from original run

    # Batch mode: scan failure_dir and replay all failures
    failure_dir: str = _VLM_CFG["failure_dir"]
    task_ids: str = ""          # comma-separated task_ids to filter, e.g. "0,1,3,5,6" (empty = all)
    max_per_task: int = 0       # max failures to replay per task_id (0 = all)

    # Human correction files
    corrections_dir: str = _VLM_CFG["corrections_dir"]
    corrections_file: str = ""  # path to human_corrections_*.txt (per-episode corrections)
    timed_feedback: str = ""    # direct timed_feedback for single-episode mode

    # ICA context dir (global, persistent across runs)
    context_dir: str = _VLM_CFG["context_dir"]
    max_attempts: int = 1

    # Multi-trial experiment: run N independent recovery trials per episode (for statistical analysis)
    num_recovery_trials: int = 1

    # Interactive mode: load model once, accept commands from stdin
    interactive: bool = False

    # Model server mode: skip local model load, forward get_action to model_server.py
    use_model_server: bool = False

    # If True, run_N loads all attempts from run_1..run_{N-1} as prior ICA context (default: fresh start)
    use_prev_run: bool = False

    # eval settings (must match original run, defaults from config.yaml)
    num_trials_per_task: int = _VLM_CFG["num_trials_per_task"]
    num_steps_wait: int = _EVAL_CFG["num_steps_wait"]
    env_img_res: int = _EVAL_CFG["env_img_res"]
    seed: int = _EVAL_CFG["seed"]

    # openvla-specific (unused for pi05 but kept for compatibility)
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # fmt: on


# ---------------------------------------------------------------------------
# Vanilla replay (direct pi05 inference, same as original eval)
# ---------------------------------------------------------------------------

def run_vanilla_episode(cfg, env, task_description, model, resize_size, processor,
                        action_head, proprio_projector, noisy_action_projector,
                        initial_state):
    t0 = time.time()
    env.reset()
    obs = env.set_init_state(initial_state)

    if cfg.model_family == "openvla":
        open_loop_steps = cfg.num_open_loop_steps
    else:
        from policies.pi05.pi05_utils import N_ACTION_STEPS
        open_loop_steps = N_ACTION_STEPS

    action_queue = deque()
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            replay_images.append(img)

            if len(action_queue) == 0:
                observation = {
                    "full_image": _resize_image(img, resize_size, cfg.model_family),
                    "wrist_image": _resize_image(wrist_img, resize_size, cfg.model_family),
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )),
                }
                actions = _robot_utils.get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        import traceback
        print(f"Episode error: {e}\n{traceback.format_exc()}")

    latency = time.time() - t0
    return success, replay_images, latency


# ---------------------------------------------------------------------------
# Ours episode: human correction → action bias + VLM-guided refinement
# ---------------------------------------------------------------------------

def _next_run_dir(ep_dir: str) -> tuple:
    """
    Returns (run_dir, run_num) for the next run_N/ subdir under ep_dir.
    Scans existing run_1/, run_2/, ... and increments.
    """
    n = 1
    while os.path.exists(os.path.join(ep_dir, f"run_{n}")):
        n += 1
    return os.path.join(ep_dir, f"run_{n}"), n


def _load_attempts(ep_dir: str, TaskICADir) -> list:
    """Load all attempt_K/ subdirs from ep_dir in order."""
    attempts = []
    for entry in sorted(os.listdir(ep_dir)):
        if not entry.startswith("attempt_"):
            continue
        full = os.path.join(ep_dir, entry)
        if os.path.isdir(full):
            attempts.append(TaskICADir(full))
    return attempts


def _setup_vlm(cfg):
    """Load OPENAI_API_KEY and import VLM modules from policies.vlm package."""
    api_key = _load_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in correctvla/.env")

    import policies.vlm.vlm_hl.vlm_methods as vlmi
    from policies.vlm.vlm_hl.vlm_methods import LLMStats
    from policies.vlm.ica.reasoning_ica import TaskICADir
    import policies.vlm.run_utils as run_utils

    return vlmi, LLMStats, TaskICADir, run_utils


def _load_prev_attempt_params(attempt_dir: str) -> dict:
    """Load correction_params.json from a previous attempt directory."""
    path = os.path.join(attempt_dir, "correction_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_correction_feedback(attempt_dir: str) -> list:
    """Load eef displacement feedback saved by world.act() for the previous attempt."""
    path = os.path.join(attempt_dir, "correction_feedback.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _load_correction_reasoning(attempt_dir: str) -> str:
    """Load VLM reasoning text saved from the previous a18 refinement."""
    path = os.path.join(attempt_dir, "correction_reasoning.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def _generate_correction_from_video(
    failure_mp4: str, task_description: str, vlmi, save_dir: str = None,
) -> dict:
    """VLM watches failure video → generates correction params from scratch (no human input)."""
    try:
        failure_frames = _decode_mp4_frames(failure_mp4)
        refined_text = vlmi.generate_correction_from_failure(failure_frames, task_description)
        print(f"  [llm] VLM generated correction: {refined_text}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "vlm_correction.txt"), "w") as f:
                f.write(refined_text)
            full_response = vlmi.get_last_refine_full_response("llm")
            with open(os.path.join(save_dir, "correction_reasoning.txt"), "w") as f:
                f.write(full_response or refined_text)
        from policies.vlm.vlm_hl.vlm_methods import extract_correction_params_multi
        return extract_correction_params_multi(refined_text, task_description)
    except Exception as e:
        print(f"  [llm] _generate_correction_from_video failed: {e}")
        return None


def _refine_params_from_video(
    failure_mp4: str, attempt_mp4: str, task_description: str,
    correction_text: str, fallback_params: dict, vlmi,
    correction_feedback: list = None, previous_reasoning: str = None,
    save_reasoning_dir: str = None,
) -> dict:
    """VLM watches failure + attempt video → refined correction text → params."""
    try:
        failure_frames = _decode_mp4_frames(failure_mp4)
        attempt_frames = _decode_mp4_frames(attempt_mp4)
        kwargs = {}
        if correction_feedback:
            kwargs["correction_feedback"] = correction_feedback
        if previous_reasoning:
            kwargs["previous_reasoning"] = previous_reasoning
        refined_text = vlmi.refine_human_correction_a18(
            failure_frames, attempt_frames, task_description, correction_text, **kwargs)
        print(f"  [refine] refined correction text: {refined_text}")
        if save_reasoning_dir:
            os.makedirs(save_reasoning_dir, exist_ok=True)
            reasoning_path = os.path.join(save_reasoning_dir, "correction_reasoning.txt")
            full_response = vlmi.get_last_refine_full_response("a18")
            with open(reasoning_path, "w") as f:
                f.write(full_response or refined_text)
        from policies.vlm.vlm_hl.vlm_methods import extract_correction_params_multi
        return extract_correction_params_multi(refined_text, task_description)
    except Exception as e:
        print(f"  [refine] _refine_params_from_video failed: {e}, falling back")
        return fallback_params


def run_episode(cfg, env, task_description, model, resize_size, processor,
                action_head, proprio_projector, noisy_action_projector,
                initial_state, context_dir, vlmi, run_utils,
                forced_correction_params=None):
    """
    Run one episode with direct action forcing.
    Single world.act() call — bias injection handles timing.
    """
    from policies.vlm.world import LiberoWorldStub
    import functools

    t0 = time.time()
    world = LiberoWorldStub(
        env=env, initial_state=initial_state, task_description=task_description,
        model=model, cfg=cfg, resize_size=resize_size, processor=processor,
        action_head=action_head, proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
    )
    world.physical_reset()

    # Inject correction_params into every world.act() call
    if forced_correction_params is not None:
        _orig_act = world.act
        @functools.wraps(_orig_act)
        def _biased_act(command, **kwargs):
            kwargs.setdefault("correction_params", forced_correction_params)
            return _orig_act(command, **kwargs)
        world.act = _biased_act
        print(f"  [bias] correction_params: {forced_correction_params}")

    # Execute single act() call — no subtask decomposition
    print(f"[{cfg.eval.upper()}] world.act({task_description!r})")
    world.act(task_description)

    # VLM verification: override simulator false positives
    success = world.episode_success
    if success:
        # Grab initial and final frames from the subtask recording
        frames = world.subtask_frame_tuples[0][1] if world.subtask_frame_tuples else []
        if len(frames) >= 2:
            from PIL import Image as PILImage
            initial_img = PILImage.fromarray(frames[0])
            final_img = PILImage.fromarray(frames[-1])
            vlm_success = vlmi.determine_vla_success(initial_img, final_img, task_description)
            if not vlm_success:
                print(f"  [verify] Simulator done=True but VLM says FAILURE — overriding to False")
                success = False

    os.makedirs(context_dir, exist_ok=True)

    # Save eef displacement feedback for potential refinement
    feedback = getattr(world, "_last_correction_feedback", None)
    if feedback:
        with open(os.path.join(context_dir, "correction_feedback.json"), "w") as _f:
            json.dump(feedback, _f, indent=2)

    replay_images = []
    for _, frames in world.subtask_frame_tuples:
        replay_images.extend(frames)

    latency = time.time() - t0
    return success, replay_images, latency


# ---------------------------------------------------------------------------
# Shared replay_one dispatcher
# ---------------------------------------------------------------------------

def replay_one(cfg, suite_name: str, episode: int, model, resize_size, processor,
               action_head, proprio_projector, noisy_action_projector,
               vlm_state: Optional[dict] = None, fname: str = "", video_dir: str = "",
               run_dir: str = ""):
    cfg.task_suite_name = suite_name

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    task_id, episode_idx = episode_to_task(episode, cfg.num_trials_per_task)
    assert task_id < task_suite.n_tasks, f"task_id {task_id} out of range for {suite_name}"
    assert episode_idx < cfg.num_trials_per_task, f"episode_idx {episode_idx} out of range"

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    print(f"\n[Failure replay] eval={cfg.eval} | suite={suite_name} | task_id={task_id} | episode_idx={episode_idx} | episode={episode}")
    print(f"  Task: {task_description}")

    initial_state = initial_states[episode_idx]

    # Per-episode seed: matches vla_eval.py so vanilla results are reproducible
    set_seed_everywhere(cfg.seed + episode - 1)

    trial_results = []
    model_tag = f"{cfg.model_family}_{cfg.eval}" if cfg.eval != "vanilla" else cfg.model_family

    for trial in range(cfg.num_recovery_trials):
        trial_tag = f"t{trial+1}" if cfg.num_recovery_trials > 1 else ""
        if cfg.num_recovery_trials > 1:
            print(f"  [Trial {trial+1}/{cfg.num_recovery_trials}]")

        if cfg.eval in ("ours", "llm_baseline"):
            assert vlm_state is not None

            # ep_dir: context/ep{N}/
            ep_dir = vlm_state["source_to_epdir"].get(fname)
            if ep_dir is None:
                ep_dir = os.path.join(vlm_state["context_dir"], f"ep{episode}")
                os.makedirs(ep_dir, exist_ok=True)
                if fname:
                    with open(os.path.join(ep_dir, "source.txt"), "w") as f:
                        f.write(fname)

            # Correction lookup: CLI timed_feedback > per-episode file > per-task_id folder
            correction_data = None
            if cfg.eval == "ours":
                if cfg.timed_feedback:
                    correction_data = {
                        "base": cfg.timed_feedback,
                        "correction_params": parse_timed_feedback(cfg.timed_feedback),
                    }
                elif vlm_state.get("ep_corrections", {}).get((suite_name, episode)):
                    correction_data = vlm_state["ep_corrections"][(suite_name, episode)]
                else:
                    correction_data = vlm_state["task_corrections"].get(task_id)

            # Build attempt_0 on-the-fly from failure .mp4
            attempt0_dir = os.path.join(ep_dir, "attempt_0")
            if not os.path.exists(attempt0_dir) and fname:
                mp4_path = os.path.join(cfg.failure_dir, fname)
                if os.path.exists(mp4_path):
                    build_attempt0_from_mp4(
                        mp4_path, task_description, ep_dir,
                        vlm_state["vlmi"], vlm_state["run_utils"],
                        eval_mode=cfg.eval,
                        correction_data=correction_data,
                        video_dir=video_dir,
                    )

            active_ep_dir, run_num = _next_run_dir(ep_dir)
            os.makedirs(active_ep_dir, exist_ok=True)

            run_info = {
                "results_dir": run_dir, "episode": episode, "run": run_num, "task_id": task_id,
                "correction_version": correction_data.get("version") if correction_data else None,
                "correction_dir": correction_data.get("path") if correction_data else None,
            }
            with open(os.path.join(active_ep_dir, "run_info.json"), "w") as _f:
                json.dump(run_info, _f, indent=2)

            attempt0_src = os.path.join(ep_dir, "attempt_0")
            attempt0_dst = os.path.join(active_ep_dir, "attempt_0")
            if os.path.exists(attempt0_src) and not os.path.exists(attempt0_dst):
                shutil.copytree(attempt0_src, attempt0_dst)

            task_icadirs = _load_attempts(active_ep_dir, vlm_state["TaskICADir"])
            if cfg.use_prev_run and run_num > 1:
                for prev_n in range(1, run_num):
                    prev_dir = os.path.join(ep_dir, f"run_{prev_n}")
                    task_icadirs += _load_attempts(prev_dir, vlm_state["TaskICADir"])

            eval_tag = cfg.eval.upper()
            print(f"[{eval_tag}] ep{episode} run_{run_num}: {len(task_icadirs)} attempt(s) in context"
                  + (f" (including prev runs)" if cfg.use_prev_run and run_num > 1 else ""))

            total_latency = 0.0
            precomputed_params = (correction_data or {}).get("correction_params")

            # For ours: overwrite ICA dirs with precomputed params
            if precomputed_params and cfg.eval == "ours":
                for icadir in task_icadirs:
                    cp_path = os.path.join(icadir.dir_path, "correction_params.json")
                    with open(cp_path, "w") as _f:
                        json.dump(precomputed_params, _f, indent=2)
                    icadir.correction_params_path = cp_path

            success = False
            for attempt in range(cfg.max_attempts):
                print(f"[{eval_tag}] ep{episode} attempt {attempt+1}/{cfg.max_attempts}")
                attempt_dir = os.path.join(active_ep_dir, f"attempt_{len(task_icadirs)}")

                # Determine correction params for this attempt
                if cfg.eval == "ours":
                    # Human correction: fixed precomputed params, all attempts
                    forced_params = precomputed_params
                else:
                    # llm_baseline: VLM generates correction from failure video
                    failure_mp4 = os.path.join(cfg.failure_dir, fname) if fname else ""
                    if not failure_mp4 and cfg.failure_dir:
                        same_task_mp4 = ""
                        for _fn in sorted(os.listdir(cfg.failure_dir)):
                            _m = _FNAME_RE.match(_fn)
                            if not _m or _m.group("ok").lower() != "false":
                                continue
                            if "pi05" not in _m.group("model"):
                                continue
                            _ep = int(_m.group("ep"))
                            if _ep == episode:
                                failure_mp4 = os.path.join(cfg.failure_dir, _fn)
                                break
                            if not same_task_mp4:
                                _ep_task, _ = episode_to_task(_ep, cfg.num_trials_per_task)
                                if _ep_task == task_id:
                                    same_task_mp4 = os.path.join(cfg.failure_dir, _fn)
                        if not failure_mp4 and same_task_mp4:
                            failure_mp4 = same_task_mp4
                            print(f"  [llm] no failure for ep={episode}, using same-task fallback: {os.path.basename(same_task_mp4)}")
                    if attempt == 0:
                        # Attempt 1: VLM watches failure video → generates params
                        forced_params = _generate_correction_from_video(
                            failure_mp4, task_description, vlm_state["vlmi"],
                            save_dir=attempt_dir) if failure_mp4 else None
                        if forced_params:
                            print(f"  [llm] attempt 1: VLM generated {forced_params}")
                        else:
                            print(f"  [llm] attempt 1: VLM generation failed, no correction")
                    elif cfg.use_vlm_refinement and task_icadirs:
                        # Attempt 2+: refine using failure+attempt video comparison
                        vlm_correction_text = ""
                        for icadir in task_icadirs:
                            _vc = os.path.join(icadir.dir_path, "vlm_correction.txt")
                            if os.path.exists(_vc):
                                with open(_vc) as _f:
                                    vlm_correction_text = _f.read().strip()
                                break
                        attempt_mp4 = ""
                        _r = os.path.join(task_icadirs[-1].dir_path, "result.json")
                        if os.path.exists(_r):
                            with open(_r) as _f:
                                attempt_mp4 = json.load(_f).get("video", "")
                        if not attempt_mp4:
                            attempt_mp4 = failure_mp4
                        prev_feedback = _load_correction_feedback(task_icadirs[-1].dir_path)
                        prev_reasoning = _load_correction_reasoning(task_icadirs[-1].dir_path)
                        if failure_mp4 and attempt_mp4 and vlm_correction_text:
                            prev_params = _load_prev_attempt_params(task_icadirs[-1].dir_path)
                            forced_params = _refine_params_from_video(
                                failure_mp4, attempt_mp4, task_description,
                                vlm_correction_text, prev_params, vlm_state["vlmi"],
                                correction_feedback=prev_feedback,
                                previous_reasoning=prev_reasoning,
                                save_reasoning_dir=attempt_dir,
                            )
                            print(f"  [llm] attempt {attempt+1}: VLM refined {forced_params}")
                        else:
                            print(f"  [llm] missing data for refinement, reusing previous params")
                            forced_params = _load_prev_attempt_params(task_icadirs[-1].dir_path)
                    else:
                        # No refinement: reuse previous attempt's params
                        forced_params = _load_prev_attempt_params(task_icadirs[-1].dir_path) if task_icadirs else None

                success, replay_images, latency = run_episode(
                    cfg, env, task_description, model, resize_size,
                    processor, action_head, proprio_projector, noisy_action_projector,
                    initial_state, context_dir=attempt_dir,
                    vlmi=vlm_state["vlmi"], run_utils=vlm_state["run_utils"],
                    forced_correction_params=forced_params,
                )
                total_latency += latency

                if forced_params:
                    os.makedirs(attempt_dir, exist_ok=True)
                    with open(os.path.join(attempt_dir, "correction_params.json"), "w") as _f:
                        json.dump(forced_params, _f, indent=2)

                task_icadirs.append(vlm_state["TaskICADir"](attempt_dir))
                print(f"  Attempt {attempt+1}: {'SUCCESS' if success else 'FAILURE'} | latency: {latency:.1f}s")
                video_tag = f"{model_tag}_a{attempt+1}{trial_tag}"
                video_path = save_rollout_video(replay_images, episode, success=success,
                                                task_description=task_description, model_family=video_tag,
                                                rollout_dir=video_dir or None)
                with open(os.path.join(attempt_dir, "result.json"), "w") as _f:
                    json.dump({"attempt": attempt + 1, "success": success,
                               "latency_s": round(latency, 2), "video": video_path}, _f, indent=2)
                if success:
                    break

            trial_results.append({"trial": trial + 1, "success": success, "latency_s": round(total_latency, 2)})

        else:
            success, replay_images, latency = run_vanilla_episode(
                cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                initial_state,
            )
            print(f"  Result: {'SUCCESS' if success else 'FAILURE'} | latency: {latency:.1f}s")
            video_tag = f"{model_tag}{trial_tag}"
            save_rollout_video(replay_images, episode, success=success,
                               task_description=task_description, model_family=video_tag,
                               rollout_dir=video_dir or None)
            trial_results.append({"trial": trial + 1, "success": success, "latency_s": round(latency, 2)})

    executed = [r for r in trial_results if r.get("success") is not None]
    any_success = any(r["success"] for r in executed) if executed else None
    total_latency = sum(r.get("latency_s", 0) for r in executed)
    return trial_results, any_success, total_latency


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_failures(cfg: FailureEvalConfig):
    assert cfg.eval in ("vanilla", "ours", "llm_baseline"), f"--eval must be 'vanilla', 'ours', or 'llm_baseline', got: {cfg.eval}"

    set_seed_everywhere(cfg.seed)

    if cfg.use_model_server:
        from experiments.robot.model_client import get_action_remote
        _robot_utils.get_action = get_action_remote
        model = action_head = proprio_projector = noisy_action_projector = processor = None
        resize_size = get_image_resize_size(cfg)
        print("[model_server] Using remote model server — no local model load.")
    else:
        assert cfg.pretrained_checkpoint, "pretrained_checkpoint required (or use --use_model_server)"
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)

    # Set up VLM state (shared across episodes)
    vlm_state = None
    if cfg.eval in ("ours", "llm_baseline"):
        vlmi, LLMStats, TaskICADir, run_utils = _setup_vlm(cfg)
        task_corrections = _load_task_corrections(
            os.path.join(cfg.corrections_dir, cfg.model_family),
            suite=cfg.task_suite_name,
        ) if cfg.eval == "ours" else {}
        # Per-episode corrections: dirs > txt file
        ep_corrections = {}
        if cfg.eval == "ours":
            ep_corrections = _load_ep_corrections_dirs(cfg.corrections_dir, cfg.model_family)
            if cfg.corrections_file:
                ep_corrections.update(_load_ep_corrections(cfg.corrections_file))
        os.makedirs(cfg.context_dir, exist_ok=True)

        source_to_epdir = {}
        for subdir in sorted(os.listdir(cfg.context_dir)):
            subdir_path = os.path.join(cfg.context_dir, subdir)
            src_file = os.path.join(subdir_path, "source.txt")
            if os.path.isdir(subdir_path) and os.path.exists(src_file):
                with open(src_file) as f:
                    source_to_epdir[f.read().strip()] = subdir_path

        vlm_state = {
            "vlmi": vlmi, "LLMStats": LLMStats, "TaskICADir": TaskICADir,
            "run_utils": run_utils, "source_to_epdir": source_to_epdir,
            "context_dir": cfg.context_dir, "task_corrections": task_corrections,
            "ep_corrections": ep_corrections,
        }
        print(f"[{cfg.eval.upper()}] context_dir: {cfg.context_dir} | task_corrections: {len(task_corrections)} | ep_corrections: {len(ep_corrections)}")

    # --- Interactive mode ---
    if cfg.interactive:
        _run_interactive(cfg, model, resize_size, processor, action_head,
                         proprio_projector, noisy_action_projector, vlm_state)
        return

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    eval_tag = f"{cfg.model_family}_{cfg.eval}"

    # --- Single-episode mode ---
    if cfg.episode > 0:
        assert cfg.task_suite_name, "--task_suite_name required with --episode"
        task_id, _ = episode_to_task(cfg.episode, cfg.num_trials_per_task)
        run_dir = os.path.join("results", eval_tag, cfg.task_suite_name,
                               f"task_{task_id}", f"ep_{cfg.episode}", run_ts)
        video_dir = os.path.join(run_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Results dir: {run_dir}")
        trial_results, any_success, total_latency = replay_one(
            cfg, cfg.task_suite_name, cfg.episode, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            vlm_state=vlm_state, fname="", video_dir=video_dir, run_dir=run_dir)
        record = {
            "eval": cfg.eval,
            "model": cfg.model_family,
            "suite": cfg.task_suite_name,
            "episode": cfg.episode,
            "any_success": any_success,
            "recovery_rate": sum(r["success"] for r in trial_results if r.get("success") is not None) / max(1, sum(1 for r in trial_results if r.get("success") is not None)),
            "trials": trial_results,
            "total_latency_s": round(total_latency, 2),
        }
        out_path = os.path.join(run_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nResults saved → {out_path}")
        return

    # --- Batch mode: scan failure_dir ---
    assert cfg.failure_dir, "Provide --episode + --task_suite_name, or --failure_dir"
    allowed_task_ids = None
    if cfg.task_ids:
        allowed_task_ids = {int(x.strip()) for x in cfg.task_ids.split(",")}
        print(f"Filtering to task_ids: {sorted(allowed_task_ids)}")

    failures = []
    task_id_counts: dict = {}
    for fname in sorted(os.listdir(cfg.failure_dir)):
        parsed = parse_failure_filename(fname)
        if parsed is None:
            continue
        ts, ep = parsed
        suite = TIMESTAMP_TO_SUITE.get(ts)
        if suite is None:
            print(f"Warning: unknown timestamp {ts} in {fname}, skipping")
            continue
        if cfg.task_suite_name and suite != cfg.task_suite_name:
            continue
        task_id, _ = episode_to_task(ep, cfg.num_trials_per_task)
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        if cfg.max_per_task > 0:
            if task_id_counts.get(task_id, 0) >= cfg.max_per_task:
                continue
            task_id_counts[task_id] = task_id_counts.get(task_id, 0) + 1
        failures.append((suite, ep, fname))

    batch_dir = os.path.join("results", eval_tag, f"batch_{run_ts}")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"Found {len(failures)} failure(s) to replay")
    print(f"Batch results dir: {batch_dir}")
    records = []
    for suite, ep, fname in failures:
        print(f"\nReplaying: {fname}")
        task_id, _ = episode_to_task(ep, cfg.num_trials_per_task)
        ep_run_dir = os.path.join("results", eval_tag, suite,
                                  f"task_{task_id}", f"ep_{ep}", run_ts)
        ep_video_dir = os.path.join(ep_run_dir, "videos")
        os.makedirs(ep_video_dir, exist_ok=True)
        trial_results, any_success, total_latency = replay_one(
            cfg, suite, ep, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            vlm_state=vlm_state, fname=fname, video_dir=ep_video_dir, run_dir=ep_run_dir)
        executed_trials = [r for r in trial_results if r.get("success") is not None]
        recovery_rate = sum(r["success"] for r in executed_trials) / len(executed_trials) if executed_trials else 0
        record = {"file": fname, "suite": suite, "episode": ep,
                  "task_id": task_id, "run_dir": ep_run_dir,
                  "any_success": any_success,
                  "recovery_rate": recovery_rate,
                  "trials": trial_results, "total_latency_s": round(total_latency, 2)}
        records.append(record)
        # Per-episode results
        with open(os.path.join(ep_run_dir, "results.json"), "w") as _f:
            json.dump(record, _f, indent=2)
        # Incremental batch summary so results survive crashes
        with open(os.path.join(batch_dir, "results.json"), "w") as _f:
            json.dump({"eval": cfg.eval, "model": cfg.model_family,
                       "context_dir": cfg.context_dir, "episodes": records}, _f, indent=2)

    recovered = sum(1 for r in records if r["any_success"])
    avg_recovery_rate = sum(r["recovery_rate"] for r in records) / len(records) if records else 0
    avg_latency = sum(r["total_latency_s"] for r in records) / len(records) if records else 0

    print("\n=== Failure Replay Summary ===")
    for r in records:
        executed_trials = [t for t in r["trials"] if t.get("success") is not None]
        n_success = sum(t["success"] for t in executed_trials)
        n_trials = len(executed_trials)
        print(f"  {'✓' if r['any_success'] else '✗'} ep={r['episode']} {r['suite']} "
              f"({n_success}/{n_trials} trials, {r['total_latency_s']:.1f}s)")
    print(f"\nAny-success rate: {recovered}/{len(records)} ({100*recovered/len(records):.1f}%)" if records else "")
    print(f"Avg recovery rate: {avg_recovery_rate:.2%}")
    print(f"Avg latency:       {avg_latency:.1f}s")

    summary = {
        "eval": cfg.eval,
        "model": cfg.model_family,
        "context_dir": cfg.context_dir,
        "num_recovery_trials": cfg.num_recovery_trials,
        "total_episodes": len(records),
        "any_success_count": recovered,
        "any_success_rate": round(recovered / len(records), 4) if records else 0,
        "avg_recovery_rate": round(avg_recovery_rate, 4),
        "avg_latency_s": round(avg_latency, 2),
        "episodes": records,
    }
    out_path = os.path.join(batch_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Interactive REPL (--interactive): load model once, accept commands from stdin
# ---------------------------------------------------------------------------

def _run_interactive(cfg, model, resize_size, processor, action_head,
                     proprio_projector, noisy_action_projector, vlm_state):
    """
    Interactive loop — model stays in GPU memory between commands.

    Commands:
      episode <N> [suite] [eval]   e.g.  episode 901 libero_90 ours
      quit / exit
    """
    interactive_ts = time.strftime("%Y%m%d_%H%M%S")
    eval_tag = f"{cfg.model_family}_{cfg.eval}"

    print(f"\n[interactive] Model loaded. Results → results/{eval_tag}/...")
    print("[interactive] Commands:  episode <N> [suite] [eval]  |  quit")
    print(f"[interactive] Defaults:  suite={cfg.task_suite_name or 'libero_90'}  eval={cfg.eval}\n")

    records = []
    while True:
        try:
            line = input("[interactive]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break

        parts = line.split()
        if parts[0] != "episode" or len(parts) < 2:
            print("  Usage: episode <N> [suite] [eval]")
            continue

        ep = int(parts[1])
        suite = parts[2] if len(parts) > 2 else (cfg.task_suite_name or "libero_90")
        eval_mode = parts[3] if len(parts) > 3 else cfg.eval

        # Temporarily override eval mode
        orig_eval = cfg.eval
        cfg.eval = eval_mode
        if vlm_state:
            vlm_state["task_corrections"] = _load_task_corrections(cfg.corrections_dir, suite=cfg.task_suite_name)

        task_id, _ = episode_to_task(ep, cfg.num_trials_per_task)
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        ep_run_dir = os.path.join("results", eval_tag, suite,
                                  f"task_{task_id}", f"ep_{ep}", run_ts)
        ep_video_dir = os.path.join(ep_run_dir, "videos")
        os.makedirs(ep_video_dir, exist_ok=True)

        print(f"  Running ep={ep} suite={suite} eval={eval_mode}")
        print(f"  Results → {ep_run_dir}")
        try:
            trial_results, any_success, latency = replay_one(
                cfg, suite, ep, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                vlm_state=vlm_state, fname="", video_dir=ep_video_dir, run_dir=ep_run_dir)
            print(f"  Result: {'SUCCESS' if any_success else 'FAILURE'} | {latency:.1f}s")
            records.append({"episode": ep, "suite": suite, "eval": eval_mode,
                            "any_success": any_success, "trials": trial_results,
                            "run_dir": ep_run_dir})
            with open(os.path.join(ep_run_dir, "results.json"), "w") as f:
                json.dump(records[-1], f, indent=2)
        except Exception as e:
            import traceback
            print(f"  Error: {e}\n{traceback.format_exc()}")
        finally:
            cfg.eval = orig_eval

    print(f"[interactive] Done. {len(records)} episode(s) completed.")


if __name__ == "__main__":
    eval_failures()
