"""
vla_eval.py

Evaluates a trained VLA policy (openvla or pi05) in LIBERO simulation benchmark tasks.
"""

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm

# Patch torch.load for PyTorch >=2.6 compatibility (LIBERO uses torch.load without weights_only=False)
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark

import wandb

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    # openvla-specific
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

    # environment
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    # logging
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7
    # fmt: on


def _import_openvla_utils():
    from policies.openvla.openvla_utils import (
        get_action_head,
        get_noisy_action_projector,
        get_processor,
        get_proprio_projector,
        resize_image_for_policy,
    )
    return get_action_head, get_noisy_action_projector, get_processor, get_proprio_projector, resize_image_for_policy


def _resize_image(img, resize_size, model_family):
    if model_family == "openvla":
        _, _, _, _, resize_image_for_policy = _import_openvla_utils()
        return resize_image_for_policy(img, resize_size)
    else:
        from PIL import Image as _PIL
        return np.array(_PIL.fromarray(img).resize((resize_size, resize_size)))


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    action_head = proprio_projector = noisy_action_projector = processor = None

    if cfg.model_family == "openvla":
        get_action_head, get_noisy_action_projector, get_processor, get_proprio_projector, _ = _import_openvla_utils()
        global _NUM_ACTIONS_CHUNK
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK as _NUM_ACTIONS_CHUNK
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
        action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
        processor = get_processor(cfg)
        # validate unnorm key
        unnorm_key = cfg.unnorm_key if cfg.unnorm_key else cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key not in model.norm_stats:
            available_keys = list(model.norm_stats.keys())
            fallback_key = available_keys[0]
            print(f"[OOD] '{cfg.task_suite_name}' not in norm_stats, using '{fallback_key}' (available: {available_keys})")
            unnorm_key = fallback_key
        cfg.unnorm_key = unnorm_key

    return model, action_head, proprio_projector, noisy_action_projector, processor


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(msg, log_file=None):
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def process_action(action, model_family):
    if model_family == "openvla":
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
    return action


def run_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector, initial_state, log_file):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    # For openvla, chunk size comes from prismatic constants; for pi05, from N_ACTION_STEPS
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
                actions = get_action(
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
        log_message(f"Episode error: {e}\n{traceback.format_exc()}", log_file)

    return success, replay_images


def run_task(cfg, task_suite, task_id, model, resize_size, processor, action_head, proprio_projector, noisy_action_projector, total_episodes, total_successes, log_file):
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    all_initial_states = None
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path) as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)

    task_episodes = task_successes = 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if all_initial_states is not None:
            key = task_description.replace(" ", "_")
            ep_key = f"demo_{episode_idx}"
            if not all_initial_states[key][ep_key]["success"]:
                log_message(f"Skipping episode {episode_idx} (failed expert demo)", log_file)
                continue
            initial_state = np.array(all_initial_states[key][ep_key]["initial_state"])
        else:
            initial_state = initial_states[episode_idx]

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            initial_state, log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, model_family=cfg.model_family)
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes: {total_episodes} | # successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_sr = task_successes / task_episodes if task_episodes > 0 else 0
    total_sr = total_successes / total_episodes if total_episodes > 0 else 0
    log_message(f"Task success rate: {task_sr:.4f}", log_file)
    log_message(f"Total success rate: {total_sr:.4f}", log_file)

    if cfg.use_wandb:
        wandb.log({f"success_rate/{task_description}": task_sr, f"num_episodes/{task_description}": task_episodes})

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert cfg.task_suite_name in [s.value for s in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    log_message(f"Task suite: {cfg.task_suite_name} | Model: {cfg.model_family}", log_file)

    total_episodes = total_successes = 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        total_episodes, total_successes = run_task(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            total_episodes, total_successes, log_file,
        )

    final_sr = total_successes / total_episodes if total_episodes > 0 else 0
    log_message(f"Final: {total_successes}/{total_episodes} = {final_sr * 100:.1f}%", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_sr, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)

    log_file.close()
    return final_sr


if __name__ == "__main__":
    eval_libero()
