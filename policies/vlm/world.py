"""
world.py

LiberoWorldStub: world interface for LIBERO simulation + pi05.

Interface:
  world.act(command: str)       — run VLA for one subtask, collect frames
  world.current_image           — PIL Image of latest observation
  world.physical_reset()        — reset env to initial state
  world.subtask_frame_tuples    — [(cmd, [np.ndarray frames]), ...]
  world.manipulable_object_uids — set by refresh_objects() via VLM

The env.step() done signal is stored in self.episode_success for the caller to read.
"""

import os
import numpy as np
from PIL import Image

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

def _load_openai_key():
    """Load OPENAI_API_KEY from correctvla/.env into environment if not already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    env_path = os.path.abspath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["OPENAI_API_KEY"] = key
                    return


def _obs_to_pil(obs) -> Image.Image:
    from experiments.robot.libero.libero_utils import get_libero_image
    return Image.fromarray(get_libero_image(obs))


class LiberoWorldStub:
    """
    World stub interface for LIBERO simulation with pi05.

    Usage:
        world = LiberoWorldStub(env, initial_state, task_description, model, cfg, resize_size)
        world.physical_reset()           # sets up env, runs wait steps, sets current_image
        exec(vlm_plan, {"world": world})
        success = world.episode_success
    """

    finetuning_tasks = "libero"

    def __init__(self, env, initial_state, task_description, model, cfg, resize_size,
                 processor=None, action_head=None, proprio_projector=None,
                 noisy_action_projector=None):
        _load_openai_key()

        self.env = env
        self.initial_state = initial_state
        self.task_description = task_description
        self.task_instruction = task_description  # BaseWorldStub compat
        self.model = model
        self.cfg = cfg
        self.resize_size = resize_size
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector

        self.subtask_frame_tuples = []
        self.manipulable_object_uids = []
        self.episode_success = False
        self.current_image = None
        self._obs = None
        self._episode_step_count = 0  # cumulative steps across all act() calls (absolute time)
        suite = getattr(cfg, "task_suite_name", "libero_90")
        self.subtask_max_steps = TASK_MAX_STEPS.get(suite, 400)

    def physical_reset(self):
        """Reset LIBERO env to the episode's initial state."""
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        self.env.reset()
        obs = self.env.set_init_state(self.initial_state)
        for _ in range(self.cfg.num_steps_wait):
            obs, _, done, _ = self.env.step(get_libero_dummy_action(self.cfg.model_family))
            if done:
                break
        self._obs = obs
        self.current_image = _obs_to_pil(obs)
        self.subtask_frame_tuples = []
        self.episode_success = False
        self._episode_step_count = 0

    def reset(self, new_task=None, keep_frames=False):
        """BaseWorldStub-compatible reset."""
        if not keep_frames:
            self.subtask_frame_tuples = []
        if new_task is not None:
            self.task_instruction = new_task
            self.task_description = new_task
        self.physical_reset()
        return self.current_image

    def refresh_objects(self, update=None):
        """Call VLM to identify scene objects for plan generation."""
        try:
            from policies.vlm.vlm_hl.vlm_methods import get_object_uids_from_scene
            img = update if update is not None else self.current_image
            if img is not None:
                self.manipulable_object_uids = get_object_uids_from_scene(
                    img, self.task_instruction
                )
        except Exception as e:
            print(f"[vlm] refresh_objects skipped: {e}")
            self.manipulable_object_uids = []

    def act(self, command: str, max_steps: int = None, correction_params: dict = None):
        """
        Execute one VLA subtask command.
        Loops up to max_steps env steps (same pattern as vla_eval.run_episode),
        re-querying the model when the action queue empties.
        Collects observation frames, updates current_image, appends to subtask_frame_tuples.
        Sets episode_success=True if env signals done.

        correction_params: optional CorrectionParams dict to apply as additive action bias.
            Directly forces the robot's actions within the specified time window without
            relying on the VLM to translate it into natural language.
        """
        from collections import deque
        from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image
        from experiments.robot.libero.vla_eval import _resize_image, process_action
        import experiments.robot.robot_utils as _robot_utils
        from experiments.robot.libero.libero_utils import quat2axisangle

        if self._obs is None:
            raise RuntimeError("Call physical_reset() before act()")

        # If the episode already ended (e.g. task succeeded in a prior subtask), skip gracefully
        if self.episode_success:
            self.subtask_frame_tuples.append((command, []))
            return

        if max_steps is None:
            max_steps = self.subtask_max_steps

        frames = []
        action_queue = deque()
        obs = self._obs

        # Compile envelopes once — evaluated per-step at exact t_now (AdaptVLA style)
        _DIM_IDX = {"x": 0, "y": 1, "z": 2}
        _compiled_envelopes = []  # list of (dim_idx, envelope_fn, t_start, t_end, suppress)
        _window_trackers = {}
        if correction_params:
            from policies.pi05.action_bias import _build_envelope, _DIM_MAP, _MAGNITUDE_TERM_RATIO
            from policies.pi05.pi05_utils import get_action_std
            action_std = get_action_std(self.cfg)
            axes_info = correction_params.get("axes") or [correction_params]
            for ax in axes_info:
                dim = ax.get("dimension", "unknown")
                if dim not in _DIM_MAP:
                    continue
                dim_idx = _DIM_MAP[dim]
                t_start = float(ax["t_start"]) if ax.get("t_start") is not None else None
                t_end = float(ax["t_end"]) if ax.get("t_end") is not None else float("inf")
                envelope_type = ax.get("envelope", "flat")
                direction = ax.get("direction", "+1")
                sign = 1.0 if direction == "+1" else -1.0
                mag_val = abs(float(ax.get("magnitude_value") or 0.0))
                term = ax.get("magnitude_term", "unknown")
                if mag_val != 0.0:
                    magnitude = sign * mag_val
                elif action_std is not None and term in _MAGNITUDE_TERM_RATIO:
                    magnitude = sign * _MAGNITUDE_TERM_RATIO[term] * float(action_std[dim_idx])
                else:
                    magnitude = sign * 1.0
                mode = ax.get("mode", "add")
                suppress = mode in ("suppress", "override")
                _compiled_envelopes.append({
                    "dim_idx": dim_idx, "dim": dim,
                    "t_start": t_start, "t_end": t_end,
                    "envelope_type": envelope_type,
                    "magnitude": magnitude,
                    "suppress": suppress,
                    "add": mode in ("add", "override"),
                })
                if dim in _DIM_IDX and t_start is not None:
                    # Key by (dim, direction) so +Z and -Z get separate trackers
                    tracker_key = f"{dim}_{direction}"
                    _window_trackers[tracker_key] = {
                        "dim": dim, "dim_idx": dim_idx,
                        "t_start": max(0.0, t_start), "t_end": t_end,
                        "eef_start": None, "eef_end": None,
                        "applied_sum": 0.0, "n": 0,
                    }

        def _eval_envelope_at(env_cfg, t):
            """Evaluate envelope scalar at time t."""
            t_start = env_cfg["t_start"]
            t_end = env_cfg["t_end"]
            if t_start is None:
                return 1.0
            if t < t_start or t > t_end:
                return 0.0
            etype = env_cfg["envelope_type"]
            if etype == "ramp":
                duration = max(t_end - t_start, 1e-6)
                return (t - t_start) / duration
            if etype == "triangle":
                duration = max(t_end - t_start, 1e-6)
                t_mid = (t_start + t_end) / 2.0
                if t <= t_mid:
                    return 2.0 * (t - t_start) / duration
                else:
                    return 2.0 * (t_end - t) / duration
            # flat: ramp 0.2s up, hold, ramp 0.2s down
            ramp = 0.2
            if t <= t_start + ramp:
                return (t - t_start) / ramp
            if t >= t_end - ramp:
                return (t_end - t) / ramp
            return 1.0

        for _ in range(max_steps):
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            frames.append(img.copy())

            t_now = self._episode_step_count / 30.0  # video time (1 step = 1 frame at 30 FPS)

            # Log eef position every 0.5s
            if self._episode_step_count % 10 == 0:
                eef = obs["robot0_eef_pos"]
                print(f"  [eef] t={t_now:.1f}s  x={eef[0]:.3f}  y={eef[1]:.3f}  z={eef[2]:.3f}")

            # Record eef at window boundaries
            for _tk, tr in _window_trackers.items():
                di = tr["dim_idx"]
                if tr["eef_start"] is None and t_now >= tr["t_start"]:
                    tr["eef_start"] = float(obs["robot0_eef_pos"][di])
                if tr["eef_start"] is not None and t_now <= tr["t_end"]:
                    tr["eef_end"] = float(obs["robot0_eef_pos"][di])

            if len(action_queue) == 0:
                observation = {
                    "full_image": _resize_image(img, self.resize_size, self.cfg.model_family),
                    "wrist_image": _resize_image(wrist_img, self.resize_size, self.cfg.model_family),
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )),
                }
                actions = _robot_utils.get_action(
                    self.cfg, self.model, observation, command,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=self.noisy_action_projector,
                    use_film=self.cfg.use_film,
                )
                action_queue.extend(actions)

            self._episode_step_count += 1
            t_now = self._episode_step_count / 30.0  # video time after increment

            action = process_action(action_queue.popleft(), self.cfg.model_family)

            # Apply bias per-step at exact t_now (smooth interpolation, AdaptVLA style)
            if _compiled_envelopes:
                for env_cfg in _compiled_envelopes:
                    phi = _eval_envelope_at(env_cfg, t_now)
                    if phi == 0.0:
                        continue
                    di = env_cfg["dim_idx"]
                    if env_cfg["suppress"]:
                        action[di] *= (1.0 - phi)
                    if env_cfg["add"]:
                        bias_val = phi * env_cfg["magnitude"]
                        action[di] += bias_val
                        dim = env_cfg["dim"]
                        _dir = "+1" if env_cfg["magnitude"] >= 0 else "-1"
                        _tk = f"{dim}_{_dir}"
                        if _tk in _window_trackers:
                            _window_trackers[_tk]["applied_sum"] += abs(bias_val)
                            _window_trackers[_tk]["n"] += 1
                        print(f"  [bias] t={t_now:.3f}s dim={dim} phi={phi:.3f} bias={bias_val:.4f} action[{di}]={action[di]:.4f}")

            try:
                obs, _, done, _ = self.env.step(action.tolist())
            except ValueError as e:
                if "terminated episode" in str(e):
                    print(f"  [world] env already terminated during act({command[:40]!r}), treating as done")
                    self.episode_success = True
                    break
                raise
            if done:
                self.episode_success = True
                frames.append(get_libero_image(obs).copy())
                break

        self._obs = obs
        self.current_image = _obs_to_pil(obs)
        self.subtask_frame_tuples.append((command, frames))

        # Save per-window eef displacement feedback for VLM calibration
        if _window_trackers:
            import json as _json
            feedback = []
            for _tk, tr in _window_trackers.items():
                dim = tr["dim"]
                # _tk format: "z_+1" or "z_-1" — extract direction sign
                _sign = "+" if _tk.endswith("_+1") else "-"
                eef_s = tr["eef_start"]
                eef_e = tr["eef_end"]
                displacement = round(eef_e - eef_s, 4) if (eef_s is not None and eef_e is not None) else None
                applied = round(tr["applied_sum"] / tr["n"], 4) if tr["n"] > 0 else None
                feedback.append({
                    "dim": f"{_sign}{dim}",
                    "t_start": tr["t_start"], "t_end": tr["t_end"],
                    "applied_bias_mean": applied,
                    "eef_displacement_m": displacement,
                })
                print(f"  [feedback] dim={_sign}{dim} t={tr['t_start']}~{tr['t_end']}s "
                      f"applied={applied} displacement={displacement}m")
            # Only overwrite if this subtask actually observed the bias window (applied is not None).
            # Otherwise keep the previous subtask's feedback which had real values.
            if any(f["applied_bias_mean"] is not None for f in feedback):
                self._last_correction_feedback = feedback

    def ask_tf(self, question: str) -> bool:
        from policies.vlm.vlm_hl.vlm_methods import evaluate_tf_question
        return evaluate_tf_question(question, self.current_image)

    def ask_mc(self, question: str, options) -> str:
        from policies.vlm.vlm_hl.vlm_methods import evaluate_mc_question
        return evaluate_mc_question(question, self.current_image, options)

    def ask_question(self, question: str, options=None) -> str:
        from policies.vlm.vlm_hl.vlm_methods import evaluate_open_question, evaluate_mc_question
        if options is None:
            return evaluate_open_question(question, self.current_image)
        return evaluate_mc_question(question, self.current_image, options)

    def arm_reset(self):
        """No-op: LIBERO sim doesn't need a separate arm reset."""
        pass
