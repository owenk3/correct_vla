"""Utils for evaluating pi0.5 policies via LeRobot."""

from typing import Any, Dict, List

import numpy as np
import torch

# N actions to execute per inference call (loaded from config.yaml)
from policies.vlm.config import get_pi05_config as _get_pi05_cfg
N_ACTION_STEPS = _get_pi05_cfg()["n_action_steps"]
_MAX_TOKEN_LEN = 200

# Module-level caches
_tokenizer = None
_state_mean = None
_state_std = None
_action_mean = None
_action_std = None


def _get_tokenizer():
    return _tokenizer


def _load_norm_stats(checkpoint: str):
    global _state_mean, _state_std, _action_mean, _action_std
    if _state_mean is not None:
        return

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pre = load_file(hf_hub_download(
        repo_id=checkpoint,
        filename="policy_preprocessor_step_2_normalizer_processor.safetensors",
    ))
    _state_mean = pre["observation.state.mean"].float().numpy()
    _state_std = pre["observation.state.std"].float().numpy()

    post = load_file(hf_hub_download(
        repo_id=checkpoint,
        filename="policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ))
    # Print keys on first load to confirm expected structure
    print(f"Postprocessor stats keys: {list(post.keys())}")
    _action_mean = post["action.mean"].float().numpy()
    _action_std = post["action.std"].float().numpy()
    print(f"Action mean: {_action_mean}, std: {_action_std}")


def _tokenize_task(task_label: str, device: torch.device) -> tuple:
    tok = _get_tokenizer()
    enc = tok(
        task_label,
        max_length=_MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)


def get_pi05(cfg: Any):
    """Load pi0.5 policy from HuggingFace checkpoint."""
    global _tokenizer

    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError:
        from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    policy = PI05Policy.from_pretrained(cfg.pretrained_checkpoint)
    policy = policy.to(device)
    policy.eval()
    print(f"Loaded pi0.5 model: {cfg.pretrained_checkpoint}")

    # Load paligemma tokenizer. Try local cache first, then public mirror.
    from transformers import AutoTokenizer
    paligemma = policy.model.paligemma_with_expert.paligemma
    tokenizer_name = paligemma.config._name_or_path
    for name, kwargs in [
        (tokenizer_name, {"local_files_only": True}),
        (tokenizer_name, {}),
        ("CWKSC/paligemma-cord-demo", {}),
    ]:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load paligemma tokenizer. Accept terms at https://huggingface.co/google/paligemma-3b-pt-224 and run huggingface-cli login")

    return policy


def get_pi05_action(cfg: Any, model: Any, obs: Dict[str, Any], task_label: str,
                    action_bias: np.ndarray = None,
                    action_suppress_mask: np.ndarray = None) -> List[np.ndarray]:
    """
    Get action chunk from pi0.5 policy.

    Pipeline:
    - Images: HWC uint8 → CHW float32 [0,1] (model resizes to 224x224 internally)
    - State: MEAN_STD normalized via preprocessor stats
    - predict_action_chunk → normalized actions → MEAN_STD denormalized via postprocessor stats
    - action_bias: optional (N, 7) additive bias applied after denormalization
    """
    device = next(model.parameters()).device
    _load_norm_stats(cfg.pretrained_checkpoint)

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    state_norm = (obs["state"] - _state_mean) / (_state_std + 1e-8)
    tokens, attention_mask = _tokenize_task(task_label, device)

    batch = {
        "observation.images.image": to_tensor(obs["full_image"]),
        "observation.images.image2": to_tensor(obs["wrist_image"]),
        "observation.images.empty_camera_0": torch.zeros(1, 3, 224, 224, device=device),
        "observation.state": torch.from_numpy(state_norm.astype(np.float32)).unsqueeze(0).to(device),
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": attention_mask,
    }

    with torch.no_grad():
        action_chunk = model.predict_action_chunk(batch)  # (1, chunk_size, 7) — normalized

    action_chunk = action_chunk[0].cpu().float().numpy()  # (chunk_size, 7)
    # Denormalize: model outputs in normalized space, postprocessor reverses MEAN_STD
    action_chunk = action_chunk * (_action_std + 1e-8) + _action_mean

    n = min(N_ACTION_STEPS, len(action_chunk))
    if action_suppress_mask is not None:
        action_chunk[:n] *= action_suppress_mask[:n]
    if action_bias is not None:
        action_chunk[:n] += action_bias[:n]

    return [action_chunk[i] for i in range(n)]


def get_action_std(cfg) -> "np.ndarray | None":
    """Return the loaded action std (7,) array, or None if not yet loaded."""
    _load_norm_stats(cfg.pretrained_checkpoint)
    return _action_std
