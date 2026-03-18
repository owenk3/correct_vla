# CorrectVLA

Human corrective feedback → timed action bias for VLA failure recovery.

## Setup

### OpenVLA-OFT

```bash
conda create -n correctvla python=3.10 -y && conda activate correctvla
pip install uv
git clone https://github.com/owenk3/correctvla.git && cd correctvla
uv pip install -e . && uv pip install "numpy==1.26.4" "transformers==4.40.1"
uv pip install "flash-attn==2.5.5" --no-build-isolation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv pip install -e LIBERO && uv pip install "robosuite==1.4.1" bddl easydict cloudpickle gym "imageio[ffmpeg]" matplotlib
```

### pi0.5

```bash
conda create -n correctvla-pi05 python=3.10 -y && conda activate correctvla-pi05
pip install uv && cd correctvla
uv pip install -e . --no-deps
uv pip install "numpy==1.26.4" draccus wandb tqdm pillow matplotlib pyyaml pydantic
uv pip install "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0"
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
uv pip install "git+https://github.com/huggingface/lerobot.git"
uv pip install openai opencv-python
uv pip install -e LIBERO && uv pip install "robosuite==1.4.1" bddl easydict cloudpickle gym "imageio[ffmpeg]"
```

## Eval

### Vanilla (no corrections)

```bash
# OpenVLA-OFT
PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO python experiments/robot/libero/vla_eval.py \
  --pretrained_checkpoint <checkpoint_path> \
  --model_family openvla --task_suite_name libero_spatial

# pi0.5
PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO python experiments/robot/libero/vla_eval.py \
  --pretrained_checkpoint lerobot/pi05_libero_finetuned \
  --model_family pi05 --task_suite_name libero_90
```

### Ours (human corrective feedback)

1. **Run vanilla eval** — observe where the policy fails
2. **Watch the failure video** — identify what went wrong and when
3. **Write a correction** — natural language timed feedback
4. **Re-run with correction** — action bias is applied during the specified time window

#### Writing Corrections

```
corrections/{model}/{suite}/ep_{N}/v{V}/correction_params.json
```

**Feedback format**: `"direction magnitude t_start~t_end"`, separated by `;`
```
"down slightly 2.2~2.6s; left more 3.0~3.5s"
```
Magnitude: `slightly` (1× std), `more` (2×), `much` (4×), or numeric (`+0.2`). Directions: left/right (Y), forward/back (X), up/down (Z).

Write corrections via CLI:
```bash
python write_correction.py --model openvla --suite libero_spatial --episode 96 \
  --feedback "down slightly 2.2~2.6s; left more 3.0~3.5s"
```

#### Step-by-Step Example

```bash
# 1. Run vanilla — generates failure video in experiments/logs/
PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO python experiments/robot/libero/vla_eval_failures.py \
  --model_family openvla --task_suite_name libero_spatial \
  --eval vanilla --episode 96

# 2. Watch video, write correction
python write_correction.py --model openvla --suite libero_spatial --episode 96 \
  --feedback "right more 1.5~2.5s; down slightly 3.0~4.0s"

# 3. Re-run with correction applied
PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO python experiments/robot/libero/vla_eval_failures.py \
  --model_family openvla --task_suite_name libero_spatial \
  --eval ours --episode 96

# 4. Compare against LLM baseline
PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO python experiments/robot/libero/vla_eval_failures.py \
  --model_family openvla --task_suite_name libero_spatial \
  --eval llm_baseline --episode 96
```

