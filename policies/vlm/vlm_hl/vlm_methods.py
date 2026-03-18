import os
from PIL import Image
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
import base64
from io import BytesIO
import cv2
from policies.vlm.ica.reasoning_ica import ReasoningICADir, TaskICADir

# Resolve prompt paths relative to this file, regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_PLAN = os.path.join(_HERE, "prompts", "plan_reasoning")
_PROMPTS_ASSESS = os.path.join(_HERE, "prompts", "assessment")

def _p(subdir, name):
    """Return absolute path to a prompt file."""
    return os.path.join(subdir, name)

def _load_openai_key():
    """Load OPENAI_API_KEY from correctvla/.env if not already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return

_load_openai_key()

def _client():
    """Return OpenAI client, always using current env key."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Load model names from config.yaml
from policies.vlm.config import get_vlm_config as _get_cfg
_cfg = _get_cfg()
vision_model = _cfg["openai_vision_model"]
text_model = _cfg["openai_text_model"]
ica_model = _cfg["openai_reasoning_model"]

token_usage = 0


class LLMStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class TFAnswer(BaseModel):
    answer: bool


from typing import Literal, Optional

class CorrectionParams(BaseModel):
    """Structured parameters extracted from a human correction string."""
    dimension: Literal["x", "y", "z", "roll", "pitch", "yaw", "gripper", "unknown"]
    direction: Literal["+1", "-1", "unknown"]   # +1 = positive axis / open, -1 = negative / close
    magnitude_term: Literal["slightly", "more", "much", "unknown"]
    magnitude_value: float                       # signed numeric: direction * quantified magnitude
    t_start: Optional[float]                     # seconds into episode, None if unspecified
    t_end: Optional[float]
    reason: str                                  # plain-English root cause
    mode: Literal["add", "suppress", "override"] = "add"  # add=additive, suppress=zero VLA, override=zero VLA + add bias


def extract_correction_params(correction: str, task_description: str) -> CorrectionParams:
    """
    Phase A+B: Parse a human correction string into structured parameters.
    Uses a fast text model — no image needed.
    """
    prompt = f"""You are parsing a human expert's robot correction into structured parameters.

Task: {task_description}
Human correction: {correction}

Coordinate frame (robot arm in LIBERO):
  X-axis: Forward (+X) / Backward (-X)
  Y-axis: Right (+Y) / Left (-Y)
  Z-axis: Up (+Z) / Down (-Z)
  roll, pitch, yaw: rotations around X, Y, Z respectively
  gripper: open (+1) / close (-1)

Extract the following fields:

- dimension: which action axis is being corrected? Choose from: x, y, z, roll, pitch, yaw, gripper, or unknown
- direction: +1 (positive axis: forward/right/up/open) or -1 (negative: backward/left/down/close), or unknown
- magnitude_term: the qualitative magnitude — one of: slightly (small correction), more (medium), much (large), or unknown
- magnitude_value: a signed float. Map magnitude_term to a value in its range, then apply direction sign:
    slightly → unsigned magnitude in [0.3, 0.5], more → [0.5, 0.8], much → [0.8, 1.0]
    unknown → use 0.4 as default unsigned, then apply direction sign
- t_start: start of the time window in seconds (e.g. 5.0). null if not specified.
- t_end: end of the time window in seconds (e.g. 6.0). null if not specified.
- reason: one sentence describing the root cause of the failure as stated by the human.

Examples:
  "move right" → dimension=y, direction=+1
  "move left" → dimension=y, direction=-1
  "move forward" → dimension=x, direction=+1
  "move backward" → dimension=x, direction=-1
  "lift up" → dimension=z, direction=+1
  "lower down" → dimension=z, direction=-1"""

    response = _client().responses.parse(
        model=text_model,
        input=[{"role": "user", "content": prompt}],
        text_format=CorrectionParams,
    )
    return response.output_parsed


class MultiAxisCorrectionParams(BaseModel):
    """Multi-axis correction params extracted from a human correction string."""
    axes: list[CorrectionParams]


def extract_correction_params_multi(correction: str, task_description: str) -> dict:
    """
    Like extract_correction_params but supports multiple axes.
    Returns {"axes": [<CorrectionParams dicts>]}.
    """
    prompt = f"""You are parsing a human expert's robot correction into structured parameters.
The correction may specify multiple axes and/or multiple time windows — extract ALL of them.

Task: {task_description}
Human correction: {correction}

Coordinate frame (robot arm in LIBERO):
  X-axis: Forward (+X) / Backward (-X)
  Y-axis: Right (+Y) / Left (-Y)
  Z-axis: Up (+Z) / Down (-Z)
  roll, pitch, yaw: rotations around X, Y, Z respectively
  gripper: open (+1) / close (-1)

For EACH separate directional instruction in the correction, extract:
- dimension: x, y, z, roll, pitch, yaw, gripper, or unknown
- direction: +1 (forward/right/up/open) or -1 (backward/left/down/close), or unknown
- magnitude_term: slightly / more / much / unknown
- magnitude_value: UNSIGNED positive float (the direction field encodes the sign). Map: slightly→[0.3,0.5], more→[0.5,0.8], much→[0.8,1.0]. If magnitude_value is already given as a float in the text, use its absolute value.
- t_start: seconds (null if unspecified)
- t_end: seconds (null if unspecified)
- reason: one sentence per axis
- mode: "override" if the text says "override" — zeroes out VLA action for this dim and adds bias. "suppress" if the text says "suppress" — zeroes VLA action, no bias. "add" otherwise (additive bias on top of VLA).

Examples:
  "-Y magnitude_value=0.16 t=0.5~5.5s override" → dimension=y, direction=-1, magnitude_value=0.16, mode=override
  "suppress X t=0.5~5.5s" → dimension=x, mode=suppress, direction=unknown, magnitude_value=0.0
  "move left" → y, -1, mode=add
  "lower down" → z, -1, mode=add"""

    response = _client().responses.parse(
        model=text_model,
        input=[{"role": "user", "content": prompt}],
        text_format=MultiAxisCorrectionParams,
    )
    parsed = response.output_parsed
    return {"axes": [a.model_dump() for a in parsed.axes]}


def format_obj_list(objects_list):
    """
    Given a list of objects, format them into a string that can be used in the prompt.
    """
    if objects_list is None:
        return ""
    formatted_obj_list = ""
    for obj in objects_list:
        formatted_obj_list += f"{obj}, "
    return formatted_obj_list[:-2]

def get_hardware_specific_instruction_space():
    """
    Load the instruction space details from a text file for your task and hardware setup.
    We've provided an example for the DROID setup here.
    You should modify this function for your own setup and hardware.
    """
    with open(_p(_PROMPTS_PLAN, "droid_specific_instruction_space.txt"), "r") as f:
        sp_instruction_space = f.readlines()
    return format_obj_list(sp_instruction_space)

def vlm_call_with_image(
    image: Image.Image, prompt: str, model: str = vision_model, tf: bool = False
):
    """
    Calls the OpenAI API with an image and a prompt, returning the response.
    """
    image_b64 = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            ],
        }
    ]
    if tf:
        response = _client().responses.parse(
            model=model,
            input=messages,
            text_format=TFAnswer,
        )
        return response.output_parsed.answer
    else:
        response = _client().responses.create(model=model, input=messages)
        vlm_response = response.output_text.strip()
        return vlm_response


def vlm_call_with_text(prompt: str, model: str = text_model, tf: bool = False):
    """
    Calls the OpenAI API with a text prompt, returning the response.
    """
    if tf:
        response = _client().responses.parse(
            model=model,
            input=prompt,
            text_format=TFAnswer,
        )
        return response.output_parsed.answer
    else:
        response = _client().responses.create(
            model=model, input=[{"role": "user", "content": prompt}]
        )
        vlm_response = response.output_text.strip()
        return vlm_response


def encode_image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def evaluate_tf_question(question, image):
    print("Asking VLM the following question: ", question)
    with open(_p(_PROMPTS_PLAN, "evaluate_tf_question.txt"), "r") as file:
        prompt = file.read().format(question=question)
    response = vlm_call_with_image(image, prompt, tf=True)
    print("VLM response: ", response)
    return response


def evaluate_mc_question(question, image, options_list):
    """
    Given a single multiple choice question, evaluate the question and return an option from the multiple choice list.
    """
    print("Asking VLM the following question: ", question)
    with open(_p(_PROMPTS_PLAN, "evaluate_mc_question.txt"), "r") as file:
        prompt = file.read()
    prompt = prompt.format(question=question, options=format_obj_list(options_list))
    image_b64 = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            ],
        }
    ]
    OptionsEnum = Enum("Options", options_list)

    class MCOptions(BaseModel):
        selection: OptionsEnum

    response = _client().responses.parse(
        model=text_model,
        input=messages,
        text_format=MCOptions,
    )
    print("VLM response: ", response.output_parsed.selection.name)
    return response.output_parsed.selection.name


def evaluate_open_question(question, image):
    print("Asking VLM the following question: ", question)
    with open(_p(_PROMPTS_PLAN, "evaluate_open_question.txt"), "r") as file:
        prompt = file.read().format(question=question)
    response = vlm_call_with_image(image, prompt)
    print("VLM response: ", response)
    return response


class ObjectUids(BaseModel):
    object_uids: list[str]


def get_object_uids_from_scene(
    current_image: Image, task_instruction: str
):
    """
    Given an image, query a VLM to identify objects in the image and output a list of objects.
    """
    with open(_p(_PROMPTS_PLAN, "generate_uids.txt"), "r") as file:
        prompt = file.read()
    prompt = prompt.format(
        instruction=task_instruction
    )
    # now, get the response from the VLM
    image_b64 = encode_image_to_base64(current_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            ],
        }
    ]
    response = _client().responses.parse(
        model=vision_model,
        input=messages,
        text_format=ObjectUids,
    )
    print("Identified objects from scene: ", response.output_parsed.object_uids)
    return response.output_parsed.object_uids

def extract_frames(video_path, frame_rate=10):
    """
    Extract frames from a video.
    Always includes the first and last frame, plus every `frame_rate`th frame.
    Returns a list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames_b64 = []
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save first, every `frame_rate`th, and last frame
        if count == 0 or count % frame_rate == 0 or count == total_frames - 1:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(frame_b64)

        count += 1

    cap.release()
    return frames_b64


def extract_frames_from_list(frame_list, frame_rate=10):
    """
    Extract frames from a list of numpy array images.
    Always includes the first and last frame, plus every `frame_rate`th frame.
    Returns a list of base64-encoded JPEG strings.
    """
    frames_b64 = []
    total_frames = len(frame_list)
    for count, frame in enumerate(frame_list):
        if count == 0 or count % frame_rate == 0 or count == total_frames - 1:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(frame_b64)
    return frames_b64


def build_reasoning_tuple_subdir(
    overall_task_idx: int, attempt_idx: int, icadir: ReasoningICADir
):
    """
    Build an OpenAI API message structure for a robot attempt directory using ReasoningICADir.
    """
    att_pre = f"VLA_ATTEMPT_{overall_task_idx}_{attempt_idx}"

    # Load images
    reasoning_tuple_dir = icadir.get_reasoning_tuple()
    initial_image = reasoning_tuple_dir["image0"]
    task = reasoning_tuple_dir["task"]
    success = reasoning_tuple_dir["success"]
    whathappened = reasoning_tuple_dir["whathappened"]
    reasoning = reasoning_tuple_dir["reasoning"]

    # Build message content
    content = [
        {"type": "input_text", "text": f"`[{att_pre}]`:"},
        {"type": "input_text", "text": f"`[{att_pre}_INITIAL_IMAGE]`:"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(initial_image)}",
        },
        {
            "type": "input_text",
            "text": f"`[VLA_INSTRUCTION {overall_task_idx}_{attempt_idx}]`: {task}",
        },
        {
            "type": "input_text",
            "text": f"`[{att_pre}_SUCCESS]`: {success}",
        },
    ]
    if whathappened is not None:
        content.append(
            {
                "type": "input_text",
                "text": f"`[{att_pre}_WHAT_HAPPENED]`: {whathappened}",
            }
        )
    content.append(
        {"type": "input_text", "text": f"`[{att_pre}_REASONING]`: {reasoning}"}
    )

    user_msg = {"role": "user", "content": content}
    return user_msg


def build_top_level_reasoning_tuple(tlicadir: TaskICADir, overall_task_idx: int):
    exec_pre = f"EXECUTION_{overall_task_idx}"

    # Load images
    task_tuple_dict = tlicadir.get_task_tuple()
    initial_image = task_tuple_dict["image0"]
    final_image = task_tuple_dict["image1"]
    overall_task = task_tuple_dict["task"]
    success = task_tuple_dict["success"]
    assessment = task_tuple_dict["assessment"]
    subtask_ica_dirs = task_tuple_dict["subtasks"]
    human_correction = task_tuple_dict.get("human_correction")

    # build subtask messages
    subtask_attempt_messages = build_multi_reasoning_tuples(
        subtask_ica_dirs, overall_task_idx
    )

    # Build message content
    content = [
        {"type": "input_text", "text": f"`[{exec_pre}]`:"},
        {"type": "input_text", "text": f"`[{exec_pre}_INITIAL_IMAGE]`:"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(initial_image)}",
        },
        {
            "type": "input_text",
            "text": f"`[OVERALL TASK {overall_task_idx}]`: {overall_task}",
        },
        {"type": "input_text", "text": f"`[{exec_pre}_FINAL_IMAGE]`:"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(final_image)}",
        },
        {
            "type": "input_text",
            "text": f"`[SUCCESS {overall_task_idx}]`: {success}",
        },
        {
            "type": "input_text",
            "text": f"`[ASSESSMENT {overall_task_idx}]`: {assessment}",
        },
    ]
    if human_correction:
        content.append({
            "type": "input_text",
            "text": f"`[HUMAN_CORRECTION {overall_task_idx}]`: {human_correction}",
        })

    top_lvl_msg = {"role": "user", "content": content}
    user_msgs = [top_lvl_msg] + subtask_attempt_messages
    return user_msgs


def build_multi_reasoning_tuples(icadirs: list[ReasoningICADir], overall_task_idx: int):
    """
    Build a combined OpenAI API message structure for multiple robot attempts.

    Args:
        icadirs (list[ReasoningICADir]): List of ReasoningICADir instances.
    Returns:
        messages (list[dict]): A list suitable for OpenAI's Responses API.
    """
    messages = []
    for idx, icadir in enumerate(icadirs):
        messages.append(build_reasoning_tuple_subdir(overall_task_idx, idx, icadir))
    return messages


def build_multi_task_tuples(taskicadirs: list[TaskICADir]):
    """
    Build a combined OpenAI API message structure for multiple robot attempts.

    Args:
        icadirs (list[ReasoningICADir]): List of ReasoningICADir instances.
    Returns:
        messages (list[dict]): A list suitable for OpenAI's Responses API.
    """
    messages = []
    for idx, icadir in enumerate(taskicadirs):
        messages.extend(build_top_level_reasoning_tuple(icadir, idx))
    return messages




def critique_vla_failure(
    initial_image: Image, final_image: Image, task_description: str
):
    with open(_p(_PROMPTS_ASSESS, "critique_failure_from_images.txt"), "r") as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _client().responses.create(
        model=vision_model,
        input=messages,
    )
    return response.output_text.strip()


def critique_vla_video_failure(
    video_frames: list, task_description: str, frame_rate=20
):
    with open(_p(_PROMPTS_ASSESS, "critique_failure_from_video.txt"), "r") as file:
        unformatted_prompt = file.read()
    frames_b64 = extract_frames_from_list(video_frames, frame_rate=frame_rate)
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    content = [{"type": "input_text", "text": formatted_prompt}]
    content.extend(
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
        for b64 in frames_b64
    )
    messages = [{"role": "user", "content": content}]
    response = _client().responses.create(
        model=ica_model,
        input=messages,
    )
    return response.output_text.strip()


def critique_vla_video_with_correction(
    video_frames: list, task_description: str, human_correction: str, frame_rate=20
):
    """Analyze failure video through the lens of the human correction."""
    with open(_p(_PROMPTS_ASSESS, "critique_failure_with_correction.txt"), "r") as f:
        prompt = f.read().format(task_instruction=task_description, human_correction=human_correction)
    frames_b64 = extract_frames_from_list(video_frames, frame_rate=frame_rate)
    content = [{"type": "input_text", "text": prompt}]
    content.extend(
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
        for b64 in frames_b64
    )
    response = _client().responses.create(model=vision_model, input=[{"role": "user", "content": content}])
    return response.output_text.strip()


def _format_correction_feedback(feedback: list) -> str:
    """Format eef displacement feedback list into a readable string for the VLM prompt."""
    if not feedback:
        return "No feedback available (first refinement attempt)."
    lines = []
    for fb in feedback:
        lines.append(
            f"  dim={fb.get('dim')} | window={fb.get('t_start')}~{fb.get('t_end')}s | "
            f"applied_bias_mean={fb.get('applied_bias_mean')} m/step | "
            f"eef_displacement={fb.get('eef_displacement_m')} m"
        )
    return "\n".join(lines)


_last_refine_full_response: dict = {}


def get_last_refine_full_response(tag: str) -> str:
    return _last_refine_full_response.get(tag, "")


def _refine_correction_call(
    prompt_file: str, tag: str,
    failure_frames: list, attempt_frames: list,
    task_description: str, human_correction: str, frame_rate=10,
    correction_feedback: list = None,
    previous_reasoning: str = None,
    success_feedback: list = None,
) -> str:
    with open(_p(_PROMPTS_ASSESS, prompt_file), "r") as f:
        template = f.read()
    feedback_str = _format_correction_feedback(correction_feedback or [])
    success_feedback_str = _format_correction_feedback(success_feedback or [])
    prev_reasoning_str = previous_reasoning or "No previous reasoning available (first refinement attempt)."
    prompt = template.replace("{task_instruction}", task_description)\
                     .replace("{human_correction}", human_correction)\
                     .replace("{correction_feedback}", feedback_str)\
                     .replace("{previous_reasoning}", prev_reasoning_str)\
                     .replace("{success_feedback}", success_feedback_str)
    failure_b64 = extract_frames_from_list(failure_frames, frame_rate=frame_rate)
    attempt_b64 = extract_frames_from_list(attempt_frames, frame_rate=frame_rate)
    content = [{"type": "input_text", "text": prompt + "\n\n[PART 1] Original failure video:"}]
    content.extend({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"} for b64 in failure_b64)
    content.append({"type": "input_text", "text": "[PART 2] Attempt video after correction was applied:"})
    content.extend({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"} for b64 in attempt_b64)
    response = _client().responses.create(model=vision_model, input=[{"role": "user", "content": content}])
    full = response.output_text.strip()
    print(f"  [{tag}-refine] full response:\n{full}")
    _last_refine_full_response[tag] = full
    # Collect ALL CORRECTION lines (VLM may output one per axis)
    corrections = []
    for line in full.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CORRECTION:"):
            corrections.append(stripped[len("CORRECTION:"):].strip())
    if corrections:
        return " | ".join(corrections)  # join for multi-axis parsing
    lines = [l.strip() for l in full.splitlines() if l.strip()]
    return lines[-1] if lines else full


def refine_human_correction_a18(
    failure_frames: list, attempt_frames: list, task_description: str, human_correction: str,
    frame_rate=10, correction_feedback: list = None, previous_reasoning: str = None,
) -> str:
    """a18: a17 + semantic reasoning (task goal, root cause, what correction achieved) carried forward across attempts."""
    return _refine_correction_call(
        "refine_correction_from_attempt_a18.txt", "a18",
        failure_frames, attempt_frames, task_description, human_correction, frame_rate,
        correction_feedback=correction_feedback,
        previous_reasoning=previous_reasoning,
    )


def generate_correction_from_failure(
    failure_frames: list, task_description: str, frame_rate=10,
) -> str:
    """Watch a failure video (no human correction) and generate correction params from scratch."""
    with open(_p(_PROMPTS_ASSESS, "generate_correction_from_failure.txt"), "r") as f:
        template = f.read()
    prompt = template.replace("{task_instruction}", task_description)
    failure_b64 = extract_frames_from_list(failure_frames, frame_rate=frame_rate)
    content = [{"type": "input_text", "text": prompt + "\n\nFailure video:"}]
    content.extend({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"} for b64 in failure_b64)
    response = _client().responses.create(model=vision_model, input=[{"role": "user", "content": content}])
    full = response.output_text.strip()
    print(f"  [llm-generate] full response:\n{full}")
    _last_refine_full_response["llm"] = full
    corrections = []
    for line in full.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CORRECTION:"):
            corrections.append(stripped[len("CORRECTION:"):].strip())
    if corrections:
        return " | ".join(corrections)
    lines = [l.strip() for l in full.splitlines() if l.strip()]
    return lines[-1] if lines else full


def format_reasoning_tuples(reasoning_tuples: list):
    formatted_tuples = ""
    for i, (subtask, success, what_happened, reasoning) in enumerate(reasoning_tuples):
        if what_happened is None:
            what_happened = "N/A"
        formatted_tuples += f"Subtask {i+1}:\n Subtask Instruction: {subtask}\n Success?: {success}\n What Happened (if failure): {what_happened}\n Reasoning: {reasoning}\n"
    return formatted_tuples


def assess_hl_success(
    initial_image: Image,
    final_image: Image,
    task_description: str,
    reasoning_tuples: list,
):
    with open(
        _p(_PROMPTS_ASSESS, "describe_hl_success_scene.txt"), "r"
    ) as file:
        unformatted_prompt = file.read()
    with open(_p(_PROMPTS_ASSESS, "pizerofive_droid_vla_model_card.txt"), "r") as file:
        model_card = file.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        subtask_reasoning_tuples=format_reasoning_tuples(reasoning_tuples),
    )
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _client().responses.create(
        model=vision_model,
        input=messages,
    )
    return response.output_text.strip()


def assess_hl_failure(
    initial_image: Image,
    final_image: Image,
    task_description: str,
    reasoning_tuples: list,
):
    with open(
        _p(_PROMPTS_ASSESS, "critique_hl_failure_scene.txt"), "r"
    ) as file:
        unformatted_prompt = file.read()
    with open(_p(_PROMPTS_ASSESS, "pizerofive_droid_vla_model_card.txt"), "r") as file:
        model_card = file.read()
    with open(
        _p(_PROMPTS_PLAN, "pizerofive_droid_instruction_space.txt"), "r", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        subtask_reasoning_tuples=format_reasoning_tuples(reasoning_tuples),
        instruction_space=instruction_space,
    )
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _client().responses.create(
        model=vision_model,
        input=messages,
    )
    return response.output_text.strip()


def describe_vla_success(initial_image: Image, task_description: str):
    with open(
        _p(_PROMPTS_ASSESS, "describe_success_scene.txt"), "r"
    ) as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    response = vlm_call_with_image(initial_image, formatted_prompt, model=text_model)
    return response


def determine_vla_success(
    initial_image: Image, final_image: Image, task_description: str
):
    with open(
        _p(_PROMPTS_ASSESS, "determine_success_from_images.txt"), "r"
    ) as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _client().responses.parse(
        model=vision_model,
        input=messages,
        text_format=TFAnswer,
    )
    return response.output_parsed.answer


def reason_about_vla_failure(
    initial_image: Image, task_description: str, what_happened: str
):
    with open(_p(_PROMPTS_ASSESS, "reason_about_failure.txt"), "r") as file:
        unformatted_prompt = file.read()

    with open(_p(_PROMPTS_ASSESS, "pizerofive_droid_vla_model_card.txt"), "r") as file:
        model_card = file.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        what_happened=what_happened,
    )

    response = vlm_call_with_image(initial_image, formatted_prompt, model=text_model)
    return response


def format_two_image_message(
    initial_image: Image, final_image: Image, formatted_prompt: str
):
    initial_image_b64 = encode_image_to_base64(initial_image)
    final_image_b64 = encode_image_to_base64(final_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": formatted_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{initial_image_b64}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{final_image_b64}",
                },
            ],
        }
    ]
    return messages