"""
PyPlan Utilities

This module contains utility functions for PyPlan execution including
logging setup, world initialization, and file I/O operations.
"""

import os
import base64
from io import BytesIO
from PIL import Image
import cv2
import tempfile
import shutil


def get_next_run_number(program_dir) -> int:
    """Get the next available run number."""
    ae_dir = f"{program_dir}/angelic_execution"
    if not os.path.exists(ae_dir):
        return 1

    existing_runs = [
        d
        for d in os.listdir(ae_dir)
        if d.startswith("run_") and os.path.isdir(f"{ae_dir}/{d}")
    ]
    if not existing_runs:
        return 1

    run_numbers = []
    for run_dir in existing_runs:
        try:
            run_num = int(run_dir.split("_")[1])
            run_numbers.append(run_num)
        except (IndexError, ValueError):
            continue  # Skip malformed directory names

    return max(run_numbers) + 1 if run_numbers else 1

def save_top_level_ica_dir(parent_dir, image0, image1, task, success, assessment,
                           human_correction=None, correction_params=None):
    """
    Save top level (overall task) ICA directory structure.
    correction_params: CorrectionParams pydantic object or None
    """
    import json
    os.makedirs(parent_dir, exist_ok=True)

    image0.save(os.path.join(parent_dir, "image0.png"))
    image1.save(os.path.join(parent_dir, "image1.png"))

    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    with open(os.path.join(parent_dir, "success.txt"), "w") as f:
        f.write(str(success))

    with open(os.path.join(parent_dir, "assessment.txt"), "w") as f:
        f.write(assessment)

    if human_correction:
        with open(os.path.join(parent_dir, "human_correction.txt"), "w") as f:
            f.write(human_correction)

    if correction_params is not None:
        with open(os.path.join(parent_dir, "correction_params.json"), "w") as f:
            data = correction_params if isinstance(correction_params, dict) else correction_params.model_dump()
            json.dump(data, f, indent=2)

    return parent_dir


def save_icl_dir(parent_dir, image0, task):
    """
    Save top level (overall task) ICL directory structure.
    """
    os.makedirs(parent_dir, exist_ok=True)

    # Save image
    image0.save(os.path.join(parent_dir, "image0.png"))

    # Save task
    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    return parent_dir


def save_ablation_dir(parent_dir, image0, task, success):
    """
    Save subtask directory structure for no-reasoning ablation.
    """
    os.makedirs(parent_dir, exist_ok=True)

    # Save image
    image0.save(os.path.join(parent_dir, "image0.png"))

    # Save task
    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    # Save success
    with open(os.path.join(parent_dir, "success.txt"), "w") as f:
        f.write(str(success))

    return parent_dir


def save_who_ablation_dir(parent_dir, image0, task, success, whathappened):
    """
    Save subtask directory structure for what-happened only ablation.
    """
    os.makedirs(parent_dir, exist_ok=True)

    # Save image
    image0.save(os.path.join(parent_dir, "image0.png"))

    # Save task
    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    # Save success
    with open(os.path.join(parent_dir, "success.txt"), "w") as f:
        f.write(str(success))

    # Save whathappened
    if whathappened is not None:
        with open(os.path.join(parent_dir, "whathappened.txt"), "w") as f:
            f.write(whathappened)

    return parent_dir


def load_icl_dir(dir_path):
    """
    Loader and utility for the positive ICL baseline.
    """
    image0_path = os.path.join(dir_path, "image0.png")
    task_path = os.path.join(dir_path, "task.txt")

    image0 = Image.open(image0_path) if os.path.exists(image0_path) else None
    task = None
    if os.path.exists(task_path):
        with open(task_path, "r") as f:
            task = f.read().strip()

    return image0, task


def load_ablation_dir(dir_path):
    """
    Loader and utility for the no-reasoning ablation.
    """
    image0_path = os.path.join(dir_path, "image0.png")
    task_path = os.path.join(dir_path, "task.txt")
    success_path = os.path.join(dir_path, "success.txt")

    image0 = Image.open(image0_path) if os.path.exists(image0_path) else None
    task = None
    if os.path.exists(task_path):
        with open(task_path, "r") as f:
            task = f.read().strip()
    if os.path.exists(success_path):
        with open(success_path, "r") as f:
            success = f.read().strip().lower() == "true"
    return image0, task, success


def load_who_ablation_dir(dir_path):
    """
    Loader and utility for the what-happened only ablation.
    """
    image0_path = os.path.join(dir_path, "image0.png")
    task_path = os.path.join(dir_path, "task.txt")
    success_path = os.path.join(dir_path, "success.txt")
    whathappened_path = os.path.join(dir_path, "whathappened.txt")

    image0 = Image.open(image0_path) if os.path.exists(image0_path) else None
    task = None
    if os.path.exists(task_path):
        with open(task_path, "r") as f:
            task = f.read().strip()
    if os.path.exists(success_path):
        with open(success_path, "r") as f:
            success = f.read().strip().lower() == "true"
    if os.path.exists(whathappened_path):
        with open(whathappened_path, "r") as f:
            whathappened = f.read().strip()
    else:
        whathappened = None
    return image0, task, success, whathappened


def save_reflexion_dir(parent_dir, image0, task, reflection):
    """
    Save top level (overall task) ICL directory structure.
    """
    os.makedirs(parent_dir, exist_ok=True)

    # Save image
    image0.save(os.path.join(parent_dir, "image0.png"))

    # Save task
    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    # Save reflection
    with open(os.path.join(parent_dir, "reflection.txt"), "w") as f:
        f.write(reflection)

    return parent_dir


def load_reflexion_dir(dir_path):
    """
    Loader and utility for the positive ICL baseline.
    """
    image0_path = os.path.join(dir_path, "image0.png")
    task_path = os.path.join(dir_path, "task.txt")
    reflection_path = os.path.join(dir_path, "reflection.txt")

    image0 = Image.open(image0_path) if os.path.exists(image0_path) else None
    task = None
    if os.path.exists(task_path):
        with open(task_path, "r") as f:
            task = f.read().strip()
    if os.path.exists(reflection_path):
        with open(reflection_path, "r") as f:
            reflection = f.read().strip()

    return image0, task, reflection


def save_reasoning_ica_dir(parent_dir, image, task, success, whathappened, reasoning):
    """
    Save ICA v2 directory structure with provided data.

    Args:
        parent_dir (str): Path to the parent directory (e.g., 'ica_v2_examples_test').
        image (PIL.Image): Image to save as 'image0.png'.
        success (bool): Success status to save in 'success.txt'.
        whathappened (str): Description of what happened to save in 'whathappened.txt'.
        reasoning (str): Reasoning text to save in 'reasoning.txt'.
    """
    os.makedirs(parent_dir, exist_ok=True)

    # Save image
    image.save(os.path.join(parent_dir, "image0.png"))

    # Save task
    with open(os.path.join(parent_dir, "task.txt"), "w") as f:
        f.write(task)

    # Save success
    with open(os.path.join(parent_dir, "success.txt"), "w") as f:
        f.write(str(success))

    # Save whathappened
    if whathappened is not None:
        with open(os.path.join(parent_dir, "whathappened.txt"), "w") as f:
            f.write(whathappened)

    # Save reasoning
    with open(os.path.join(parent_dir, "reasoning.txt"), "w") as f:
        f.write(reasoning)

    return parent_dir

def encode_image_to_base64(image: Image) -> str:
    """
    Helper method to encode image to base64.

    Args:
        image: PIL Image to encode

    Returns:
        Base64 encoded image string
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def fix_mp4_color(path: str):
    """
    Reads an mp4 file, converts frames from RGB to BGR,
    and overwrites the file with corrected colors.
    """
    # Open input video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Write to a temp file first (to avoid corruption if interrupted)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert RGB -> BGR
        corrected = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(corrected)

    cap.release()
    out.release()

    # Overwrite original file
    shutil.move(tmp_path, path)
