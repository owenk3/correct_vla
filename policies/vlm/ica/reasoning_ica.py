import os
from PIL import Image


class GenericICADir:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def _find_file(self, filename):
        path = os.path.join(self.dir_path, filename)
        return path if os.path.exists(path) else None

    def read_text(self, path):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def load_success(self):
        # Try to load from success.txt (expects "True" or "False" in file)
        if self.success_path:
            val = self.read_text(self.success_path)
            return val.strip().lower() == "true"
        return None


class TaskICADir(GenericICADir):
    """
    Top-level directory for long-horizon tasks.
    This class will contain multiple ReasoningICADirs (one for each subtask).
    In addition, it will contain metadata that includes the top-level task description, and whether or not
    the overall task was successful (this will be determined by a human). It will reason about the success
    of the overall task based on the individual subtasks (and their success information).
    - image0.png: initial scene image
    - image1.png: final scene image
    - task.txt: top-level task instruction
    - success.txt: whether or not the overall task succeeded (human labeled)
    - assessment.txt: description of what actually happened
    """

    def __init__(self, dir_path):
        super().__init__(dir_path)
        self.ica_dirs = self._load_ica_dirs()
        self.image0_path = self._find_file("image0.png")
        self.image1_path = self._find_file("image1.png")
        self.task_path = self._find_file("task.txt")
        self.success_path = self._find_file("success.txt")
        self.assessment_path = self._find_file("assessment.txt")
        self.human_correction_path = self._find_file("human_correction.txt")
        self.correction_params_path = self._find_file("correction_params.json")

    def _load_ica_dirs(self):
        ica_dirs = []
        for entry in os.listdir(self.dir_path):
            if not entry.startswith("subtask_"):
                continue
            full_path = os.path.join(self.dir_path, entry)
            if os.path.isdir(full_path):
                ica_dirs.append(ReasoningICADir(full_path))
        return sorted(ica_dirs, key=lambda d: d.dir_path)

    def get_task_tuple(self):
        import json
        image0 = Image.open(self.image0_path) if self.image0_path else None
        image1 = Image.open(self.image1_path) if self.image1_path else None
        task = self.read_text(self.task_path)
        success = self.load_success()
        assessment = self.read_text(self.assessment_path)
        human_correction = self.read_text(self.human_correction_path)
        correction_params = None
        if self.correction_params_path and os.path.exists(self.correction_params_path):
            with open(self.correction_params_path) as f:
                correction_params = json.load(f)
        return {
            "image0": image0,
            "image1": image1,
            "task": task,
            "success": success,
            "assessment": assessment,
            "subtasks": self.ica_dirs,
            "human_correction": human_correction,
            "correction_params": correction_params,  # dict or None
        }


class ReasoningICADir(GenericICADir):
    """
    Loader and utility for the new ICA directory structure.

    Loads:
      - image0.png: initial scene image
      - task.txt: task instruction
      - success.txt: whether or not the task succeeded
      - scene1.json: final scene description
      - task.txt: subtask description
      - diff.txt: diff between scene descriptions (optional)
      - success: boolean field (from success.txt, or metadata.json if present)
    """

    def __init__(self, dir_path):
        super().__init__(dir_path)
        self.image0_path = self._find_file("image0.png")
        self.task_path = self._find_file("task.txt")
        self.success_path = self._find_file("success.txt")
        self.whathappened_path = self._find_file("whathappened.txt")
        self.reasoning_path = self._find_file("reasoning.txt")

    def get_reasoning_tuple(self):
        image0 = Image.open(self.image0_path)
        reasoning = self.read_text(self.reasoning_path)
        whathappened = self.read_text(self.whathappened_path)
        success = self.load_success()
        task = self.read_text(self.task_path)
        return {
            "image0": image0,
            "reasoning": reasoning,
            "whathappened": whathappened,
            "success": success,
            "task": task,
        }
