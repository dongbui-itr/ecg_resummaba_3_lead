"""Finding the checkpoint an evaluation should score."""
import glob
import os
import re

from . import config

_F1_IN_NAME = re.compile(r'_f1_([0-9.]+?)_epoch')


def f1_of(path):
    """F1 embedded in a checkpoint name (best_model_f1_<f1>_epoch_<n>.keras), -1 if absent."""
    m = _F1_IN_NAME.search(os.path.basename(path))
    return float(m.group(1)) if m else -1.0


def best_checkpoint(model_name, run_dir=None):
    """Highest-F1 checkpoint of `model_name`, picked by the F1 in the file name.

    By F1 and not by mtime: a run writes a checkpoint every time the metric improves, so the
    newest file is the best one only if training never restarted - and reruns are normal
    here. The F1 in a name is only comparable INSIDE one run (each run scores its own eval
    split), which is why the search never leaves `run_dir`.
    """
    root = run_dir or config.RUN_DIR
    found = glob.glob(os.path.join(root, 'checkpoints', model_name, 'BEST_F1', '*.keras'))
    if not found:
        fallback = os.path.join(root, 'checkpoints', model_name, 'best_model.keras')
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            f"no checkpoint for {model_name} under {root}/checkpoints/{model_name}/ - "
            f"train it first")
    return max(found, key=lambda p: (f1_of(p), os.path.getmtime(p)))
