#!/usr/bin/env python3
"""One entry point for the released rhythm checkpoint: check -> evaluate -> view.

    python3 run.py          # edit the CONFIG block first

No command-line arguments (same rule as evaluate.py and webservice/app.py): the CONFIG block
below is the record of what was run. It only chains the existing tools - every setting not
overridden here keeps the value in that tool's own CONFIG block:

    'test'      pytest tests/test_evaluate.py: the checkpoint folder and its hash, the decode
                files, the evaluate.py -> ecgr.rhythm.config hand-over, the viewer API
    'evaluate'  evaluate.py with EVALUATE overrides: EC57 + rhythm_eval -> eval_results/...
    'viewer'    webservice/app.py with VIEWER overrides, pointed at the folder evaluate.py
                writes to; runs until Ctrl+C, so it is always the last step

A failing step stops the chain (a failed test does not spend 40 minutes of GPU).
"""
import os
import subprocess
import sys
import time

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

STEPS = ['test', 'evaluate', 'viewer']          # any subset, in this order

# evaluate.py CONFIG overrides ({} = its own defaults: the official full run). Examples:
#   {'DATABASES': ['mitdb'], 'SCORE_RHYTHM_EVAL': False, 'MAX_RECORDS': 2}     smoke test
#   {'TAG': 'ec57_ov5_official', 'DECODE_ONLY': True}                          re-score stored npz
#   {'DECODE': 'checkpoints/dualunet_rhythm_1m_e26/decode/overlap1_af_context.json',
#    'PREDICT_OVERLAP_SECONDS': 1}                                             an experiment
#   {'GPU_MEMORY_MB': 2000}                                                    card shared with training
EVALUATE = {}

# webservice/app.py CONFIG overrides ({} = 127.0.0.1:8765). EVAL_ROOT follows evaluate.py's
# WORK_DIR / RUN_TAG unless set here.
VIEWER = {}

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
KNOWN_STEPS = ('test', 'evaluate', 'viewer')


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def apply_overrides(module, overrides, label):
    for key, value in overrides.items():
        if not hasattr(module, key) or not key.isupper():
            raise SystemExit(f"{label}: unknown CONFIG name {key!r}")
        setattr(module, key, value)


def step_test():
    cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
           os.path.join(HERE, 'tests', 'test_evaluate.py')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3')
    return subprocess.run(cmd, cwd=HERE, env=env).returncode


def step_evaluate():
    import evaluate
    apply_overrides(evaluate, EVALUATE, 'EVALUATE')
    return evaluate.main()


def viewer_eval_root():
    """The folder evaluate.py writes its runs to, with this run's overrides."""
    import evaluate
    work = EVALUATE.get('WORK_DIR', evaluate.WORK_DIR)
    tag = EVALUATE.get('RUN_TAG', evaluate.RUN_TAG)
    return os.path.join(evaluate.resolve(work), tag, 'ec57')


def step_viewer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'webservice_app', os.path.join(HERE, 'webservice', 'app.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    apply_overrides(app, dict({'EVAL_ROOT': viewer_eval_root()}, **VIEWER), 'VIEWER')
    os.makedirs(app.project_path(app.EVAL_ROOT), exist_ok=True)
    app.main()
    return 0


def main():
    unknown = [s for s in STEPS if s not in KNOWN_STEPS]
    if unknown:
        raise SystemExit(f"unknown STEPS {unknown}; known: {list(KNOWN_STEPS)}")
    if 'viewer' in STEPS and STEPS[-1] != 'viewer':
        raise SystemExit("'viewer' runs until Ctrl+C - put it last in STEPS")
    runners = dict(test=step_test, evaluate=step_evaluate, viewer=step_viewer)
    for name in STEPS:
        banner(f"run.py: {name}")
        t0 = time.time()
        code = runners[name]()
        if code:
            print(f"\nrun.py: step '{name}' failed (exit {code}) - stopping", file=sys.stderr)
            return code
        print(f"run.py: {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
