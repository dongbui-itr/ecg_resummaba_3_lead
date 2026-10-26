#!/usr/bin/env python3
"""Checkpoint -> X-CUBE-AI C code (+ the TFLite twins the PC side runs) for the firmware.

    python3 firmware/stm32f746g_disco/export_model.py      # edit the CONFIG block first

No command-line arguments (same rule as evaluate.py): the CONFIG block is the record of what
was exported. The model goes to the board in three parts (ecgr/rhythm/mcu_export.py says
why): net A (X-CUBE-AI), the QRST-cancellation / RR stage (App/ecg_stage.c, hand-written),
net B (X-CUBE-AI). Steps:

  1. CHECKPOINT -> ecga.tflite / ecgb.tflite; A -> stage -> B checked against Keras
  2. stedgeai generate --target stm32f7 for both (legacy ai_ecga_* / ai_ecgb_* C API)
  3. stedgeai validate --mode host for both: the GENERATED C runs on this PC against the
     TFLite file; the export stops if any output's cosine is below MIN_COS. (It was this
     check that found X-CUBE-AI miscompiling the single-graph export.)
  4. Model/model_meta.h: tag, MACC, activation sizes, the stage constants and net B's input
     order (X-CUBE-AI reorders the inputs), Model/model_meta.json for the PC side

Then `make` in this folder builds the firmware.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

# The 30k student (r30k_A_e36): 23 k params, 13.6 M MACC, 114 KB flash, A 352 KB + B 261 KB
# activations (SDRAM). The 1M teacher's weights alone (4 MB) do not fit the 1 MB flash.
CHECKPOINT = '/media/Project/ECG/Model_Dong/ecgr_rhythm/201026_rhythm_dual_30k_A/release/e36.keras'
MODEL_TAG = 'r30k_A_e36'          # shown on the LCD and reported to the PC (<= 23 chars)
STEDGEAI = None                   # None = newest X-CUBE-AI pack under ~/STM32Cube
MAX_DIFF = 1e-3                   # A -> stage -> B (TFLite) vs Keras, per output
MIN_COS = 0.99999                 # generated C vs TFLite (stedgeai validate --mode host)

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL_DIR = os.path.join(HERE, 'Model')
WORK = os.path.join(HERE, 'build', 'stedgeai')


def find_stedgeai():
    if STEDGEAI:
        return os.path.expanduser(STEDGEAI)
    packs = glob.glob(os.path.expanduser(
        '~/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/*/Utilities/linux/stedgeai'))
    if not packs:
        raise SystemExit('stedgeai not found: install X-CUBE-AI or set STEDGEAI')
    version = lambda p: [int(v) for v in p.split('/X-CUBE-AI/')[1].split('/')[0].split('.')]  # noqa: E731
    return max(packs, key=version)


def stedgeai(se, mode, name, tflite):
    out = os.path.join(WORK, f'{name}_{mode}')
    cmd = [se, mode, '--model', tflite, '--target', 'stm32f7', '--name', name,
           '--compression', 'none', '--c-api', 'legacy',
           '--workspace', os.path.join(out, 'ws'), '--output', out]
    if mode == 'validate':
        cmd += ['--mode', 'host', '--batch-size', '3']
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = (r.stdout + r.stderr).replace('\r', '\n')
    with open(os.path.join(WORK, f'{name}_{mode}.log'), 'w') as f:
        f.write(log)
    if r.returncode:
        raise SystemExit(f'stedgeai {mode} {name} failed ({r.returncode}):\n{log[-3000:]}')
    return out, log


def c_info(out, name):
    with open(os.path.join(out, f'{name}_c_info.json')) as f:
        info = json.load(f)
    g = info['graphs'][0]
    mem = info['memory_footprint']
    size = lambda v: sum(x['size_bytes'] for x in v) if isinstance(v, list) else int(v)  # noqa: E731
    return dict(macc=sum(n.get('macc') or 0 for n in g['nodes']),
                weights=size(mem['weights']), activations=size(mem['activations']),
                inputs=[i['name'] for i in g['original_inputs']],
                in_shapes=[i['shape'][1:] for i in g['original_inputs']],
                out_shapes=[o['shape'][1:] for o in g['original_outputs']])


def validated_cos(log):
    """Cosine of every 'X-cross #i' row of a stedgeai validate report."""
    rows = [l.split() for l in log.split('\n') if l.strip().startswith('X-cross #')]
    return [float(r[9]) for r in rows]


def main():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, ROOT)
    import keras
    from ecgr.rhythm import dualunet, model as _model  # noqa: F401 (registers custom layers)
    from ecgr.rhythm import mcu_export as me

    if len(MODEL_TAG) > 23:
        raise SystemExit('MODEL_TAG: at most 23 characters (it travels in a 24-byte field)')
    print(f'checkpoint : {CHECKPOINT}')
    model = keras.models.load_model(CHECKPOINT, compile=False)
    ex = me.export(model)
    diff = me.check(model, ex)
    print(f'split      : A -> stage -> B vs keras, max |diff| {diff}')
    if max(diff.values()) > MAX_DIFF:
        raise SystemExit(f'split pipeline differs from the Keras model by more than {MAX_DIFF}')

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(MODEL_DIR, 'ecg*.[ch]')):
        os.remove(old)
    se = find_stedgeai()
    print(f'stedgeai   : {se}')
    nets = {}
    for key, name in (('a', 'ecga'), ('b', 'ecgb')):
        tfl = os.path.join(MODEL_DIR, f'{name}.tflite')
        with open(tfl, 'wb') as f:
            f.write(ex[key])
        out, _ = stedgeai(se, 'generate', name, tfl)
        for path in glob.glob(os.path.join(out, f'{name}*.[ch]')):
            shutil.copy(path, MODEL_DIR)
        info = c_info(out, name)
        _, log = stedgeai(se, 'validate', name, tfl)
        cos = validated_cos(log)
        print(f'{name:10s} : MACC {info["macc"]:,}, weights {info["weights"]:,} B, activations '
              f'{info["activations"]:,} B, C-vs-TFLite cos {min(cos) if cos else "?"}')
        if not cos or min(cos) < MIN_COS:
            raise SystemExit(f'{name}: generated C does not reproduce the TFLite model '
                             f'(cos {cos}); see {WORK}/{name}_validate.log')
        nets[key] = info

    # C output order = TFLite output order (validate compares them in that order); C input
    # order is X-CUBE-AI's own: map each C input back to the TFLite argument xs_<k>.
    a_shapes = [list(s) for _, s in ex['a_outputs']]
    if [list(s) for s in nets['a']['out_shapes']] != a_shapes:
        raise SystemExit(f'net A output shapes {nets["a"]["out_shapes"]} != {a_shapes}')
    b_order = [int(re.fullmatch(r'xs_(\d+)', n).group(1)) for n in nets['b']['inputs']]
    if sorted(b_order) != list(range(len(me.B_INPUTS))):
        raise SystemExit(f'net B inputs {nets["b"]["inputs"]}')

    # The x86 build of the X-CUBE-AI runtime that `validate --mode host` used: host/ links the
    # firmware's portable C + the generated nets against it (the PC's virtual kit).
    ws = os.path.join(WORK, 'ecga_validate', 'ws', 'inspector_ecga', 'workspace')
    host_ai = os.path.join(HERE, 'build', 'host_ai')
    shutil.rmtree(host_ai, ignore_errors=True)
    shutil.copytree(os.path.join(ws, 'include'), os.path.join(host_ai, 'include'))
    shutil.copytree(os.path.join(ws, 'lib', 'static'), os.path.join(host_ai, 'lib'))
    pack = se.split('/Utilities/')[0]
    with open(os.path.join(MODEL_DIR, 'xcubeai.mk'), 'w') as f:
        f.write('# Written by export_model.py: the X-CUBE-AI pack the C code was generated with\n'
                '# (its runtime library must be the same version).\n'
                f'XCUBEAI ?= {pack}\n')

    k = ex['constants']
    with open(CHECKPOINT, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    b_names = ', '.join(f'ECGB_IN_{me.B_INPUTS[i].upper()}' for i in b_order)
    lines = ['/* Written by export_model.py - do not edit. */', '#ifndef MODEL_META_H',
             '#define MODEL_META_H', '',
             f'#define ECG_MODEL_TAG        "{MODEL_TAG}"',
             f'#define ECG_MODEL_SHA256     "{sha}"',
             f'#define ECG_MODEL_MACC       {nets["a"]["macc"] + nets["b"]["macc"]}u',
             f'#define ECG_MODEL_WEIGHTS    {nets["a"]["weights"] + nets["b"]["weights"]}u',
             f'#define ECG_MODEL_ACT_A      {nets["a"]["activations"]}u',
             f'#define ECG_MODEL_ACT_B      {nets["b"]["activations"]}u', '',
             '/* QRST cancellation / RR autocorrelation stage (App/ecg_stage.c) */',
             f'#define STAGE_PRE            {k["pre"]}      /* template steps before R */',
             f'#define STAGE_POST           {k["post"]}     /* ... and after */',
             f'#define STAGE_NMS            {k["nms"]}      /* beat NMS window, steps */',
             f'#define STAGE_MIN_PROB       {k["min_prob"]:.6f}f',
             f'#define STAGE_MASK           {k["mask"]}     /* QRS mask window, steps */',
             f'#define STAGE_RR_LO          {k["rr_lo"]}    /* first lag */',
             f'#define STAGE_RR_HI          {k["rr_hi"]}    /* one past the last lag */', '',
             '/* net A outputs, in C order (= TFLite order) */']
    lines += [f'#define ECGA_OUT_{n.upper():10s} {i}' for i, (n, _) in enumerate(ex['a_outputs'])]
    lines += ['', '/* net B inputs by meaning */']
    lines += [f'#define ECGB_IN_{n.upper():10s} {i}' for i, n in enumerate(me.B_INPUTS)]
    lines += ['/* C input slot -> ECGB_IN_* it takes (X-CUBE-AI orders the inputs itself) */',
              f'#define ECGB_C_INPUT_ORDER   {{ {b_names} }}', '', '#endif', '']
    with open(os.path.join(MODEL_DIR, 'model_meta.h'), 'w') as f:
        f.write('\n'.join(lines))
    meta = dict(checkpoint=CHECKPOINT, sha256=sha, tag=MODEL_TAG, stedgeai=se,
                nets={k: {kk: v for kk, v in n.items()} for k, n in nets.items()},
                constants=k, a_outputs=ex['a_outputs'], b_inputs=ex['b_inputs'],
                b_c_input_order=b_order, split_vs_keras=diff)
    with open(os.path.join(MODEL_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'written    : {MODEL_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
