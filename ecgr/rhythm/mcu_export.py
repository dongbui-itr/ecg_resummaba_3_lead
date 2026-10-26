"""Dual U-Net -> two float32 TFLite graphs + one hand-written stage, for an STM32 (X-CUBE-AI).

X-CUBE-AI (stedgeai 2.2) compiles the convolutional parts of the model exactly, but not the
two layers built from frames / gathers: QRSTCancel and RhythmDescriptor either fail to import
(5-D GATHER, BOOL tensors, PAD of rank-2 tensors) or, rewritten with tile / reshape tricks,
compile into C that crashes or returns wrong numbers (checked with `stedgeai validate --mode
host`, 2026-10-25). Both are a few hundred thousand multiplies of plain loops, so the model
is cut around them:

    net A   input (2500, 3) -> pos_logit, p_pos (1250, 1) + the encoder tensors net B needs
    stage   QRST cancellation (residual (1250, 3), qrs (1250, 1)) and the RR autocorrelation
            (rr (n_lags,)) from the input and p_pos - written in C on the device,
            `c_stage` below is its reference
    net B   residual, qrs, rr + net A's tensors -> rhythm (1250, 5), beat (1250, 4),
            channel (5, 4)

A -> stage -> B reproduces the Keras model exactly (same layers, same weights). The one graph
rewrite left is DiagSSM1D: its FFT convolution (RFFT / IRFFT, flex ops) becomes the fixed
two-sided FIR filter it is, as one DepthwiseConv with 2L-1 taps (the layer's docstring calls
for this).
"""
import contextlib

import keras
import numpy as np
import tensorflow as tf

from ..models.layers import DiagSSM1D
from . import config as rc

# net A outputs / net B inputs: name -> layer whose output it is, in the order the C code
# copies them (net B's inputs are matched by this order, see `split`).
A_OUTPUTS = {'pos_logit': 'beat_pos', 'p_pos': 'beat_pos_prob', 'v1': 'v1_b_relu',
             'v2': 'v2_b_relu', 'v3': 'v3_b_relu', 'vb': 'vb_b_relu', 'vs': 'vs_out_relu',
             'chan': 'chan_embed'}
B_FROM_A = ('pos_logit', 'v1', 'v2', 'v3', 'vb', 'vs', 'chan')
B_INPUTS = ('residual', 'qrs', 'rr') + B_FROM_A
OUTPUTS = ('rhythm', 'beat', 'channel')


def fir_taps(layer):
    """DiagSSM1D -> (C, 2L-1) cross-correlation taps of its two-sided filter: tap L-1+j
    multiplies x[t + j] (j < 0: the forward kernel, j > 0: the backward one, j = 0: both)."""
    k = layer.kernel_numpy()                                         # (1|2, C, L)
    length = k.shape[2]
    taps = np.zeros((k.shape[1], 2 * length - 1), np.float32)
    taps[:, :length] += k[0][:, ::-1]
    if len(k) > 1:
        taps[:, length - 1:] += k[1]
    return taps


def _ssm_fir_call(self, inputs):
    t, c = inputs.shape[1], inputs.shape[2]
    w = tf.constant(self._mcu_taps.T[None, :, :, None])              # (1, 2L-1, C, 1)
    y = tf.nn.depthwise_conv2d(tf.reshape(inputs, [-1, 1, t, c]), w, [1, 1, 1, 1], 'SAME')
    return tf.reshape(y, [-1, t, c])


@contextlib.contextmanager
def fir_ssm(model):
    """Inside the block, DiagSSM1D layers run as depthwise FIR convolutions."""
    for layer in model._flatten_layers():
        if isinstance(layer, DiagSSM1D):
            layer._mcu_taps = fir_taps(layer)
    saved = DiagSSM1D.call
    DiagSSM1D.call = _ssm_fir_call
    try:
        yield model
    finally:
        DiagSSM1D.call = saved


def split(model):
    """(net A, net B) Keras models cut out of a dual U-Net (see the module docstring)."""
    a = keras.Model(model.input, [model.get_layer(n).output for n in A_OUTPUTS.values()])
    qrst = model.get_layer('qrst_cancel').output
    rr = model.get_layer('rr_ac').output
    feeds = dict(zip(A_OUTPUTS, a.outputs))
    b_inputs = [qrst[0], qrst[1], rr] + [feeds[k] for k in B_FROM_A]
    b = keras.Model(b_inputs, [model.output[k] for k in OUTPUTS])
    return a, b


def stage_constants(model):
    """The integers and thresholds of the hand-written stage, read from the model's own
    layers (with Python's rounding, which is what the layers use: round(12.5) = 12)."""
    q = model.get_layer('qrst_cancel')
    r = model.get_layer('rr_ac')
    if r.channel is None:
        raise NotImplementedError('RhythmDescriptor(channel=None) has no C stage')
    return dict(pre=int(round(q.pre * q.step_hz)), post=int(round(q.post * q.step_hz)),
                nms=2 * int(round(q.nms * q.step_hz)) + 1, min_prob=float(q.min_prob),
                mask=2 * int(round(q.mask * q.step_hz)) + 1, rr_lo=int(r.lo), rr_hi=int(r.hi),
                rr_channel=int(r.channel))


def c_stage(x, p, k):
    """Reference of the device's C stage for ONE window: x (2500, 3) model input, p (1250,)
    p(beat) from net A, k = stage_constants(). Returns residual (1250, 3), qrs (1250,), rr
    (n_lags,), float32, computed in float64 (the C code sums in float32; differences are
    float rounding)."""
    x = np.asarray(x, np.float64)
    p = np.asarray(p, np.float64).reshape(-1)
    t = len(p)
    xd = x.reshape(t, -1, x.shape[1]).mean(axis=1)                       # 2 samples / step
    pre, post = k['pre'], k['post']
    length = pre + post + 1
    half = k['nms'] // 2
    peak = np.array([p[max(0, i - half):i + half + 1].max() for i in range(t)])
    w = np.where((p >= peak) & (p >= k['min_prob']), p, 0.0)
    xp = np.pad(xd, ((pre, post), (0, 0)))
    template = np.stack([w @ xp[l:l + t] for l in range(length)]) / (w.sum() + 1e-6)   # (L, c)
    x_hat = np.zeros_like(xd)
    for s in np.flatnonzero(w):                                            # x_hat[t] += w[s] tpl[t - s + pre]
        a, b = max(0, s - pre), min(t, s - pre + length)
        x_hat[a:b] += w[s] * template[a - s + pre:b - s + pre]
    mh = k['mask'] // 2
    beat = (w > 0).astype(np.float64)
    qrs = np.array([beat[max(0, i - mh):i + mh + 1].max() for i in range(t)])
    e = np.abs(p) if k['rr_channel'] == 0 else None
    e = e - e.mean()
    lo, hi = k['rr_lo'], k['rr_hi']
    span = t - hi
    base = e[:span]
    num = np.array([np.mean(base * e[lag:lag + span]) for lag in range(lo, hi)])
    rr = num / (np.mean(base ** 2) + 1e-6)
    return (xd - x_hat).astype(np.float32), qrs.astype(np.float32), rr.astype(np.float32)


def to_tflite(keras_model, input_shapes, model=None):
    """Batch-1 float32 TFLite bytes of a (sub-)model, variables frozen (BatchNorm folds).
    `model` = the full model whose DiagSSM1D layers to swap for FIR filters (default: the
    sub-model itself)."""
    from tensorflow.python.framework.convert_to_constants import \
        convert_variables_to_constants_v2
    specs = [tf.TensorSpec([1, *s], tf.float32) for s in input_shapes]
    with fir_ssm(model or keras_model):
        @tf.function(input_signature=specs)
        def serve(*xs):
            y = keras_model(list(xs) if len(xs) > 1 else xs[0], training=False)
            return list(y) if isinstance(y, (list, tuple)) else [y]
        frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    return tf.lite.TFLiteConverter.from_concrete_functions([frozen]).convert()


def export(model):
    """{'a': tflite bytes, 'b': tflite bytes, 'constants': stage_constants, 'a_outputs':
    [(name, shape)], 'b_inputs': [(name, shape)]} for a loaded dual U-Net."""
    a, b = split(model)
    a_shapes = [tuple(t.shape[1:]) for t in a.outputs]
    b_shapes = [tuple(t.shape[1:]) for t in b.inputs]
    return dict(a=to_tflite(a, [(rc.SEGMENT_SAMPLES, rc.IN_CHANNELS)], model),
                b=to_tflite(b, b_shapes, model), constants=stage_constants(model),
                a_outputs=list(zip(A_OUTPUTS, a_shapes)), b_inputs=list(zip(B_INPUTS, b_shapes)))


class TFLiteRunner:
    """One TFLite graph, batch 1, inputs / outputs in the order of the source Keras model.
    The converter keeps neither order nor names, so they are matched by the order of the
    signature arguments (inputs) and by `out_names` = the output order of the frozen
    function, recovered from the 'Identity_<i>' tensor names."""

    def __init__(self, tflite_bytes):
        self.it = tf.lite.Interpreter(model_content=tflite_bytes)
        self.it.allocate_tensors()
        ins = self.it.get_input_details()
        self.inputs = [d['index'] for d in sorted(ins, key=lambda d: _arg_order(d['name']))]
        outs = self.it.get_output_details()
        self.outputs = [d['index'] for d in sorted(outs, key=lambda d: _out_order(d['name']))]

    def __call__(self, *xs):
        for idx, x in zip(self.inputs, xs):
            self.it.set_tensor(idx, np.asarray(x, np.float32)[None])
        self.it.invoke()
        return [self.it.get_tensor(i)[0].copy() for i in self.outputs]


def _arg_order(name):
    """'serving_default_xs_3:0' / 'xs_3' -> 3 ('xs' alone = 0)."""
    base = name.split(':')[0]
    tail = base.rsplit('xs', 1)[-1]
    return int(tail[1:]) if tail.startswith('_') and tail[1:].isdigit() else 0


def _out_order(name):
    """'Identity_2' -> 2, 'Identity' -> 0."""
    base = name.split(':')[0]
    return int(base.rsplit('_', 1)[1]) if '_' in base and base.rsplit('_', 1)[1].isdigit() else 0


def run_split(exported, x):
    """Full forward pass of ONE window through net A -> c_stage -> net B with TFLite:
    x (2500, 3) -> {'rhythm', 'beat', 'channel'}. The PC-side reference of the firmware."""
    runners = exported.setdefault('_runners', {})
    if not runners:
        runners['a'] = TFLiteRunner(exported['a'])
        runners['b'] = TFLiteRunner(exported['b'])
    ya = dict(zip(A_OUTPUTS, runners['a'](x)))
    residual, qrs, rr = c_stage(x, ya['p_pos'], exported['constants'])
    feeds = dict(residual=residual, qrs=qrs[:, None], rr=rr, **{k: ya[k] for k in B_FROM_A})
    return dict(zip(OUTPUTS, runners['b'](*[feeds[k] for k in B_INPUTS])))


def check(model, exported, n=2, seed=0):
    """Largest |split TFLite pipeline - Keras| per output on `n` random windows."""
    x = np.random.default_rng(seed).standard_normal((n, rc.SEGMENT_SAMPLES, rc.IN_CHANNELS))
    x = x.astype(np.float32)
    ref = {k: np.asarray(v) for k, v in model(x, training=False).items()}
    worst = dict.fromkeys(OUTPUTS, 0.0)
    for i in range(n):
        got = run_split(exported, x[i])
        for k in OUTPUTS:
            worst[k] = max(worst[k], float(np.abs(got[k] - ref[k][i]).max()))
    return worst
