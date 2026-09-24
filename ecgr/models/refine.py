"""Temporal refinement head: a second stage that re-reads the predicted beat train.

The base model labels every 20 ms step from morphology plus whatever inter-beat context its
state-space path carries implicitly. The S/N decision, though, is mostly a statement about
TIMING - a supraventricular ectopic arrives early against the rhythm around it - and that
evidence lives in the sequence of beats the base model has just predicted, not in any one
step's features. This head is a small bidirectional state-space network run over exactly
that: the base features concatenated with the base's own per-step class distribution. With a
512-step kernel each way it sees ~10 s of predicted beats on either side of a step, i.e. the
10-20 R-R intervals a cardiologist reads prematurity against.

Two properties are built in rather than hoped for:

  * **The background probability is preserved exactly.** The head emits a correction over the
    three BEAT classes only and the result is composed as
    `[p_None, (1 - p_None) * softmax(beat_logits + delta)]`, so it redistributes mass among
    N / V / S and cannot turn a detection into background or vice versa by itself. QRS
    detection therefore stays the base model's - the one thing this stage must not break.
  * **It starts as the identity.** The correction's last layer is zero-initialised, so at
    epoch 0 the refined model IS the base model, to the bit. Whatever checkpoint selection
    does afterwards (training/refine.py), "no change" is always among the candidates, which
    is what makes a no-regression rule satisfiable at all.

The base model is frozen while the head trains: its metrics are the floor, and a frozen base
cannot move off it.
"""
import keras
import tensorflow as tf
from keras import layers

from .. import config
from .layers import BN_MOMENTUM, PKG, ssm_block

HEAD_INPUT = 'head_drop'          # the base layer whose INPUT is the pre-softmax feature map


@keras.saving.register_keras_serializable(package=PKG)
class BeatClassRefine(layers.Layer):
    """[p1 (B,T,4), delta] -> refined (B,T,4). Two modes, both leave p_None untouched.

    mode='beats' (delta has 3 channels): redistribute among N / V / S,
        q = softmax(log p1[N,V,S] + delta);  out = [p_None, (1 - p_None) * q]
    mode='s_only' (delta has 1 channel): redistribute between N and S ONLY, p_V kept too,
        m = p_N + p_S;  s = sigmoid(logit(p_S / m) + delta);  out = [p_None, m(1-s), p_V, m s]

    's_only' exists because of what 'beats' did on portal-eval: on both the 1m and the 2m
    base every trained epoch lowered V_+P (the head learned to move N mass into V as well as
    into S). The S/N boundary is the target; V is not, so the mode that cannot touch V is the
    one whose no-regression selection has a chance. Rows sum to one in both modes.
    """

    def __init__(self, epsilon=1e-7, mode='beats', **kw):
        super().__init__(**kw)
        self.epsilon = float(epsilon)
        if mode not in ('beats', 's_only'):
            raise ValueError(f"mode must be 'beats' or 's_only', got {mode!r}")
        self.mode = mode

    def call(self, inputs):
        p1, delta = inputs
        p_none = p1[..., :1]
        if self.mode == 'beats':
            beat_logits = tf.math.log(p1[..., 1:] + self.epsilon) + delta
            q = tf.nn.softmax(beat_logits, axis=-1)
            return tf.concat([p_none, (1.0 - p_none) * q], axis=-1)
        p_n, p_v, p_s = p1[..., 1:2], p1[..., 2:3], p1[..., 3:4]
        mass = p_n + p_s
        share = p_s / (mass + self.epsilon)
        logit = tf.math.log(share + self.epsilon) - tf.math.log(1.0 - share + self.epsilon)
        s = tf.sigmoid(logit + delta[..., :1])
        return tf.concat([p_none, mass * (1.0 - s), p_v, mass * s], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon, 'mode': self.mode}


@keras.saving.register_keras_serializable(package=PKG)
class LogProb(layers.Layer):
    """log(p + eps). A registered layer rather than a Lambda: Keras refuses to deserialize a
    Lambda holding a Python lambda unless the loader passes safe_mode=False, and every eval
    stage loads checkpoints with a bare load_model(..., compile=False)."""

    def __init__(self, epsilon=1e-7, **kw):
        super().__init__(**kw)
        self.epsilon = float(epsilon)

    def call(self, p):
        return tf.math.log(p + self.epsilon)

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon}


def attach_refinement(base, width=48, blocks=2, state_dim=8, kernel_len=512, dropout=0.1,
                      freeze_base=True, mode='beats', name=None):
    """Wrap a trained base model with the refinement head; returns the composed keras.Model.

    `base` is a model built by models.build (or loaded from one of its checkpoints). Its
    graph is reused, not copied: the head reads the tensor feeding `head_drop` and the base's
    softmax output through a sub-model over the same layers, so the base's weights are shared
    and, with freeze_base, untouched by training.

    The refined model keeps the base's output layout: its first output is the refined beat
    softmax under the name `beat_cls`, and a base that also emits `lead_quality` passes it
    through unchanged as the second output - the head has nothing to say about lead quality.
    """
    feat = base.get_layer(HEAD_INPUT).input
    extra = list(base.outputs[1:])                      # lead_quality, when the base has it
    probe = keras.Model(base.inputs, [feat, base.outputs[0]] + extra, name=f'{base.name}_probe')
    probe.trainable = not freeze_base

    inp = keras.Input(shape=base.input_shape[1:], name='input')
    probed = probe(inp)
    feat, p1, passthrough = probed[0], probed[1], probed[2:]

    # log-probabilities as well as probabilities: the head's first job is to find beats in
    # p1, and a beat at p=0.9 vs 0.99 is a large difference in log space and a small one in
    # probability space - both readings are cheap to give it.
    h = layers.Concatenate(name='refine_in')([feat, p1, LogProb(name='refine_logp')(p1)])
    h = layers.Conv1D(width, 1, use_bias=False, name='refine_proj')(h)
    h = layers.BatchNormalization(momentum=BN_MOMENTUM, name='refine_proj_bn')(h)
    h = layers.Activation('silu', name='refine_proj_act')(h)
    for i in range(blocks):
        h = ssm_block(h, width, state_dim, kernel_len, name=f'refine_ssm{i}')
    h = layers.Dropout(dropout, name='refine_drop')(h)
    # zeros: the head starts as the identity (see module docstring)
    n_out = 1 if mode == 's_only' else config.NUM_CLASSES - 1
    delta = layers.Conv1D(n_out, 1, kernel_initializer='zeros', bias_initializer='zeros',
                          name='refine_delta')(h)
    out = BeatClassRefine(mode=mode, name='refine_compose')([p1, delta])
    # Named like the base's beat output so the same dataset dict and loss dict fit both.
    out = layers.Identity(name='beat_cls')(out)
    outputs = [out] + [layers.Identity(name='lead_quality')(q) for q in passthrough]
    return keras.Model(inp, outputs if len(outputs) > 1 else out,
                       name=name or f'{base.name}_refined')


def head_parameters(model):
    """Trainable parameter count of the refinement head alone."""
    return sum(int(tf.size(w)) for w in model.trainable_weights)
