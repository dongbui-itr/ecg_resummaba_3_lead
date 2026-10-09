"""Dual U-Net: one shared encoder with a VENTRICULAR and an ATRIAL branch, three decoders.

    input (2500, 3) 250 Hz, z-scored per lead
      |
      per-lead stem (same weights on every lead)                      (3, 2500, s)
      |   +-- lead weights: softmax over the leads per 0.2 s ---------+ (also 'channel')
      lead pooling: sum_l w_l F_l || max_l F_l || mean_l F_l          (2500, 3s)
      |
      +-- V branch (QRS, T): max-pool U-Net encoder                   1250 / 250 / 50 / 10
      |     + SSM at 250 (multi-beat context for the beat positions)
      |     -> beat POSITION decoder -> p(beat) (1250,)  --stop-gradient--+
      |                                                                   |
      +-- QRST cancellation: average beat template at the beats ---------+
      |     residual = signal - template (+-100 / +450 ms around R)       (1250, 3) 125 Hz
      |     -> what is left is atrial activity: P waves, F / f waves
      |
      +-- A branch (P, F, f): per-lead conv on the residual, lead pooling, dilated convs at
      |     8 ms, AVERAGE pooling (a 50 uV oscillation survives it, a max-pool keeps QRS)
      |                                                                   1250 / 250 / 50 / 10
      +-- shared neck: both bottoms climb to 250 through both branches' skips, then a
            bidirectional SSM (state space, no attention)                 (250, width)

    decoders
      'beat'    (1250, 4)  none / N / S / V: p(beat) from the V branch, the TYPE from neck +
                           both 8 ms skips (a premature beat with an abnormal or absent P is S)
      'rhythm'  (1250, 5)  SINUS / AFIB (AFL included) / SVT / VT / AVB (2nd or 3rd degree): neck +
                           stop-gradient beat probabilities + an RR autocorrelation FiLM,
                           climbing through both 8 ms skips
      'channel' (5, 4)     per 2 s segment: NOISE / CH1 / CH2 / CH3 - the cleanest lead, or
                           NOISE when the segment cannot be read (the 'noise' head's CLEAN rule)

Why two branches: the QRS is 1-2 mV, a P wave 0.1-0.25 mV and f waves 0.05-0.1 mV. One
encoder that max-pools after its stem keeps the QRS and drops the rest, and a model trained on
rhythm labels then learns R-R regularity only - exactly the errors left on mitdb (232 sinus +
PACs and 200/203 PVCs called AF, 215 sinus tachy called SVT, AVB2 missed). The atrial branch
reads the signal with the ventricular complexes subtracted, at full resolution, so P present /
absent / replaced by F or f waves is something it can see.

Nothing here is a transformer: convolutions, pooling, a diagonal state-space block
(models.layers.ssm_block) and a fixed-lag autocorrelation.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from ..models.layers import RhythmDescriptor, StopGradient, conv_bn_act, ssm_block
from . import config as rc

PKG = 'ecgr_rhythm'

BEAT_HZ = rc.BEAT_STEPS // rc.OUTPUT_SECONDS                 # 125 steps / s
LEAD_WEIGHT_STEPS = 50                                       # lead weights every 0.2 s
QRST_PRE_SECONDS = 0.10                                      # template starts 100 ms before R
QRST_POST_SECONDS = 0.45                                     # ... and ends 450 ms after (T end)
QRST_NMS_SECONDS = 0.20                                      # one beat per 200 ms (300 bpm)
QRST_MIN_PROB = 0.3                                          # weaker peaks build no template
QRS_MASK_SECONDS = 0.06                                      # +-60 ms around R = 'QRS here'


# --- registered layers (their class names are part of the .keras format) -----------------

@keras.saving.register_keras_serializable(package=PKG)
class LeadPool(layers.Layer):
    """[(B, c, T, d) per-lead features, (B, c, K, 1) lead logits] -> (B, T, 3d):
    softmax-over-leads weighted sum || max over leads || mean over leads.

    The logits come at K steps and are repeated to T; the three summaries are symmetric in the
    leads, so whatever follows is invariant to the lead order by construction - the lead
    permutation of training no longer has to teach it."""

    def call(self, inputs):
        f, logits = inputs
        t, k = f.shape[2], logits.shape[2]
        w = tf.nn.softmax(logits, axis=1)
        w = tf.repeat(w, t // k, axis=2)                                   # (B, c, T, 1)
        weighted = tf.reduce_sum(w * f, axis=1)
        return tf.concat([weighted, tf.reduce_max(f, axis=1), tf.reduce_mean(f, axis=1)], -1)

    def compute_output_shape(self, input_shape):
        b, c, t, d = input_shape[0]
        return (b, t, 3 * d)


@keras.saving.register_keras_serializable(package=PKG)
class QRSTCancel(layers.Layer):
    """[(B, 2500, c) signal, (B, 1250, 1) p(beat)] -> [(B, 1250, c) residual, (B, 1250, 1)
    QRS mask], at 125 Hz.

    Average beat subtraction inside the graph: the beats are the local maxima of p(beat)
    (non-maximum suppression over QRST_NMS_SECONDS, at least QRST_MIN_PROB), the template is
    the p-weighted mean of the signal from -QRST_PRE to +QRST_POST seconds around them, and it
    is put back at every beat and subtracted. The template window starts 100 ms before R, so
    the P wave (120-250 ms before R) is NOT part of it and stays in the residual; f waves are
    not phase-locked to the QRS, so averaging removes them from the template and they stay
    too. p(beat) is cut from the gradient: the beat head is trained by its own loss only.
    A window without beats gives a zero template and residual = signal."""

    def __init__(self, pre=QRST_PRE_SECONDS, post=QRST_POST_SECONDS, nms=QRST_NMS_SECONDS,
                 min_prob=QRST_MIN_PROB, mask=QRS_MASK_SECONDS, step_hz=BEAT_HZ, **kw):
        super().__init__(**kw)
        self.pre, self.post, self.nms = float(pre), float(post), float(nms)
        self.min_prob, self.mask, self.step_hz = float(min_prob), float(mask), float(step_hz)

    def call(self, inputs):
        x, p = inputs
        p = tf.stop_gradient(p[..., 0])                                    # (B, T)
        t = p.shape[1]
        stride = x.shape[1] // t
        xd = tf.nn.avg_pool1d(x, stride, stride, 'VALID')                  # (B, T, c)
        pre = int(round(self.pre * self.step_hz))
        post = int(round(self.post * self.step_hz))
        length = pre + post + 1
        k = 2 * int(round(self.nms * self.step_hz)) + 1
        peak = tf.nn.max_pool1d(p[..., None], k, 1, 'SAME')[..., 0]
        w = tf.where((p >= peak) & (p >= self.min_prob), p, 0.0)          # (B, T) spikes

        frames = tf.signal.frame(tf.pad(xd, [[0, 0], [pre, post], [0, 0]]), length, 1,
                                 axis=1)                                   # (B, T, L, c)
        template = tf.einsum('bt,btlc->blc', w, frames) / \
            (tf.reduce_sum(w, axis=1)[:, None, None] + 1e-6)               # (B, L, c)
        # x_hat[t] = sum_s w[s] template[t - s + pre]
        wf = tf.signal.frame(tf.pad(w, [[0, 0], [post, pre]]), length, 1, axis=1)  # (B, T, L)
        x_hat = tf.einsum('btj,bjc->btc', wf, tf.reverse(template, axis=[1]))
        m = 2 * int(round(self.mask * self.step_hz)) + 1
        qrs = tf.nn.max_pool1d(tf.cast(w > 0, x.dtype)[..., None], m, 1, 'SAME')
        return [xd - x_hat, qrs]

    def compute_output_shape(self, input_shape):
        (b, n, c), (_, t, _) = input_shape
        return [(b, t, c), (b, t, 1)]

    def get_config(self):
        return {**super().get_config(), 'pre': self.pre, 'post': self.post, 'nms': self.nms,
                'min_prob': self.min_prob, 'mask': self.mask, 'step_hz': self.step_hz}


@keras.saving.register_keras_serializable(package=PKG)
class BeatCompose(layers.Layer):
    """[(B, T, 1) position logit, (B, T, 3) N/S/V logits] -> (B, T, 4) none / N / S / V that
    sums to 1: p(none) = 1 - sigmoid(pos), p(type) = sigmoid(pos) * softmax(types). The loss
    (objectives.beat_loss) reads exactly these two factors back - the BCE on 1 - p(none) and
    the CE on the renormalised N/S/V - so position and type stay separate heads."""

    def call(self, inputs):
        pos, types = inputs
        p = tf.sigmoid(pos)
        return tf.concat([1.0 - p, p * tf.nn.softmax(types, axis=-1)], axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0][:-1]) + (4,)


@keras.saving.register_keras_serializable(package=PKG)
class FiLM(layers.Layer):
    """[(B, T, d) features, (B, 2d) condition] -> x * (1 + gamma) + beta."""

    def call(self, inputs):
        x, cond = inputs
        gamma, beta = tf.split(cond[:, None, :], 2, axis=-1)
        return x * (1.0 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.saving.register_keras_serializable(package=PKG)
class LeadImage(layers.Layer):
    """(B, T, c) -> (B, c, T, 1): the leads as rows of an image, so a Conv2D with a (1, k)
    kernel applies the same weights to every lead."""

    def call(self, x):
        return tf.transpose(x, [0, 2, 1])[..., None]

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        return (b, c, t, 1)


# --- blocks -------------------------------------------------------------------------------

def _double_conv(x, filters, kernel, name):
    x = conv_bn_act(x, filters, kernel, name=f'{name}_a')
    return conv_bn_act(x, filters, kernel, name=f'{name}_b')


def _lead_conv(x, filters, kernel, n, name):
    """n x (Conv2D (1, k) -> BN -> ReLU) on a (B, c, T, d) lead image: shared over leads."""
    for i in range(n):
        x = layers.Conv2D(filters, (1, kernel), padding='same', use_bias=False,
                          name=f'{name}{i}_conv')(x)
        x = layers.BatchNormalization(name=f'{name}{i}_bn')(x)
        x = layers.Activation('relu', name=f'{name}{i}_relu')(x)
    return x


def _lead_logits(f, name):
    """(B, c, T, d) per-lead features -> (B, c, LEAD_WEIGHT_STEPS, 1) lead logits."""
    t = f.shape[2]
    h = layers.AveragePooling2D((1, t // LEAD_WEIGHT_STEPS), name=f'{name}_avg')(f)
    h = layers.Dense(max(4, f.shape[-1] // 2), activation='relu', name=f'{name}_hidden')(h)
    return layers.Dense(1, name=f'{name}_logit')(h)


def _dilated_block(x, filters, kernel, dilations, name):
    """Residual stack of dilated convs at one resolution: a wide view (P to P, F to F)
    without pooling the small waves away."""
    x = conv_bn_act(x, filters, 1, name=f'{name}_in')
    for d in dilations:
        h = conv_bn_act(x, filters, kernel, dilation_rate=d, name=f'{name}_d{d}')
        x = layers.Add(name=f'{name}_d{d}_res')([x, h])
    return x


# --- the model ----------------------------------------------------------------------------

def build_dual_unet_model(stem=12, v_filters=(32, 48, 64, 96), a_filters=(24, 32, 48, 64),
                          width=64, ssm_blocks=2, state_dim=8, kernel_len=128, beat_width=32,
                          rhythm_width=48, lead_width=16, dropout=0.15,
                          name='dualunet_rhythm'):
    """(2500, 3) -> {'beat': (1250, 4), 'rhythm': (1250, NUM_CLASSES), 'channel': (5, 4)}."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    t1, t2 = rc.BEAT_STEPS, rc.BACKBONE_STEPS                            # 1250, 250
    p1, p2, p3 = n // t1, t1 // t2, 5                                    # 2, 5, 5
    v1f, v2f, v3f, v4f = v_filters
    a1f, a2f, a3f, a4f = a_filters
    seg = rc.NOISE_SEGMENTS

    inp = keras.Input(shape=(n, c), name='input')

    # --- shared per-lead stem and lead pooling ---------------------------------------------
    img = LeadImage(name='lead_image')(inp)                             # (B, c, 2500, 1)
    f0 = _lead_conv(img, stem, 9, 2, 'stem')                            # (B, c, 2500, stem)
    lead_logit = _lead_logits(f0, 'lead_w')                             # (B, c, 50, 1)
    x0 = LeadPool(name='stem_pool')([f0, lead_logit])                   # (B, 2500, 3 stem)

    # --- V branch: QRS / T, max-pool U-Net encoder -----------------------------------------
    v1 = _double_conv(layers.MaxPooling1D(p1, name='v0_pool')(x0), v1f, 7, 'v1')      # 1250
    v2 = _double_conv(layers.MaxPooling1D(p2, name='v1_pool')(v1), v2f, 5, 'v2')      # 250
    v3 = _double_conv(layers.MaxPooling1D(p3, name='v2_pool')(v2), v3f, 5, 'v3')      # 50
    vb = _double_conv(layers.MaxPooling1D(p3, name='v3_pool')(v3), v4f, 3, 'vb')      # 10
    vs = ssm_block(conv_bn_act(v2, width, 1, name='vs_in'), width, state_dim, kernel_len,
                   name='vs')                                           # (250, width)

    # --- beat POSITION decoder (V branch only: the template below depends on it) -----------
    h = layers.UpSampling1D(p3, name='pos3_up')(vb)
    h = _double_conv(layers.Concatenate(name='pos3_skip')([h, v3]), v3f, 5, 'pos3')
    h = layers.UpSampling1D(p3, name='pos2_up')(h)
    h = _double_conv(layers.Concatenate(name='pos2_skip')([h, v2, vs]), v2f, 5, 'pos2')
    h = layers.UpSampling1D(p2, name='pos1_up')(h)
    h = _double_conv(layers.Concatenate(name='pos1_skip')([h, v1]), beat_width, 7, 'pos1')
    pos_logit = layers.Conv1D(1, 1, name='beat_pos')(h)                 # (1250, 1)
    p_pos = layers.Activation('sigmoid', name='beat_pos_prob')(pos_logit)

    # --- QRST cancellation -> A branch: P / F / f ------------------------------------------
    residual, qrs = QRSTCancel(name='qrst_cancel')([inp, p_pos])        # (1250, c), (1250, 1)
    r_img = LeadImage(name='atrial_image')(residual)                    # (B, c, 1250, 1)
    fa = _lead_conv(r_img, stem, 7, 2, 'astem')                         # (B, c, 1250, stem)
    a_logit = _lead_logits(fa, 'alead_w')
    a0 = LeadPool(name='astem_pool')([fa, a_logit])                     # (1250, 3 stem)
    a0 = layers.Concatenate(name='a0_mask')([a0, qrs])
    a1 = _dilated_block(a0, a1f, 7, (1, 2, 4, 8), 'a1')                 # 1250, ~0.75 s view
    a2 = _dilated_block(layers.AveragePooling1D(p2, name='a1_pool')(a1), a2f, 5, (1, 2, 4),
                        'a2')                                           # 250
    a3 = _double_conv(layers.AveragePooling1D(p3, name='a2_pool')(a2), a3f, 5, 'a3')  # 50
    ab = _double_conv(layers.AveragePooling1D(p3, name='a3_pool')(a3), a4f, 3, 'ab')  # 10

    # --- shared neck: both bottoms up to 250, then state-space context ----------------------
    h = layers.Concatenate(name='neck_bottom')([vb, ab])
    h = layers.UpSampling1D(p3, name='neck3_up')(h)
    h = _double_conv(layers.Concatenate(name='neck3_skip')([h, v3, a3]), v3f, 5, 'neck3')
    h = layers.UpSampling1D(p3, name='neck2_up')(h)
    h = _double_conv(layers.Concatenate(name='neck2_skip')([h, v2, a2, vs]), width, 5, 'neck2')
    for i in range(ssm_blocks):
        h = ssm_block(h, width, state_dim, kernel_len, name=f'neck_ssm{i}')
    z = h                                                               # (250, width)

    # --- decoder 1: beat TYPE (N / S / V), then the 'beat' output ---------------------------
    h = layers.UpSampling1D(p2, name='type_up')(z)
    h = _double_conv(layers.Concatenate(name='type_skip')([h, v1, a1]), beat_width, 7, 'type')
    h = layers.Dropout(dropout, name='type_drop')(h)
    type_logit = layers.Conv1D(3, 1, name='beat_type')(h)
    beat = BeatCompose(name='beat')([pos_logit, type_logit])            # (1250, 4)

    # --- decoder 2: rhythm, conditioned on the beats (stop-gradient) ------------------------
    beat_sg = StopGradient(name='beat_sg')(beat)
    pooled_beat = layers.MaxPooling1D(p2, name='beat_to_grid')(beat_sg)  # (250, 4)
    h = layers.Concatenate(name='rhythm_beat_cond')([z, pooled_beat])
    h = conv_bn_act(h, width, 1, name='rhythm_fuse')
    p_sg = StopGradient(name='pos_sg')(p_pos)
    rr = RhythmDescriptor(step_hz=BEAT_HZ, channel=0, name='rr_ac')(p_sg)   # RR autocorrelation
    rr = layers.Dense(width, activation='relu', name='rr_hidden')(rr)
    rr = layers.Dense(2 * width, name='rr_film')(rr)
    h = FiLM(name='rhythm_film')([h, rr])
    h = ssm_block(h, width, state_dim, kernel_len, name='rhythm_ssm')
    h = layers.UpSampling1D(p2, name='rhythm_up')(h)
    h = _double_conv(layers.Concatenate(name='rhythm_skip')([h, v1, a1]), rhythm_width, 7,
                     'rhythm_dec')
    h = layers.Dropout(dropout, name='rhythm_drop')(h)
    rhythm = layers.Conv1D(rc.NUM_CLASSES, 1, activation='softmax', name='rhythm')(h)

    # --- decoder 3: clean channel / NOISE per 2 s segment -----------------------------------
    e = f0
    for i, pool in enumerate((5, 5, 4)):                                # 2500 -> 500 -> 100 -> 25
        e = _lead_conv(e, lead_width, 7, 1, f'chan{i}_')
        e = layers.MaxPooling2D((1, pool), name=f'chan{i}_pool')(e)
    e = layers.AveragePooling2D((1, e.shape[2] // seg), name='chan_seg')(e)   # (B, c, 5, w)
    e = layers.Dense(lead_width, activation='relu', name='chan_embed')(e)
    score = layers.Dense(1, name='chan_score')(e)                       # (B, c, 5, 1)
    score = layers.Permute((2, 1), name='chan_score_t')(
        layers.Reshape((c, seg), name='chan_score_flat')(score))       # (B, 5, c)
    lead_mean = layers.Reshape((seg, lead_width), name='chan_lead_mean')(
        layers.AveragePooling2D((c, 1), name='chan_mean')(e))
    lead_max = layers.Reshape((seg, lead_width), name='chan_lead_max')(
        layers.MaxPooling2D((c, 1), name='chan_max')(e))
    ctx = layers.AveragePooling1D(t2 // seg, name='chan_ctx')(z)        # (B, 5, width)
    s = layers.Concatenate(name='chan_noise_in')([lead_mean, lead_max, ctx])
    s = layers.Dense(lead_width, activation='relu', name='chan_noise_hidden')(s)
    noise_logit = layers.Dense(1, name='chan_noise_logit')(s)           # (B, 5, 1)
    channel = layers.Softmax(name='channel')(
        layers.Concatenate(name='chan_logits')([noise_logit, score]))  # (B, 5, 1 + c)

    return keras.Model(inp, {'beat': beat, 'rhythm': rhythm, 'channel': channel}, name=name)


# Widths fitted to the project's budgets (2m 1,899,421 / 1m 988,773 / 100k 80,189 / 30k 23,361).
SIZES = {
    'rhythm_dual_2m': dict(stem=16, v_filters=(48, 72, 112, 144), a_filters=(40, 64, 80, 112),
                           width=96, beat_width=56, rhythm_width=72, lead_width=24),
    'rhythm_dual_1m': dict(stem=12, v_filters=(36, 56, 72, 96), a_filters=(32, 48, 64, 80),
                           width=72, beat_width=40, rhythm_width=56, lead_width=16),
    'rhythm_dual_100k': dict(stem=6, v_filters=(10, 14, 20, 24), a_filters=(8, 12, 16, 20),
                             width=20, beat_width=12, rhythm_width=16, lead_width=6,
                             ssm_blocks=1),
    'rhythm_dual_30k': dict(stem=4, v_filters=(6, 8, 10, 12), a_filters=(4, 6, 8, 10),
                            width=10, beat_width=6, rhythm_width=8, lead_width=4,
                            ssm_blocks=1, state_dim=4),
}
BUDGETS = {'rhythm_dual_2m': 2_000_000, 'rhythm_dual_1m': 1_000_000,
           'rhythm_dual_100k': 100_000, 'rhythm_dual_30k': 30_000}
