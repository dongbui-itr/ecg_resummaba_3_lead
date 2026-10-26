"""Three-U-Net encoder + CNN/CfC decoder: `rhythm_u3cfc_*`.

    input (2500, 3) 250 Hz, z-scored per lead
      |
      shared per-lead stem + lead pooling (softmax-weighted || max || mean over leads)
      |
      +-- U1  QRS MORPHOLOGY   raw signal, MAX-pool, kernels 7/5/5/3     1250/250/50/10 -> 1250
      |        a max keeps the tallest thing in a window - the QRS; short kernels (40-56 ms
      |        at 125 Hz) resolve its shape: width, notch, polarity, the slurred V upstroke.
      |        Its top also gives a QRS position logit -> p(QRS) -------------------------+
      |                                                                                    |
      +-- QRST cancellation (average beat template at the QRS, parameter-free) <-----------+
      |
      +-- U2  P / T SLOW WAVES  raw signal at 125 Hz with the QRS BLANKED, AVERAGE-pool,
      |        wide kernels 11/9/7/5 (88 ms at 125 Hz up to 1.4 s at 5 Hz): the 0.5-10 Hz,
      |        80-300 ms waves - P before the QRS, ST/T after it - without the QRS dominating
      |        every filter. A P wave is 0.1-0.25 mV next to a 1-2 mV QRS.
      |
      +-- U3  ATRIAL / F-f WAVES  QRST-cancelled residual, AVERAGE-pool, dilated convs
      |        (1/2/4/8 at 8 ms: a 4-9 Hz flutter or fibrillation wave, period 110-250 ms,
      |        is seen over several cycles without pooling it away). A 50 uV f wave survives
      |        an average; a max-pool would replace it with the nearest spike.
      |
      ENCODER OUTPUT z (1250, x) = [fused U1|U2|U3 features | QRS logit | per-lead block]
          the per-lead block keeps the lead identity the lead pooling removed, so the
          channel decoder can still say WHICH lead is cleanest.

    decoders - CNN and CfC only (every projection is a 1x1 convolution)
      'beat'    (1250, 4)  none / N / S / V: CfC over z at 8 ms; the position logit is the
                           encoder's QRS logit plus a decoder correction (zero-initialised,
                           so training starts from U1's detector)
      'rhythm'  (20, 5)    SINUS / AFIB / SVT / VT / AVB per 0.5 s: CfC at 40 ms over z and
                           the stop-gradient beat probabilities (the R-R pattern), then
                           x2 / 25 to the 20-step grid (1250 -> 250 -> 500 -> 20, all integer)
      'channel' (5, 4)     NOISE / CH1 / CH2 / CH3 per 2 s: per-lead convolutions on the
                           per-lead block, shared over leads
      'noise'   (5, 2)     CLEAN / NOISE per 2 s: CfC over the quality features

Same four outputs, same shapes, same names as `rhythm_dualcfc_*`: the training pipeline, the
rhythm EC57 path and scripts/dualcfc_bxb.py take it unchanged.

Parameter budget is checked by layer-name prefix: every encoder layer is named enc_*, every
decoder layer dec_* or one of the four output names. `param_split` refuses a layer that is
neither, so a new layer cannot slip into the wrong side of the count.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from ..models.layers import StopGradient, cfc_block, conv_bn_act
from . import config as rc
from .dualunet import (BeatCompose, LeadImage, LeadPool, QRSTCancel, _dilated_block,
                       _double_conv, _lead_conv, _lead_logits)

PKG = 'ecgr_rhythm'
OUTPUTS = ('beat', 'rhythm', 'channel', 'noise')


@keras.saving.register_keras_serializable(package=PKG)
class BlankQRS(layers.Layer):
    """[(B, T, c) signal, (B, T, 1) QRS mask] -> signal * (1 - mask): the QRS region set to
    zero so the slow-wave U-Net sees P, ST and T without the QRS in every filter."""

    def call(self, inputs):
        x, m = inputs
        return x * (1.0 - m)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.saving.register_keras_serializable(package=PKG)
class TemplateResidual(layers.Layer):
    """[(B, T, c) signal at 125 Hz, (B, T, c) QRST-cancelled residual] ->
    (B, T, 2c): per lead, log10 of residual power over template power, then log10 of the
    lead's power - both smoothed over `win` steps (1 s at 125 Hz).

    The signal-quality feature the noise decoder reads, and on purpose the ONLY morphology-free
    thing it reads. The template is the average beat of the window (QRSTCancel); what it does
    not explain is noise - plus P and f waves, which sit 20-30 dB under the QRS. Measured on
    nstdb with no training: noisy blocks of the 0 / -6 dB records +6.6 to +8.8 dB, clean
    blocks -7 to -3 dB, AF records -16 to -2 dB. A head trained on the generator's own
    noise learned the generator (p 0.98 on synthetic noise, 0.18 on real noise at the same
    RMS); this ratio does not care what the noise looks like, only how much of the strip the
    beats fail to explain. The second half tells a flat lead (zero fill) from a clean one."""

    def __init__(self, win=125, **kw):
        super().__init__(**kw)
        self.win = int(win)

    def call(self, inputs):
        x, r = inputs
        t = x - r                                                       # the template track
        def power(v):
            return tf.nn.avg_pool1d(tf.square(v), self.win, 1, 'SAME')
        eps = 1e-4
        ratio = (tf.math.log(power(r) + eps) - tf.math.log(power(t) + eps)) / np.log(10.0)
        level = tf.math.log(power(x) + eps) / np.log(10.0)
        return tf.concat([ratio, level], axis=-1)

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape[0]
        return (b, t, 2 * c)

    def get_config(self):
        return {**super().get_config(), 'win': self.win}


@keras.saving.register_keras_serializable(package=PKG)
class TemplateResidualTP(layers.Layer):
    """[(B, T, c) signal at 125 Hz, (B, T, c) residual, (B, T, 1) QRS mask] -> (B, T, 3c + 1):
    per lead the full residual/template ratio (as TemplateResidual), the residual power in
    the TP stretches over the template power at the QRS, the lead power - all log10 - and
    the share of the strip that is TP at all.

    Why three, measured with no training (2026-10-23): the full ratio flags real noise
    (nstdb 0 dB blocks +6.8..+8.8 dB) but ALSO clean ECG with PVCs, because one average
    template cannot fit two beat shapes (mitdb 200 +3.5 dB median) - gating on it cut mitdb
    V Se 97.0 -> 68.3. Between beats (outside -150..+450 ms of every detected QRS) a PVC
    leaves nothing: mitdb 119/200/203/208/233 TP ratio -22..-38 dB. Heavy noise, though,
    makes the detector fire everywhere, the TP stretches vanish and the TP ratio reads low
    (nstdb 0 dB -19..-23) - which is what the coverage channel says. Each is
    morphology-free; together they separate noise from ectopy."""

    def __init__(self, win=312, pre=19, post=56, **kw):
        super().__init__(**kw)
        self.win, self.pre, self.post = int(win), int(pre), int(post)

    def call(self, inputs):
        x, r, q = inputs
        t = x - r
        # QRST mask: a sample is inside if a QRS sits within [t - post, t + pre]
        m = tf.nn.max_pool1d(tf.pad(q, [[0, 0], [self.post, self.pre], [0, 0]]),
                             self.pre + self.post + 1, 1, 'VALID')
        tp = 1.0 - m                                                      # (B, T, 1)

        def sm(v):
            return tf.nn.avg_pool1d(v, self.win, 1, 'SAME')
        eps, lg = 1e-4, (lambda v: tf.math.log(v) / np.log(10.0))
        full = lg(sm(tf.square(r)) + eps) - lg(sm(tf.square(t)) + eps)
        r_tp = sm(tp * tf.square(r)) / (sm(tp) + 1e-3)
        ref = sm(q * tf.square(t)) / (sm(q) + 1e-3)
        tp_ratio = lg(r_tp + eps) - lg(ref + eps)
        level = lg(sm(tf.square(x)) + eps)
        return tf.concat([full, tp_ratio, level, sm(tp)], axis=-1)

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape[0]
        return (b, t, 3 * c + 1)

    def get_config(self):
        return {**super().get_config(), 'win': self.win, 'pre': self.pre, 'post': self.post}


@keras.saving.register_keras_serializable(package=PKG)
class ChannelSlice(layers.Layer):
    """(B, T, d) -> (B, T, stop - start): channels [start, stop) of the encoder output."""

    def __init__(self, start, stop, **kw):
        super().__init__(**kw)
        self.start, self.stop = int(start), int(stop)

    def call(self, x):
        return x[..., self.start:self.stop]

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.stop - self.start,)

    def get_config(self):
        return {**super().get_config(), 'start': self.start, 'stop': self.stop}


def _unet(x0, filters, kernels, pool, name, dilations=None):
    """Four-level 1-D U-Net at 1250 / 250 / 50 / 10 steps and back up to 1250.

    pool       'max' or 'avg' - the pooling rule decides what survives a downsampling
    kernels    one kernel per level, top to bottom
    dilations  optional per-level dilation tuples (top two levels): residual dilated stacks
               instead of plain double convolutions"""
    f1, f2, f3, f4 = filters
    k1, k2, k3, k4 = kernels
    down = layers.MaxPooling1D if pool == 'max' else layers.AveragePooling1D

    def level(x, f, k, i):
        if dilations and i < len(dilations) and dilations[i]:
            return _dilated_block(x, f, k, dilations[i], f'{name}_l{i}')
        return _double_conv(x, f, k, f'{name}_l{i}')

    e1 = level(x0, f1, k1, 0)                                            # 1250
    e2 = level(down(5, name=f'{name}_p1')(e1), f2, k2, 1)                # 250
    e3 = level(down(5, name=f'{name}_p2')(e2), f3, k3, 2)                # 50
    eb = level(down(5, name=f'{name}_p3')(e3), f4, k4, 3)                # 10
    h = layers.UpSampling1D(5, name=f'{name}_u3')(eb)
    h = _double_conv(layers.Concatenate(name=f'{name}_s3')([h, e3]), f3, k3, f'{name}_d3')
    h = layers.UpSampling1D(5, name=f'{name}_u2')(h)
    h = _double_conv(layers.Concatenate(name=f'{name}_s2')([h, e2]), f2, k2, f'{name}_d2')
    h = layers.UpSampling1D(5, name=f'{name}_u1')(h)
    return _double_conv(layers.Concatenate(name=f'{name}_s1')([h, e1]), f1, k1, f'{name}_d1')


def _stem(x, stem, kernel, name):
    """Per-lead conv (same weights on every lead) + lead pooling -> (T, 3 stem)."""
    f = _lead_conv(LeadImage(name=f'{name}_image')(x), stem, kernel, 2, f'{name}_')
    return f, LeadPool(name=f'{name}_pool')([f, _lead_logits(f, f'{name}_w')])


def build_u3cfc_model(stem=12, q_filters=(40, 56, 72, 96), t_filters=(24, 32, 48, 64),
                      a_filters=(24, 32, 48, 64), enc_width=96, lead_width=8,
                      beat_width=64, beat_cfc=2, rhythm_width=56, rhythm_cfc=2,
                      quality_width=16, state_dim=1, dropout=0.15, sqi=False, noise_width=16,
                      name='u3cfc_rhythm'):
    """(2500, 3) -> {'beat': (1250, 4), 'rhythm': (20, 5), 'channel': (5, 4),
    'noise': (5, 2)}.

    sqi=True appends the TemplateResidual block to the encoder output and gives 'noise' (and
    the NOISE column of 'channel') a decoder of its own that reads ONLY that block and the
    stop-gradient beat positions - not the morphology features, which let the original head
    learn the noise generator instead of 'the beats do not explain this strip'."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    t1 = rc.BEAT_STEPS                                                   # 1250
    seg, rsteps = rc.NOISE_SEGMENTS, rc.RHYTHM_STEPS                     # 5, 20

    inp = keras.Input(shape=(n, c), name='input')

    # =========================== ENCODER (enc_*) ===========================================
    f0, x0 = _stem(inp, stem, 9, 'enc_stem')                            # (2500, 3 stem)

    # U1 - QRS morphology: max-pool, short kernels
    q = _unet(layers.MaxPooling1D(2, name='enc_q_p0')(x0), q_filters, (7, 5, 5, 3), 'max',
              'enc_q')                                                   # (1250, q1)
    pos_enc = layers.Conv1D(1, 1, name='enc_qrs_pos')(q)                 # (1250, 1) logit
    p_qrs = layers.Activation('sigmoid', name='enc_qrs_prob')(pos_enc)
    residual, qrs = QRSTCancel(name='enc_qrst_cancel')([inp, p_qrs])     # (1250, c), (1250, 1)

    # U2 - P / T slow waves: raw at 125 Hz, QRS blanked, wide kernels, average-pool
    raw = layers.AveragePooling1D(2, name='enc_t_raw')(inp)              # (1250, c)
    _ft, t0 = _stem(BlankQRS(name='enc_t_blank')([raw, qrs]), stem, 15, 'enc_tstem')
    t = _unet(layers.Concatenate(name='enc_t_in')([t0, qrs]), t_filters, (11, 9, 7, 5),
              'avg', 'enc_t')                                            # (1250, t1)

    # U3 - atrial activity: QRST-cancelled residual, dilated convs, average-pool
    _fa, a0 = _stem(residual, stem, 7, 'enc_astem')
    a = _unet(layers.Concatenate(name='enc_a_in')([a0, qrs]), a_filters, (7, 5, 5, 3), 'avg',
              'enc_a', dilations=((1, 2, 4, 8), (1, 2, 4)))              # (1250, a1)

    # fusion -> z_main, plus the per-lead block for the channel decoder
    h = conv_bn_act(layers.Concatenate(name='enc_fuse_in')([q, t, a]), enc_width, 1,
                    name='enc_fuse1')
    z_main = conv_bn_act(h, enc_width, 5, name='enc_fuse2')             # (1250, enc_width)
    e = _lead_conv(f0, lead_width, 7, 1, 'enc_lead')                     # (B, c, 2500, L)
    e = layers.MaxPooling2D((1, n // t1), name='enc_lead_pool')(e)       # (B, c, 1250, L)
    e = layers.Permute((2, 1, 3), name='enc_lead_t')(e)                  # (B, 1250, c, L)
    lead_block = layers.Reshape((t1, c * lead_width), name='enc_lead_flat')(e)
    parts = [z_main, pos_enc, lead_block]
    if sqi:
        if sqi >= 2:
            parts.append(TemplateResidualTP(name='enc_sqi2')([raw, residual, qrs]))  # (1250, 3c+1)
        else:
            parts.append(TemplateResidual(name='enc_sqi')([raw, residual]))   # (1250, 2c)
    z = layers.Concatenate(name='enc_output')(parts)
    w0, w1 = enc_width, enc_width + 1                                    # slice points
    w2 = w1 + c * lead_width

    # =========================== DECODER (dec_* + outputs) =================================
    zm = ChannelSlice(0, w0, name='dec_z_main')(z)
    zpos = ChannelSlice(w0, w1, name='dec_z_pos')(z)
    zl = ChannelSlice(w1, w2, name='dec_z_lead')(z)

    # beat: CfC over the strip at 8 ms, then position correction + type
    h = conv_bn_act(zm, beat_width, 1, name='dec_beat_in')
    for i in range(beat_cfc):
        h = cfc_block(h, beat_width, state_dim, 256, name=f'dec_beat_cfc{i}')
    h = layers.Dropout(dropout, name='dec_beat_drop')(
        _double_conv(h, beat_width, 7, 'dec_beat_conv'))
    delta = layers.Conv1D(1, 1, kernel_initializer='zeros', name='dec_beat_pos')(h)
    pos = layers.Add(name='dec_beat_pos_sum')([zpos, delta])
    types = layers.Conv1D(3, 1, name='dec_beat_type')(h)
    beat = BeatCompose(name='beat')([pos, types])                       # (1250, 4)

    # rhythm: z and the (stop-gradient) beat probabilities at 40 ms, CfC, then to 0.5 s
    r = layers.AveragePooling1D(5, name='dec_rhythm_z')(
        conv_bn_act(zm, rhythm_width, 1, name='dec_rhythm_in'))         # (250, rw)
    b = layers.MaxPooling1D(5, name='dec_rhythm_beat')(StopGradient(name='dec_beat_sg')(beat))
    r = conv_bn_act(layers.Concatenate(name='dec_rhythm_cat')([r, b]), rhythm_width, 3,
                    name='dec_rhythm_fuse')
    for i in range(rhythm_cfc):
        # horizon 256 >= 250 steps: the scan is exact over the whole window
        r = cfc_block(r, rhythm_width, state_dim, 256, name=f'dec_rhythm_cfc{i}')
    r = layers.UpSampling1D(2, name='dec_rhythm_up')(r)                  # (500, rw)
    r = layers.AveragePooling1D(500 // rsteps, name='dec_rhythm_grid')(r)   # (20, rw)
    r = layers.Dropout(dropout, name='dec_rhythm_drop')(
        _double_conv(r, rhythm_width, 3, 'dec_rhythm_dec'))
    rhythm = layers.Conv1D(rc.NUM_CLASSES, 1, activation='softmax', name='rhythm')(r)

    # quality: per-lead image from the per-lead block, shared over leads
    li = layers.Permute((2, 1, 3), name='dec_q_t')(
        layers.Reshape((t1, c, lead_width), name='dec_q_img')(zl))      # (B, c, 1250, L)
    for i, pool in enumerate((5, 5)):                                   # 1250 -> 250 -> 50
        li = _lead_conv(li, quality_width, 7, 1, f'dec_q{i}_')
        li = layers.MaxPooling2D((1, pool), name=f'dec_q{i}_pool')(li)  # (B, c, 50, qw)
    score = layers.Conv2D(1, 1, name='dec_q_score')(
        layers.AveragePooling2D((1, li.shape[2] // seg), name='dec_q_seg')(li))  # (B, c, 5, 1)
    score = layers.Permute((2, 1), name='dec_q_score_t')(
        layers.Reshape((c, seg), name='dec_q_score_flat')(score))       # (B, 5, c)
    if sqi:
        # noise from the template-residual block and the beat positions ONLY
        nq = 3 * c + 1 if sqi >= 2 else 2 * c
        pfx = {1: 'dec_n', 3: 'dec_k'}[int(sqi)]   # each version a fresh path
        zq = ChannelSlice(w2, w2 + nq, name=f'{pfx}_z_sqi' if sqi >= 2 else 'dec_z_sqi')(z)
        if sqi == 3:
            # the beat TYPES too: a residual that V beats explain is ectopy, not noise
            p_sg = StopGradient(name=f'{pfx}_beat_sg')(beat)                    # (1250, 4)
        else:
            p_sg = StopGradient(name=f'{pfx}_pos_sg')(
                layers.Activation('sigmoid', name=f'{pfx}_pos_prob')(pos))       # (1250, 1)
        s = layers.Concatenate(name=f'{pfx}_in')([zq, p_sg])
        s = layers.AveragePooling1D(t1 // 50, name=f'{pfx}_pool')(s)            # (50, 2c + 1)
        s = conv_bn_act(s, noise_width, 3, name=f'{pfx}_conv')
        s = cfc_block(s, noise_width, state_dim, 64, name=f'{pfx}_cfc')        # exact over 50
        s = layers.AveragePooling1D(s.shape[1] // seg, name=f'{pfx}_segpool')(s)  # (5, qw)
        s = conv_bn_act(s, noise_width, 1, name=f'{pfx}_hidden')
        noise = layers.Conv1D(len(rc.NOISE_CLASSES), 1, activation='softmax', name='noise')(s)
        noise_logit = layers.Conv1D(1, 1, name=f'{pfx}_noise_logit')(s)
    else:
        s = layers.Concatenate(name='dec_q_cat')([
            layers.Reshape((li.shape[2], quality_width), name='dec_q_mean_flat')(
                layers.AveragePooling2D((c, 1), name='dec_q_mean')(li)),
            layers.Reshape((li.shape[2], quality_width), name='dec_q_max_flat')(
                layers.MaxPooling2D((c, 1), name='dec_q_max')(li)),
            layers.AveragePooling1D(t1 // li.shape[2], name='dec_q_ctx')(zm)])  # (50, .)
        s = conv_bn_act(s, quality_width, 1, name='dec_q_fuse')
        # horizon 64 >= 50 steps: exact over the window
        s = cfc_block(s, quality_width, state_dim, 64, name='dec_q_cfc')
        s = layers.AveragePooling1D(s.shape[1] // seg, name='dec_q_segpool')(s)   # (5, qw)
        s = conv_bn_act(s, quality_width, 1, name='dec_q_hidden')
        noise = layers.Conv1D(len(rc.NOISE_CLASSES), 1, activation='softmax', name='noise')(s)
        noise_logit = layers.Conv1D(1, 1, name='dec_q_noise_logit')(s)
    channel = layers.Softmax(name='channel')(
        layers.Concatenate(name='dec_q_logits')([noise_logit, score]))  # (5, 1 + c)

    return keras.Model(inp, {'beat': beat, 'rhythm': rhythm, 'channel': channel,
                             'noise': noise}, name=name)


def param_split(model):
    """(encoder params, decoder params), by layer name. Raises on a layer that is neither."""
    enc = dec = 0
    for layer in model.layers:
        p = layer.count_params()
        if layer.name.startswith('enc_') or layer.name == 'input':
            enc += p
        elif layer.name.startswith('dec_') or layer.name in OUTPUTS:
            dec += p
        else:
            raise ValueError(f"layer {layer.name!r} is neither enc_* nor dec_*")
    return enc, dec


# One size, fitted to the brief: encoder <= 900K, decoder <= 300K.
# Measured: encoder 881,648 + decoder 293,221 = 1,174,869.
SIZES = {
    'rhythm_u3cfc_1m': dict(stem=12, q_filters=(44, 60, 80, 104), t_filters=(28, 36, 52, 68),
                            a_filters=(28, 36, 52, 68), enc_width=104, lead_width=8,
                            beat_width=88, beat_cfc=2, rhythm_width=80, rhythm_cfc=2,
                            quality_width=24, state_dim=1),
}
# The same model with the template-residual noise decoder (sqi=True). Every layer the two share
# keeps its name and shape, so `--init-matching` from a rhythm_u3cfc_1m checkpoint warm-starts
# all of it and only the new noise path starts from scratch.
SIZES['rhythm_u3cfc_1m_sqi'] = dict(SIZES['rhythm_u3cfc_1m'], sqi=True)
# sqi=3 - the FINAL model: noise from the full ratio + TP ratio + TP coverage
# (TemplateResidualTP) and the stop-gradient beat-type probabilities. Its noise path has its
# own names and width, so warm-starting from a _sqi checkpoint leaves that path fresh and
# everything else frozen-able. Trained with ECGR_RHYTHM_NOISE_LABEL_MIN_LEADS=1.
SIZES['rhythm_u3cfc_1m_sqi3'] = dict(SIZES['rhythm_u3cfc_1m'], sqi=3, noise_width=24)
ENC_BUDGET, DEC_BUDGET = 900_000, 300_000
BUDGETS = {name: ENC_BUDGET + DEC_BUDGET for name in SIZES}
