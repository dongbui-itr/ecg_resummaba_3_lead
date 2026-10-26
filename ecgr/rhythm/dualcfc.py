"""Dual U-Net encoder + CfC, four decoders: `rhythm_dualcfc_*`.

The dual U-Net of dualunet.py with two changes, and nothing else moved so the pair stays
comparable:

  1. **Every time mixer is a CfC cell** (models.layers.cfc_block) instead of the diagonal
     state space. The reason is the hardware, and it is the same reason as for the beat
     family: DiagSSM1D exports to a depthwise FIR `kernel_len` taps deep, which costs one
     MAC, one coefficient and one delay-line word per tap per channel; a CfC keeps ONE state
     register per cell and computes its time constant from the input. Measured over the beat
     models, that is 70-190x less runtime memory and 9-24x fewer MACs for the mixer.
     It also puts back the input-dependent dynamics the diagonal SSM gave away - the cell can
     hold its state through an artefact and release it at a QRS.
  2. **Four decoders on one encoder**, at the resolutions the task actually needs rather than
     at whatever the encoder happens to produce:

        beat    (1250, 4)   nobeat / N / S / V      8 ms
        rhythm  (20, 5)     SINUS/AFIB/SVT/VT/AVB   0.5 s
        channel (5, 4)      NOISE / CH1 / CH2 / CH3 2 s
        noise   (5, 2)      CLEAN / NOISE           2 s

     `channel` and `noise` are separate heads here. In dualunet.py NOISE is the first column
     of `channel`, which forces one softmax to answer two questions - "is this segment
     readable" and "which lead is cleanest" - and a segment can be clean on CH2 while the
     answer to the first is still NOISE. Split, each head has its own target
     (augment.targets gives both) and neither distorts the other.

Why the two branches keep different filters: the QRS is 1-2 mV, a P wave 0.1-0.25 mV, f waves
0.05-0.1 mV. The V branch max-pools (a max keeps the tallest thing in the window - the QRS,
the T) and the A branch average-pools the QRST-cancelled residual (a 50 uV oscillation
survives an average, a max-pool replaces it with the nearest spike). Same U-Net shape, two
filter ladders, two pooling rules - that is what lets one model see both.

**The 20-step rhythm grid does not divide the 1250-step encoder.** 1250 = 2 x 5^4 and 20 =
2^2 x 5, so 1250/20 = 62.5: there is no integer pooling from the encoder straight to the
rhythm head. The path here pools 1250 -> 10 (one second, /125, exact) and upsamples x2 to the
half-second grid, with a 20-step skip taken from the 2500-sample stem (2500/125 = 20, also
exact). Every resampling in this file is an integer ratio; none of it is a resize.
"""
import keras
import numpy as np
from keras import layers

from ..models.layers import RhythmDescriptor, StopGradient, cfc_block, conv_bn_act
from . import config as rc
from .dualunet import (BEAT_HZ, BeatCompose, FiLM, LeadImage, LeadPool, QRSTCancel,
                       _dilated_block, _double_conv, _lead_conv, _lead_logits)

PKG = 'ecgr_rhythm'


def build_dual_cfc_model(stem=12, v_filters=(36, 56, 72, 96), a_filters=(32, 48, 64, 80),
                         width=72, cfc_blocks=2, state_dim=1, horizon=256, beat_width=40,
                         rhythm_width=48, lead_width=16, dropout=0.15,
                         name='dualcfc_rhythm'):
    """(2500, 3) -> {'beat': (1250,4), 'rhythm': (20,5), 'channel': (5,4), 'noise': (5,2)}."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    t1, t2 = rc.BEAT_STEPS, rc.BACKBONE_STEPS                            # 1250, 250
    p1, p2, p3 = n // t1, t1 // t2, 5                                    # 2, 5, 5
    v1f, v2f, v3f, v4f = v_filters
    a1f, a2f, a3f, a4f = a_filters
    seg, rsteps = rc.NOISE_SEGMENTS, rc.RHYTHM_STEPS                     # 5, 20

    inp = keras.Input(shape=(n, c), name='input')

    # --- shared per-lead stem and lead pooling ---------------------------------------------
    img = LeadImage(name='lead_image')(inp)
    f0 = _lead_conv(img, stem, 9, 2, 'stem')                             # (B, c, 2500, stem)
    lead_logit = _lead_logits(f0, 'lead_w')
    x0 = LeadPool(name='stem_pool')([f0, lead_logit])                    # (2500, 3 stem)

    # --- V branch: QRS / T, max-pool ---------------------------------------------------------
    v1 = _double_conv(layers.MaxPooling1D(p1, name='v0_pool')(x0), v1f, 7, 'v1')      # 1250
    v2 = _double_conv(layers.MaxPooling1D(p2, name='v1_pool')(v1), v2f, 5, 'v2')      # 250
    v3 = _double_conv(layers.MaxPooling1D(p3, name='v2_pool')(v2), v3f, 5, 'v3')      # 50
    vb = _double_conv(layers.MaxPooling1D(p3, name='v3_pool')(v3), v4f, 3, 'vb')      # 10
    vs = cfc_block(conv_bn_act(v2, width, 1, name='vs_in'), width, state_dim, horizon,
                   name='vs')

    # --- beat POSITION decoder (the template below depends on it) --------------------------
    h = layers.UpSampling1D(p3, name='pos3_up')(vb)
    h = _double_conv(layers.Concatenate(name='pos3_skip')([h, v3]), v3f, 5, 'pos3')
    h = layers.UpSampling1D(p3, name='pos2_up')(h)
    h = _double_conv(layers.Concatenate(name='pos2_skip')([h, v2, vs]), v2f, 5, 'pos2')
    h = layers.UpSampling1D(p2, name='pos1_up')(h)
    h = _double_conv(layers.Concatenate(name='pos1_skip')([h, v1]), beat_width, 7, 'pos1')
    pos_logit = layers.Conv1D(1, 1, name='beat_pos')(h)
    p_pos = layers.Activation('sigmoid', name='beat_pos_prob')(pos_logit)

    # --- QRST cancellation -> A branch: P / F / f, average-pool ------------------------------
    residual, qrs = QRSTCancel(name='qrst_cancel')([inp, p_pos])
    r_img = LeadImage(name='atrial_image')(residual)
    fa = _lead_conv(r_img, stem, 7, 2, 'astem')
    a_logit = _lead_logits(fa, 'alead_w')
    a0 = LeadPool(name='astem_pool')([fa, a_logit])
    a0 = layers.Concatenate(name='a0_mask')([a0, qrs])
    a1 = _dilated_block(a0, a1f, 7, (1, 2, 4, 8), 'a1')                               # 1250
    a2 = _dilated_block(layers.AveragePooling1D(p2, name='a1_pool')(a1), a2f, 5, (1, 2, 4),
                        'a2')                                                          # 250
    a3 = _double_conv(layers.AveragePooling1D(p3, name='a2_pool')(a2), a3f, 5, 'a3')   # 50
    ab = _double_conv(layers.AveragePooling1D(p3, name='a3_pool')(a3), a4f, 3, 'ab')   # 10

    # --- ENCODER: both bottoms up to 1250, CfC over the whole strip ------------------------
    h = layers.Concatenate(name='neck_bottom')([vb, ab])
    h = layers.UpSampling1D(p3, name='neck3_up')(h)
    h = _double_conv(layers.Concatenate(name='neck3_skip')([h, v3, a3]), v3f, 5, 'neck3')
    h = layers.UpSampling1D(p3, name='neck2_up')(h)
    h = _double_conv(layers.Concatenate(name='neck2_skip')([h, v2, a2, vs]), width, 5, 'neck2')
    h = layers.UpSampling1D(p2, name='neck1_up')(h)
    h = _double_conv(layers.Concatenate(name='neck1_skip')([h, v1, a1]), width, 7, 'neck1')
    for i in range(cfc_blocks):
        h = cfc_block(h, width, state_dim, horizon, name=f'neck_cfc{i}')
    z = h                                                                # ENCODER (1250, width)

    # --- decoder 1: beat type, composed with the position into (1250, 4) -------------------
    h = layers.Dropout(dropout, name='type_drop')(
        _double_conv(layers.Concatenate(name='type_skip')([z, v1, a1]), beat_width, 7, 'type'))
    type_logit = layers.Conv1D(3, 1, name='beat_type')(h)
    beat = BeatCompose(name='beat')([pos_logit, type_logit])             # (1250, 4)

    # --- decoder 2: rhythm at 0.5 s, conditioned on the beats (stop-gradient) --------------
    # 1250 -> 10 is exact (/125) and 10 -> 20 is an integer upsample; the 20-step skip comes
    # from the 2500-sample stem (2500/125 = 20). See the module docstring on why not /62.5.
    beat_sg = StopGradient(name='beat_sg')(beat)
    sec = layers.Concatenate(name='rhythm_sec_in')([
        layers.AveragePooling1D(t1 // rc.OUTPUT_SECONDS, name='rhythm_z_sec')(z),
        layers.MaxPooling1D(t1 // rc.OUTPUT_SECONDS, name='rhythm_beat_sec')(beat_sg)])
    h = layers.UpSampling1D(rsteps // rc.OUTPUT_SECONDS, name='rhythm_up')(sec)   # (20, .)
    skip = layers.AveragePooling1D(n // rsteps, name='rhythm_stem_sec')(x0)       # (20, .)
    h = conv_bn_act(layers.Concatenate(name='rhythm_skip')([h, skip]), rhythm_width, 1,
                    name='rhythm_fuse')
    p_sg = StopGradient(name='pos_sg')(p_pos)
    rr = RhythmDescriptor(step_hz=BEAT_HZ, channel=0, name='rr_ac')(p_sg)
    rr = layers.Dense(rhythm_width, activation='relu', name='rr_hidden')(rr)
    rr = layers.Dense(2 * rhythm_width, name='rr_film')(rr)
    h = FiLM(name='rhythm_film')([h, rr])
    # horizon 32 >= the 20 steps: the scan is then exact over the whole rhythm grid.
    h = cfc_block(h, rhythm_width, state_dim, 32, name='rhythm_cfc')
    h = _double_conv(h, rhythm_width, 3, 'rhythm_dec')
    h = layers.Dropout(dropout, name='rhythm_drop')(h)
    rhythm = layers.Conv1D(rc.NUM_CLASSES, 1, activation='softmax', name='rhythm')(h)

    # --- decoders 3 and 4: per-lead embedding, then channel and noise separately ------------
    e = f0
    for i, pool in enumerate((5, 5, 4)):                                 # 2500 -> 500 -> 100 -> 25
        e = _lead_conv(e, lead_width, 7, 1, f'chan{i}_')
        e = layers.MaxPooling2D((1, pool), name=f'chan{i}_pool')(e)
    e = layers.AveragePooling2D((1, e.shape[2] // seg), name='chan_seg')(e)
    e = layers.Dense(lead_width, activation='relu', name='chan_embed')(e)      # (B, c, 5, w)
    score = layers.Permute((2, 1), name='chan_score_t')(
        layers.Reshape((c, seg), name='chan_score_flat')(
            layers.Dense(1, name='chan_score')(e)))                            # (B, 5, c)
    lead_mean = layers.Reshape((seg, lead_width), name='chan_lead_mean')(
        layers.AveragePooling2D((c, 1), name='chan_mean')(e))
    lead_max = layers.Reshape((seg, lead_width), name='chan_lead_max')(
        layers.MaxPooling2D((c, 1), name='chan_max')(e))
    ctx = layers.AveragePooling1D(t1 // seg, name='chan_ctx')(z)                # (B, 5, width)
    s = layers.Concatenate(name='chan_noise_in')([lead_mean, lead_max, ctx])
    s = layers.Dense(lead_width, activation='relu', name='chan_noise_hidden')(s)

    # 'noise' answers readability on its own logit pair, and 'channel' keeps a NOISE column
    # so it stays a distribution over the four labels augment.targets emits for it.
    noise = layers.Dense(len(rc.NOISE_CLASSES), activation='softmax', name='noise')(s)
    noise_logit = layers.Dense(1, name='chan_noise_logit')(s)
    channel = layers.Softmax(name='channel')(
        layers.Concatenate(name='chan_logits')([noise_logit, score]))           # (B, 5, 4)

    return keras.Model(inp, {'beat': beat, 'rhythm': rhythm, 'channel': channel,
                             'noise': noise}, name=name)


# Widths fitted to the budgets. A CfC mixer is far cheaper in parameters than the FIR bank it
# replaces (6 weights per cell against 4 x state_dim x 2), so each size can afford slightly
# wider convolutions than its dualunet twin and still land under the same ceiling.
SIZES = {
    'rhythm_dualcfc_1m': dict(stem=12, v_filters=(34, 52, 70, 92), a_filters=(30, 44, 60, 78),
                              width=70, beat_width=38, rhythm_width=52, lead_width=16,
                              cfc_blocks=2, state_dim=1, horizon=256),
    'rhythm_dualcfc_100k': dict(stem=6, v_filters=(10, 15, 21, 26), a_filters=(8, 13, 17, 21),
                                width=21, beat_width=12, rhythm_width=16, lead_width=6,
                                cfc_blocks=1, state_dim=1, horizon=256),
    'rhythm_dualcfc_30k': dict(stem=4, v_filters=(6, 8, 11, 13), a_filters=(4, 6, 8, 11),
                               width=11, beat_width=6, rhythm_width=9, lead_width=4,
                               cfc_blocks=1, state_dim=1, horizon=128),
}
BUDGETS = {'rhythm_dualcfc_1m': 1_000_000, 'rhythm_dualcfc_100k': 100_000,
           'rhythm_dualcfc_30k': 30_000}
OUTPUTS = ('beat', 'rhythm', 'channel', 'noise')


def _builder(size):
    def build(**kw):
        kwargs = dict(SIZES[size])
        kwargs['name'] = size
        kwargs.update(kw)
        return build_dual_cfc_model(**kwargs)
    build.__name__ = f'build_{size}'
    build.__doc__ = f'{size}: {SIZES[size]}'
    return build


BUILDERS = {size: _builder(size) for size in SIZES}
