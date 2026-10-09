"""ResUMamba for rhythm: the beat model's architecture, re-headed onto a one-second grid.

Everything up to the head is the beat family's own code (models/resumamba.py, models/layers.py),
called with a different output resolution rather than copied:

    input (2500, 3)
      -> backbone: stem -> ResU path (morphology) || diagonal-SSM path (multi-beat context)
                   -> fusion, at BACKBONE_STEPS = 250 steps (40 ms)
      -> AdaIN conditioned on the CPC-style strip context encoder
      -> cross-attention to rhythm tokens from the envelope autocorrelation
      -> per-second head:  avg || max pool to 10 steps -> SSM block across the seconds
      -> output 'rhythm' (10, 6): six rhythm classes per second, softmax

    input (2500, 3)
      -> lead-quality branch: ONE small conv net applied to every lead separately
      -> output 'lead' (4,): NOISE | CH1 | CH2 | CH3 for the whole window, softmax

Three choices differ from the beat model, all because the question is different:

  * The rhythm descriptor reads the max over ALL leads (channel=None). The beat model pins it
    to lead 0 because its labels refer to lead 0; a rhythm label refers to no lead, and the
    training pipeline permutes the leads anyway.
  * A second-level SSM block runs over the 10 pooled seconds. Whether second 4 is AF depends on
    the irregularity of the R-R intervals around it, and an AF episode, an AV block or a
    re-entrant run is a property of several seconds at once.

The lead branch is built to be permutation-EQUIVARIANT, because its target is: the leads are
shuffled in training and the answer moves with them. Every lead goes through the same weights
(a Conv2D with a (1, k) kernel over a (leads, time) image), each gets one score, and the three
scores are the CH1..CH3 logits - so the model cannot learn "CH2 is usually best", only "this
trace looks clean"; the ranking of CH1..CH3 is exactly equivariant (tests/test_rhythm.py).
The NOISE logit reads the mean and max of the three lead embeddings, which are invariant, plus
the pooled backbone features, which mix the leads and are invariant only as far as training on
permuted leads makes them.

Sizes: rhythm_2m, rhythm_1m, rhythm_100k, rhythm_30k - the beat family's widths, same budgets.
"""
import keras
from keras import layers

from ..models.layers import (AdaIN, BeatProbability, RhythmDescriptor, StopGradient,
                             conv_bn_act, ssm_block)
from ..models.resumamba import SIZES as BEAT_SIZES
from ..models.resumamba import build_backbone, build_context_encoder
from . import config as rc
from .dualunet import BUDGETS as DUAL_BUDGETS
from .dualunet import SIZES as DUAL_SIZES
from .dualunet import build_dual_unet_model
from .unet import beat_head, build_unet_backbone, sample_head


def build_rhythm_model(width=64, resu_mid=32, resu_depths=(3, 2), ssm_channels=64,
                       ssm_blocks=2, state_dim=8, kernel_len=128, ctx_dim=64, ctx_width=16,
                       adain_channels=48, adain_kernels=(3, 7), rhythm_tokens=4,
                       rhythm_dim=32, attn_heads=4, attn_key_dim=24, dropout=0.15,
                       separable=False, stem_channels=None, head_dim=None, lead_width=16,
                       lead_kernel=7, use_rhythm=True, use_context=True, backbone='resu',
                       unet_filters=None, unet_kernels=(7, 5, 5, 3), rhythm_steps=None,
                       quality_head='lead', trunk_pools=(5, 2), beat_decoder=False,
                       beat_tokens=4, name='resumamba_rhythm'):
    """(SEGMENT_SAMPLES, 3) -> {'rhythm': (steps, NUM_CLASSES), 'lead': (NUM_LEAD_CLASSES,)}.

    backbone='resu' is ResUMamba (ResU || Mamba), rhythm per SECOND (steps = 10).
    backbone='unet' is UNet-Mamba (unet.py: one full-depth U-Net || the same Mamba path),
    rhythm per SAMPLE (steps = 2500) through the U-Net's upper half (unet.sample_head), or
    per 20 ms with rhythm_steps = 500.
    quality_head='lead' is the window-level NOISE / CH1..CH3 output; 'noise' replaces it by
    'noise' (NOISE_SEGMENTS, 2), CLEAN / NOISE per 2 s (noise_head).
    beat_decoder=True (UNet backbones) adds the 'beat' output (unet.beat_head, none/N/S/V per
    enc1 step) and conditions the rhythm decoder on it through a stop-gradient: the beat
    probabilities pooled onto the 250-step grid join the fused features before an extra SSM
    block, and the autocorrelation of the beat-probability train (RhythmDescriptor on
    p(beat) instead of the signal envelope - RR regularity read from the beats themselves)
    adds `beat_tokens` tokens to the rhythm attention.
    The conditioning and the rhythm attention are shared."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    steps = rc.BACKBONE_STEPS
    head_dim = head_dim or width

    inp = keras.Input(shape=(n, c), name='input')
    if backbone == 'unet':
        backbone = build_unet_backbone(n, c, steps, width=width, unet_filters=unet_filters,
                                       unet_kernels=unet_kernels, ssm_channels=ssm_channels,
                                       ssm_blocks=ssm_blocks, state_dim=state_dim,
                                       kernel_len=kernel_len, separable=separable,
                                       stem_channels=stem_channels,
                                       trunk_pools=tuple(trunk_pools))
    elif backbone == 'resu':
        backbone = build_backbone(n, c, steps, width=width, resu_mid=resu_mid,
                                  resu_depths=resu_depths, ssm_channels=ssm_channels,
                                  ssm_blocks=ssm_blocks, state_dim=state_dim,
                                  kernel_len=kernel_len, separable=separable,
                                  stem_channels=stem_channels)
    else:
        raise ValueError(f"backbone must be 'resu' or 'unet', not {backbone!r}")
    skips = None
    y = backbone(inp)                                                     # (B, 250, width)
    if isinstance(y, (list, tuple)):
        y, *skips = y

    if use_context:
        ctx = build_context_encoder(n, c, dim=ctx_dim, width=ctx_width, separable=separable)
        f_p = ctx(inp)[0]
        for i, k in enumerate(adain_kernels):
            y = layers.Conv1D(adain_channels, k, padding='same', use_bias=False,
                              name=f'cond{i}_conv')(y)
            y = AdaIN(name=f'cond{i}_adain')([y, f_p])
            y = layers.LeakyReLU(0.1, name=f'cond{i}_act')(y)

    beat = None
    if beat_decoder:
        if not skips:
            raise ValueError("the beat decoder needs the UNet backbone's enc1 skip")
        enc1 = skips[0]
        beat = beat_head(y, enc1, separable=separable, dropout=dropout)
        beat_sg = StopGradient(name='beat_sg')(beat)                       # (B, enc1, 4)
        pooled_beat = layers.MaxPooling1D(enc1.shape[1] // steps, name='beat_to_grid')(beat_sg)
        y = layers.Concatenate(name='beat_cond')([y, pooled_beat])          # (B, 250, d + 4)
        y = conv_bn_act(y, y.shape[-1] - len(rc.BEAT_CLASSES), 1, name='beat_fuse')
        y = ssm_block(y, y.shape[-1], state_dim, kernel_len, name='beat_ctx')

    if use_rhythm:
        pooled = layers.AveragePooling1D(5, name='rhythm_pool')(inp)       # 50 Hz envelope
        ac = RhythmDescriptor(step_hz=rc.SAMPLING_RATE / 5.0, channel=None,
                              name='rhythm_ac')(pooled)
        tok = layers.Dense(rhythm_tokens * rhythm_dim, activation='relu',
                           name='rhythm_proj')(ac)
        tok = layers.Reshape((rhythm_tokens, rhythm_dim), name='rhythm_tokens')(tok)
        if beat is not None:
            # p(beat) per step: a spike train whose autocorrelation IS the RR structure
            p_beat = BeatProbability(name='beat_prob')(beat_sg)
            ac_b = RhythmDescriptor(step_hz=enc1.shape[1] / rc.SEGMENT_SECONDS, channel=0,
                                    name='beat_ac')(p_beat)
            tok_b = layers.Dense(beat_tokens * rhythm_dim, activation='relu',
                                 name='beat_tok_proj')(ac_b)
            tok_b = layers.Reshape((beat_tokens, rhythm_dim), name='beat_tokens')(tok_b)
            tok = layers.Concatenate(axis=1, name='rhythm_all_tokens')([tok, tok_b])
        attn = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim,
                                         name='rhythm_mha')(query=y, value=tok, key=tok)
        y = layers.LayerNormalization(name='rhythm_ln')(layers.Add(name='rhythm_add')([y, attn]))

    if quality_head == 'noise':
        quality = {'noise': noise_head(inp, y, lead_width, lead_kernel)}
    elif quality_head == 'lead':
        quality = {'lead': lead_quality_head(inp, y, lead_width, lead_kernel)}
    else:
        raise ValueError(f"quality_head must be 'lead' or 'noise', not {quality_head!r}")
    if skips:
        rhythm = sample_head(y, *skips, rc.NUM_CLASSES, separable=separable, dropout=dropout,
                             steps=rhythm_steps)
        outputs = {'rhythm': rhythm, **quality}
        if beat is not None:
            outputs['beat'] = beat
        return keras.Model(inp, outputs, name=name)
    if rhythm_steps not in (None, rc.OUTPUT_SECONDS):
        raise ValueError("only the UNet backbone emits rhythm finer than one step a second")

    # --- per-second head ---------------------------------------------------------------
    per_sec = steps // rc.OUTPUT_SECONDS
    s = layers.Concatenate(name='sec_pool')([
        layers.AveragePooling1D(per_sec, name='sec_avg')(y),
        layers.MaxPooling1D(per_sec, name='sec_max')(y)])                 # (B, 10, 2*width)
    local = conv_bn_act(s, head_dim, 1, name='sec_proj')

    ctx_sec = ssm_block(local, head_dim, state_dim, kernel_len=rc.OUTPUT_SECONDS,
                        name='sec_ssm')
    ctx_sec = layers.Dropout(dropout, name='head_drop')(ctx_sec)
    rhythm = layers.Conv1D(rc.NUM_CLASSES, 1, activation='softmax', name='rhythm')(ctx_sec)
    return keras.Model(inp, {'rhythm': rhythm, **quality}, name=name)


def lead_quality_head(inp, features, width, kernel, n_convs=4, pool=4):
    """(B, n, c) signal + (B, steps, d) backbone features -> (B, 1 + c) softmax."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    h = layers.Permute((2, 1), name='lead_perm')(inp)                     # (B, c, n)
    h = layers.Reshape((c, n, 1), name='lead_image')(h)                   # leads as rows
    for i in range(n_convs):
        h = layers.Conv2D(width, (1, kernel), padding='same', use_bias=False,
                          name=f'lead_conv{i}')(h)
        h = layers.BatchNormalization(name=f'lead_bn{i}')(h)
        h = layers.Activation('relu', name=f'lead_relu{i}')(h)
        h = layers.MaxPooling2D((1, pool), padding='same', name=f'lead_pool{i}')(h)
    t = h.shape[2]
    e = layers.Concatenate(name='lead_stats')([
        layers.AveragePooling2D((1, t), name='lead_avg')(h),
        layers.MaxPooling2D((1, t), name='lead_max')(h)])                 # (B, c, 1, 2w)
    e = layers.Reshape((c, 2 * width), name='lead_embed_flat')(e)
    e = layers.Dense(width, activation='relu', name='lead_embed')(e)      # (B, c, w)

    score = layers.Reshape((c,), name='lead_score_flat')(
        layers.Dense(1, name='lead_score')(e))                           # (B, c)
    pooled = layers.Concatenate(name='noise_in')([
        layers.GlobalAveragePooling1D(name='lead_mean')(e),
        layers.GlobalMaxPooling1D(name='lead_maxpool')(e),
        layers.GlobalAveragePooling1D(name='feat_mean')(features)])
    noise = layers.Dense(1, name='noise_logit')(pooled)                  # (B, 1)
    logits = layers.Concatenate(name='lead_logits')([noise, score])
    return layers.Softmax(name='lead')(logits)


def noise_head(inp, features, width, kernel, n_convs=4, pool=4):
    """(B, n, c) signal + (B, steps, d) backbone features -> (B, NOISE_SEGMENTS, 2) softmax.

    The lead branch's shared per-lead conv (same weights on every lead, a (1, k) kernel over
    the leads x time image), kept at NOISE_SEGMENTS time positions instead of pooled away:
    each 2 s segment gets one embedding per lead. CLEAN needs CLEAN_MIN_LEADS readable leads,
    so the segment reads the mean AND the max over the leads (how many are good, and the
    best one), plus the backbone features pooled to the same five segments."""
    n, c = rc.SEGMENT_SAMPLES, rc.IN_CHANNELS
    k = rc.NOISE_SEGMENTS
    h = layers.Permute((2, 1), name='noise_perm')(inp)                    # (B, c, n)
    h = layers.Reshape((c, n, 1), name='noise_image')(h)
    for i in range(n_convs):
        h = layers.Conv2D(width, (1, kernel), padding='same', use_bias=False,
                          name=f'noise_conv{i}')(h)
        h = layers.BatchNormalization(name=f'noise_bn{i}')(h)
        h = layers.Activation('relu', name=f'noise_relu{i}')(h)
        h = layers.MaxPooling2D((1, pool), padding='same', name=f'noise_pool{i}')(h)
    t = h.shape[2]
    if t % k:
        raise ValueError(f"noise branch ends at {t} time steps, not a multiple of {k}")
    h = layers.AveragePooling2D((1, t // k), name='noise_seg')(h)         # (B, c, k, w)
    h = layers.Dense(width, activation='relu', name='noise_embed')(h)
    lead_mean = layers.Reshape((k, width), name='noise_lead_mean')(
        layers.AveragePooling2D((c, 1), name='noise_mean')(h))
    lead_max = layers.Reshape((k, width), name='noise_lead_max')(
        layers.MaxPooling2D((c, 1), name='noise_max')(h))
    feat = layers.AveragePooling1D(features.shape[1] // k, name='noise_feat')(features)
    s = layers.Concatenate(name='noise_in')([lead_mean, lead_max, feat])  # (B, k, 2w + d)
    s = layers.Dense(width, activation='relu', name='noise_hidden')(s)
    return layers.Dense(len(rc.NOISE_CLASSES), activation='softmax', name='noise')(s)


SIZES = {f"rhythm_{k[len('resumamba_'):]}": dict(v) for k, v in BEAT_SIZES.items()}
BUDGETS = {'rhythm_2m': 2_000_000, 'rhythm_1m': 1_000_000,
           'rhythm_100k': 100_000, 'rhythm_30k': 30_000}
# The beat widths plus the second-level rhythm head and the lead branch overshoot every
# budget, so the head is narrower than the backbone and the ResU bottleneck gives up a little
# (2m 1,960,608 / 1m 941,176 / 100k 99,589 / 30k 29,983 params).
SIZES['rhythm_2m'].update(head_dim=64, resu_mid=80, lead_width=32)
SIZES['rhythm_1m'].update(head_dim=48, resu_mid=60, lead_width=24)
SIZES['rhythm_100k'].update(head_dim=20, lead_width=8)
SIZES['rhythm_30k'].update(head_dim=8, ctx_width=5, resu_mid=10, lead_width=6, lead_kernel=5)

# UNet-Mamba, per-sample output: the same sizes with the ResU path swapped for unet.py's U-Net
# and the per-second head for its upper half. The Mamba path, conditioning, attention and lead
# head keep the ResUMamba widths; the U-Net filters take the budget the ResU path and the
# per-second head had (2m 1,985,296 / 1m 985,208 / 100k 98,497 / 30k 29,115 params).
for _size, _unet in {'2m': dict(unet_filters=(32, 64, 96, 128)),
                     '1m': dict(unet_filters=(32, 48, 72, 96)),
                     '100k': dict(unet_filters=(14, 24, 32, 40)),
                     '30k': dict(unet_filters=(8, 10, 16, 20), adain_channels=20)}.items():
    SIZES[f'rhythm_unet_{_size}'] = dict(SIZES[f'rhythm_{_size}'], backbone='unet', **_unet)
    BUDGETS[f'rhythm_unet_{_size}'] = BUDGETS[f'rhythm_{_size}']

# UNet-Mamba at 20 ms with a CLEAN / NOISE output: 'rhythm' (500, 6) + 'noise' (5, 2). The
# decoder stops at enc1 (no stem-level climb) and the lead head becomes noise_head; the widths
# are re-fitted to the budgets.
for _size, _unet in {'2m': dict(unet_filters=(32, 64, 96, 128)),
                     '1m': dict(unet_filters=(32, 48, 72, 96)),
                     '100k': dict(unet_filters=(14, 24, 32, 40)),
                     '30k': dict(unet_filters=(8, 10, 16, 20), adain_channels=20)}.items():
    SIZES[f'rhythm_unet500_{_size}'] = dict(SIZES[f'rhythm_unet_{_size}'], rhythm_steps=500,
                                            quality_head='noise', **_unet)
    BUDGETS[f'rhythm_unet500_{_size}'] = BUDGETS[f'rhythm_{_size}']

# UNet-Mamba at 8 ms with the CLEAN / NOISE output: 'rhythm' (1250, 6) + 'noise' (5, 2). The
# trunk pools 2 then 5 (2500 -> 1250 -> 250) so enc1 sits at 1250 steps and the head climbs
# 250 -> 1250 through it; everything else is the 500 family.
for _size in ('2m', '1m', '100k', '30k'):
    SIZES[f'rhythm_unet1250_{_size}'] = dict(SIZES[f'rhythm_unet500_{_size}'],
                                             rhythm_steps=1250, trunk_pools=(2, 5))
    BUDGETS[f'rhythm_unet1250_{_size}'] = BUDGETS[f'rhythm_{_size}']

# ... with the beat decoder and the beat-conditioned rhythm decoder: 'rhythm' (1250, 6),
# 'noise' (5, 2), 'beat' (1250, 4). The U-Net filters give up a little for the two extra
# blocks so every size stays under its budget.
for _size, _unet in {'2m': dict(unet_filters=(28, 56, 80, 96)),
                     '1m': dict(unet_filters=(28, 44, 56, 72)),
                     '100k': dict(unet_filters=(8, 12, 16, 20), adain_channels=28),
                     '30k': dict(unet_filters=(4, 8, 8, 12), adain_channels=12,
                                 rhythm_dim=8, beat_tokens=2, lead_width=4)}.items():
    SIZES[f'rhythm_unet1250b_{_size}'] = dict(SIZES[f'rhythm_unet1250_{_size}'],
                                              beat_decoder=True, **_unet)
    BUDGETS[f'rhythm_unet1250b_{_size}'] = BUDGETS[f'rhythm_{_size}']

# ... the warm-start variant: the 1m U-Net filters UNCHANGED (32, 48, 72, 96) so every encoder
# and U-Net tensor of a trained rhythm_unet1250_1m fits by name and shape
# (train.init_matching; the reduced-filter 1250b_1m only inherits the stem, SSM path, context
# branch and noise head). The beat blocks push it over the 1m budget (~1.19 M parameters).
SIZES['rhythm_unet1250b_1mw'] = dict(SIZES['rhythm_unet1250_1m'], beat_decoder=True)
BUDGETS['rhythm_unet1250b_1mw'] = 1_300_000

# Dual U-Net (dualunet.py): shared encoder with a ventricular and an atrial (QRST-cancelled)
# branch, three decoders - 'beat' (1250, 4), 'rhythm' (1250, 6), 'channel' (5, 4).
for _name, _kw in DUAL_SIZES.items():
    SIZES[_name] = dict(_kw, family='dual')
    BUDGETS[_name] = DUAL_BUDGETS[_name]


def rhythm_steps(model):
    """Rows of the model's 'rhythm' output per 10 s window: 10, 500 or 2500."""
    return int(model.output['rhythm'].shape[1])


def is_per_sample(model):
    """True when the model's 'rhythm' output is one row per input sample."""
    return rhythm_steps(model) == rc.SEGMENT_SAMPLES


def output_names(model):
    return tuple(model.output)


def list_models():
    return sorted(SIZES, key=lambda n: ('dual' in n, 'unet1250b' in n, 'unet1250' in n,
                                        'unet500' in n, 'unet' in n, -BUDGETS[n]))


def keras_name(name):
    if name.startswith('rhythm_dual_'):
        return f"dualunet_rhythm_{name[len('rhythm_dual_'):]}"
    if name.startswith('rhythm_unet1250b_'):
        return f"unetmamba1250b_rhythm_{name[len('rhythm_unet1250b_'):]}"
    if name.startswith('rhythm_unet1250_'):
        return f"unetmamba1250_rhythm_{name[len('rhythm_unet1250_'):]}"
    if name.startswith('rhythm_unet500_'):
        return f"unetmamba500_rhythm_{name[len('rhythm_unet500_'):]}"
    if name.startswith('rhythm_unet_'):
        return f"unetmamba_rhythm_{name[len('rhythm_unet_'):]}"
    return f"resumamba_{name}"


def build(name, **kw):
    if name not in SIZES:
        raise KeyError(f"unknown rhythm model {name!r}; available: {list_models()}")
    kwargs = dict(SIZES[name], name=keras_name(name))
    kwargs.update(kw)
    if kwargs.pop('family', None) == 'dual':
        return build_dual_unet_model(**kwargs)
    return build_rhythm_model(**kwargs)
