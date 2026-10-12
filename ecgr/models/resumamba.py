"""ResUMamba seq2seq - Heo et al. (ESWA 331, 2026) adapted to this project's beat contract.

The paper ("Patient-conditioned ECG beat classification via self-supervised embeddings")
classifies ONE beat at a time: a 720-sample window centred on an annotated R peak goes into a
dual-path ResUNet + Mamba backbone, a CPC-pretrained patient embedding modulates the features
through AdaIN, hand-crafted R-R descriptors are fused by cross-attention, and a Poly-2 loss
fights the class imbalance.

Here the contract is different - input (SEGMENT_SAMPLES, IN_CHANNELS) = 10 s at 250 Hz over
three leads, output (OUTPUT_STEPS, NUM_CLASSES) softmax, i.e. detection AND classification in
one pass with no R peaks given - so four things are adapted rather than copied:

- Dual path, kept. The ResU path (Hwang et al. 2023 blocks the paper cites) covers local
  morphology; the state-space path covers the multi-beat context. Both run at output-step
  resolution, so every one of the 500 steps gets its own class instead of one label per window.
- Mamba -> diagonal SSM as a long convolution (DiagSSM1D). A selective S6 scan is a sequential
  recurrence: in TF it means tf.scan over 500 steps, which is slow to train and does not export.
  A diagonal (input-independent) SSM has a closed-form kernel, so the whole recurrence becomes
  one depthwise FIR convolution - same linear complexity and long memory, parallel in time, and
  it lowers to a plain conv for TFLite. It is also run in BOTH directions here: the paper's
  Mamba is causal because it streams, while this model already sees the whole 10 s strip, and
  the T-P segment that separates S from N sits BEFORE the beat being labelled.
- Patient embedding -> strip context embedding. The paper needs 60 s of unlabeled calibration
  ECG per patient; the tfrecord pipeline has no patient identity and no calibration segment, so
  the same CPC machinery (windowed encoder + autoregressive context model, training/cpc.py) is
  run over the 10 s strip itself. It still does what AdaIN needs it to do: carry amplitude,
  baseline and rhythm regime, which is exactly the nuisance variation AdaIN removes.
- Hand-crafted R-R statistics -> in-graph rhythm descriptor. R-R statistics presuppose the beat
  positions this model is supposed to produce, so the rhythm cue is taken from the envelope
  autocorrelation over the 30-220 bpm lag band instead (RhythmDescriptor) - no beat detector, a
  few tens of thousands of MACs. Cross-attention is also reversed: the paper makes the clinical
  vector the query because it wants one output, this model makes the per-step features the
  queries and the rhythm tokens the keys/values, because it wants 500 outputs.

**Three leads.** The channel axis carries the whole portal montage with the annotated lead on
channel 0 (signal_ops.build_leads). Only two places in the graph care which lead is which -
the flatness test outside the model and RhythmDescriptor's envelope - and both read channel 0,
so the model behaves identically whether it is handed three different leads or, as on the EC57
databases, one lead three times.

**Two nested sub-models.** `backbone` (stem + both paths + their fusion) and
`context_encoder` are built as their own keras.Models rather than inlined, because both are
pretrained without labels and then loaded by name: training/ssl.py trains the backbone by
masked reconstruction, training/cpc.py trains the context encoder by InfoNCE. Nothing else
about the layout changes - the same layers in the same order, one nesting level down.

Four sizes, all reachable through models.build(name): resumamba_2m, resumamba_1m,
resumamba_100k and resumamba_30k.
"""
import keras
from keras import layers

from .. import config
from .layers import (AdaIN, CfC1D, DiagSSM1D, Frame1D, MergeWindows, RhythmDescriptor,
                     SplitWindows, TIME_MIXERS, conv_bn_act, res_u_block, ssm_block,
                     _pool_plan)


def _ar_layer(time_mixer, state_dim, memory, name):
    """The context encoder's autoregressive layer, in whichever time mixer is in use.

    Causal in both cases, and for the same reason: the CPC objective is to predict a future
    latent, which is trivially solvable if the context vector may look at it.
    """
    if time_mixer == 'cfc':
        return CfC1D(state_dim=state_dim, horizon=memory, bidirectional=False, name=name)
    return DiagSSM1D(state_dim=state_dim, kernel_len=memory, bidirectional=False, name=name)


def build_context_encoder(input_length, in_channels, dim=64, width=16, window=500, hop=250,
                          state_dim=4, separable=False, time_mixer='ssm',
                          name='context_encoder'):
    """CPC-style context encoder: window -> shared embedding net -> AR model -> mean.

    Mirrors Fig. 3 of the paper (embedding network + autoregressive context model, the context
    vectors averaged into one embedding) at the scale this pipeline allows: the 10 s strip is
    cut into 2 s windows with 50% overlap, giving 9 windows instead of the paper's 60 s of
    calibration. Kept as its own Model so training/cpc.py can train these exact weights with
    the InfoNCE objective and hand them back frozen.

    The AR model is causal on purpose - a context vector may only summarise the past, or the CPC
    objective (predict the future latent) would be trivially solvable.
    """
    n_win = (input_length - window) // hop + 1
    inp = keras.Input(shape=(input_length, in_channels), name='ctx_input')

    # (B, T, C) -> (B, M, window, C), then ONE shared embedding net over the window axis.
    # The window axis is folded into the batch rather than wrapped in TimeDistributed, which
    # lowers to a pfor/while loop as soon as the time axis is dynamic - and the training
    # pipeline's time-scale augmentation makes it dynamic.
    frames = Frame1D(window, hop, name='ctx_frames')(inp)

    h = MergeWindows(name='ctx_merge')(frames)                           # (B*M, window, C)
    for i, mult in enumerate((1, 2, 4)):
        h = conv_bn_act(h, width * mult, 5, separable, name=f'ctx_e{i}a')
        h = conv_bn_act(h, width * mult, 5, separable, name=f'ctx_e{i}b')
        h = layers.MaxPooling1D(4, padding='same', name=f'ctx_pool{i}')(h)
    h = layers.GlobalAveragePooling1D(name='ctx_gap')(h)
    h = layers.Dense(dim, name='ctx_embed')(h)                           # (B*M, D)

    v_seq = SplitWindows(n_win, name='ctx_windows')(h)                   # (B, M, D)
    c_seq = _ar_layer(time_mixer, state_dim, min(8, n_win), 'ctx_ar')(v_seq)
    c_seq = layers.Activation('tanh', name='ctx_ar_act')(c_seq)
    f_p = layers.GlobalAveragePooling1D(name='ctx_pool')(c_seq)          # (B, D)

    return keras.Model(inp, [f_p, v_seq, c_seq], name=name)


def build_backbone(input_length, in_channels, output_steps, width=64, resu_mid=32,
                   resu_depths=(3, 2), ssm_channels=64, ssm_blocks=2, state_dim=8,
                   kernel_len=128, separable=False, stem_channels=None, time_mixer='ssm',
                   mixer_memory=None, name='backbone'):
    """Stem + the two paths + their fusion: (T, leads) -> (output_steps, width).

    Everything in this sub-model is label-independent, which is why it is the unit the
    self-supervised stage pretrains (training/ssl.py). The classifier head, the AdaIN
    conditioning and the rhythm attention sit outside it in build_resumamba_seq2seq.
    """
    pools = _pool_plan(input_length, output_steps)
    stem_channels = stem_channels or max(width // 4, 8)
    inp = keras.Input(shape=(input_length, in_channels), name='backbone_input')

    # --- stem: full-resolution morphology, then down to the label grid --------------------
    x = conv_bn_act(inp, stem_channels, 9, name='stem1')
    x = conv_bn_act(x, stem_channels, 9, name='stem2')
    for i, p in enumerate(pools):
        x = layers.MaxPooling1D(p, padding='same', name=f'stem_pool{i}')(x)
    x = conv_bn_act(x, width, 5, separable, name='stem_out')             # (B, steps, width)

    # --- path 1: ResU, multi-scale morphology --------------------------------------------
    y_u = x
    for i, depth in enumerate(resu_depths):
        y_u = res_u_block(y_u, width, resu_mid, depth, separable, name=f'resu{i}')

    # --- path 2: the time mixer, multi-beat context ---------------------------------------
    # The block keeps the name 'ssm{i}' whichever mixer is inside it, and that is load-
    # bearing rather than lazy: every weight of the block EXCEPT the mixer itself - in_proj,
    # the depthwise 3, both BatchNorms, gate_proj, out_proj - has the same name and the same
    # shape in both variants, so a cfc student can be warm-started from an ssm teacher and
    # only the mixer starts cold. See training/distill.transfer_shared_weights.
    mixer = TIME_MIXERS[time_mixer]
    memory = kernel_len if mixer_memory is None else mixer_memory
    y_m = layers.Conv1D(ssm_channels, 1, use_bias=False, name='ssm_in')(x) \
        if ssm_channels != width else x
    for i in range(ssm_blocks):
        y_m = mixer(y_m, ssm_channels, state_dim, memory, name=f'ssm{i}')

    y = layers.Concatenate(name='dual_concat')([y_u, y_m])
    y = conv_bn_act(y, width, 1, name='dual_fuse')
    return keras.Model(inp, y, name=name)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def build_resumamba_seq2seq(
                            in_channels=None,
                            width=64,
                            resu_mid=32,
                            resu_depths=(3, 2),
                            ssm_channels=64,
                            ssm_blocks=2,
                            state_dim=8,
                            kernel_len=128,
                            ctx_dim=64,
                            ctx_width=16,
                            adain_channels=48,
                            adain_kernels=(3, 7),
                            rhythm_tokens=4,
                            rhythm_dim=32,
                            attn_heads=4,
                            attn_key_dim=24,
                            dropout=0.15,
                            separable=False,
                            stem_channels=None,
                            use_rhythm=True,
                            use_context=True,
                            time_mixer='ssm',
                            mixer_memory=None,
                            ctx_state_dim=4,
                            name='resumamba_seq2seq'):
    """Dual-path (ResU + state-space) seq2seq beat detector/classifier.

    input (SEGMENT_SAMPLES, in_channels) -> output (OUTPUT_STEPS, NUM_CLASSES) softmax.
    """
    in_channels = config.IN_CHANNELS if in_channels is None else in_channels
    input_length = config.SEGMENT_SAMPLES
    output_steps = config.OUTPUT_STEPS
    num_classes = config.NUM_CLASSES

    inp = keras.Input(shape=(input_length, in_channels), name='input')

    backbone = build_backbone(input_length, in_channels, output_steps, width=width,
                              resu_mid=resu_mid, resu_depths=resu_depths,
                              ssm_channels=ssm_channels, ssm_blocks=ssm_blocks,
                              state_dim=state_dim, kernel_len=kernel_len,
                              separable=separable, stem_channels=stem_channels,
                              time_mixer=time_mixer, mixer_memory=mixer_memory)
    y = backbone(inp)

    # --- strip-context conditioning (AdaIN) ----------------------------------------------
    if use_context:
        ctx_encoder = build_context_encoder(input_length, in_channels, dim=ctx_dim,
                                            width=ctx_width, separable=separable,
                                            state_dim=ctx_state_dim, time_mixer=time_mixer)
        f_p = ctx_encoder(inp)[0]
        z = y
        for i, k in enumerate(adain_kernels):
            z = layers.Conv1D(adain_channels, k, padding='same', use_bias=False,
                              name=f'cond{i}_conv')(z)
            z = AdaIN(name=f'cond{i}_adain')([z, f_p])
            z = layers.LeakyReLU(0.1, name=f'cond{i}_act')(z)
        y = z

    # --- rhythm cross-attention ----------------------------------------------------------
    if use_rhythm:
        step_hz = config.OUTPUT_STEPS / config.SEGMENT_SECONDS
        pooled = layers.AveragePooling1D(int(input_length // output_steps),
                                         name='rhythm_pool')(inp)
        # channel=0: the annotated lead, the one quantity that is the same on a 3-lead strip
        # and on an EC57 record whose single lead was repeated (see RhythmDescriptor).
        ac = RhythmDescriptor(step_hz=step_hz, channel=0, name='rhythm_ac')(pooled)
        tok = layers.Dense(rhythm_tokens * rhythm_dim, activation='relu',
                           name='rhythm_proj')(ac)
        tok = layers.Reshape((rhythm_tokens, rhythm_dim), name='rhythm_tokens')(tok)
        attn = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim,
                                         name='rhythm_mha')(query=y, value=tok, key=tok)
        y = layers.LayerNormalization(name='rhythm_ln')(
            layers.Add(name='rhythm_add')([y, attn]))

    y = layers.Dropout(dropout, name='head_drop')(y)
    out = layers.Conv1D(num_classes, 1, activation='softmax', name='beat_cls')(y)
    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------------
# One knob is deliberately NOT scaled with the budget: the depth of the state-space path.
# The receptive field it buys is what separates S from N, and an SSM kernel costs 4 scalars
# per (channel, state) - 576 weights for the whole path in the 30k size. The budget is taken
# out of the widths and out of the ResU path's dense convolutions, which become
# depthwise-separable below 100k: a dense k=3 conv at 48 channels costs 3*48*48, the
# separable one 3*48 + 48*48.

SIZES = {
    # 1,977,660 params at 3 leads. Widest of the family, four SSM blocks, the longest
    # state-space kernel (256 steps = 5.1 s, more than four R-R intervals at 60 bpm).
    'resumamba_2m': dict(width=192, resu_mid=88, resu_depths=(3, 2), ssm_channels=192,
                         ssm_blocks=4, state_dim=16, kernel_len=256, ctx_dim=128,
                         ctx_width=32, adain_channels=128, rhythm_tokens=4, rhythm_dim=64,
                         attn_heads=4, attn_key_dim=40, dropout=0.2, separable=False,
                         stem_channels=40),
    # 934,476 params. Dense convs, 3 SSM blocks, 4 attention heads.
    'resumamba_1m': dict(width=128, resu_mid=64, resu_depths=(3, 2), ssm_channels=128,
                         ssm_blocks=3, state_dim=12, kernel_len=192, ctx_dim=96,
                         ctx_width=24, adain_channels=96, rhythm_tokens=4, rhythm_dim=48,
                         attn_heads=4, attn_key_dim=32, dropout=0.15, separable=False,
                         stem_channels=32),
    # 93,517 params. Same layout at ~1/3 the width, with separable convs in both paths.
    'resumamba_100k': dict(width=48, resu_mid=24, resu_depths=(3, 2), ssm_channels=48,
                           ssm_blocks=2, state_dim=8, kernel_len=160, ctx_dim=40,
                           ctx_width=10, adain_channels=40, rhythm_tokens=4, rhythm_dim=24,
                           attn_heads=2, attn_key_dim=16, dropout=0.15, separable=True,
                           stem_channels=14),
    # 29,741 params, the embedded target. Neither the number of SSM blocks nor the ResU
    # depth is reduced; the budget comes out of the width instead.
    'resumamba_30k': dict(width=24, resu_mid=12, resu_depths=(3, 2), ssm_channels=24,
                          ssm_blocks=2, state_dim=6, kernel_len=128, ctx_dim=20,
                          ctx_width=6, adain_channels=24, rhythm_tokens=2, rhythm_dim=16,
                          attn_heads=2, attn_key_dim=8, dropout=0.1, separable=True,
                          stem_channels=8),
}

# Parameter ceiling each size promises to stay under, checked by tests/test_models.py.
BUDGETS = {'resumamba_2m': 2_000_000, 'resumamba_1m': 1_000_000,
           'resumamba_100k': 100_000, 'resumamba_30k': 30_000}


# ---------------------------------------------------------------------------
# The CfC family: the same model with the state-space path swapped for CfC1D
# ---------------------------------------------------------------------------
# One layer per block changes and nothing else, so the EC57 difference against the
# resumamba size of the same name is attributable to the time mixer alone.
#
# Two knobs move with the swap, both because the cost model is different:
#
#   state_dim  DOWN to 1. An SSM state is four scalars in a kernel that is summed away
#              before it ever becomes an activation; a CfC state is a real (B, T, C, N)
#              tensor carried through the scan, so N multiplies the training footprint
#              directly. Measured on resucfc_2m through fit() at batch 64: N=2 costs
#              295 ms/step and 4.2 GB, N=1 costs 203 ms/step and 3.2 GB - and at batch 128
#              N=2 does not fit on a 24 GB card at all.
#              The reason 1 is enough is not the memory, though: in an SSM `state_dim` buys
#              a BANK of fixed time constants, because a kernel that cannot adapt needs
#              several. CfC computes its rate from the input at every step, so one adaptive
#              cell does what the bank was there for. It is also half the silicon state.
#   memory     UP, and free. DiagSSM1D pays one tap, one coefficient and one delay-line word
#              per step of receptive field, so the sizes bought 128-256. CfC pays log2 of it
#              in scan rounds at training time and NOTHING at inference, so every size gets
#              512 - the whole 10 s strip, exactly, in both directions.
CFC_STATE_DIM = {'2m': 1, '1m': 1, '100k': 1, '30k': 1}
CFC_HORIZON = 512

CFC_SIZES = {
    f"resucfc_{size[len('resumamba_'):]}": dict(
        spec, state_dim=CFC_STATE_DIM[size[len('resumamba_'):]],
        time_mixer='cfc', mixer_memory=CFC_HORIZON)
    for size, spec in SIZES.items()
}

# The resumamba size each CfC size mirrors - the teacher its distillation reads from.
TEACHER_OF = {f"resucfc_{s[len('resumamba_'):]}": s for s in SIZES}


def _cfc_builder(size):
    def build(in_channels=None, **kw):
        kwargs = dict(CFC_SIZES[size])
        kwargs['name'] = size
        kwargs.update(kw)
        return build_resumamba_seq2seq(in_channels=in_channels, **kwargs)
    build.__name__ = f"build_{size}"
    build.__doc__ = f"{size}: {CFC_SIZES[size]}"
    return build


def _builder(size):
    def build(in_channels=None, **kw):
        kwargs = dict(SIZES[size])
        kwargs['name'] = f"resumamba_seq2seq_{size[len('resumamba_'):]}"
        kwargs.update(kw)
        return build_resumamba_seq2seq(in_channels=in_channels, **kwargs)
    build.__name__ = f"build_{size}"
    build.__doc__ = f"{size}: {SIZES[size]}"
    return build


BUILDERS = {size: _builder(size) for size in SIZES}
CFC_BUILDERS = {size: _cfc_builder(size) for size in CFC_SIZES}
# A CfC size drops the whole FIR bank and gains six scalars per cell, so it lands well under
# the budget its name carries; the ceiling stays the same number so the two families are
# directly comparable on the one axis the name is about.
CFC_BUDGETS = {f"resucfc_{s[len('resumamba_'):]}": b for s, b in BUDGETS.items()}
