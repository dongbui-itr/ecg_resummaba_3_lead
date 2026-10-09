"""UNet-Mamba for rhythm, PER SAMPLE: the ResUMamba dual path with the ResU path replaced by
one full-depth U-Net, and an output at the input's own resolution, (2500, 6).

    input (2500, 3)                                           250 Hz
      stem   2 x conv k9                    (2500, stem)  -> max-pool 5
      enc1   2 x conv k7                    (500, f1)     -> max-pool 2          20 ms/step
      enc2   2 x conv k5                    (250, f2)     = shared trunk         40 ms
        |
        +-- U-Net path ------------------------------------------------------------------
        |     -> max-pool 5 -> enc3  2 x conv k5   (50, f3)   -> skip3           200 ms
        |     -> max-pool 5 -> bottom 2 x conv k3 (10, f4)                       1 s/step
        |     -> up 5, || skip3, 2 x conv          (50, f3)
        |     -> up 5, || enc2,  2 x conv          (250, width)
        |
        +-- Mamba path (as ResUMamba) -----------------------------------------------------
        |     -> conv 1 -> ssm_block x ssm_blocks  (250, ssm_channels)
        |
      concat -> conv 1 -> (250, width)                          = BACKBONE_STEPS
      [model.py: AdaIN + rhythm attention at 250 steps, shared with ResUMamba]
      sample head (sample_head below), the U-Net's upper half:
            up 2, || enc1, 2 x conv (500, f1) -> up 5, || stem, 2 x conv (2500, stem)
            -> conv 1 softmax -> 'rhythm' (2500, 6)

What changes against the ResU path: a ResU block is a shallow U-Net (2-3 levels of 2x pooling)
running at 40 ms with a residual around it, stacked twice. Here there is ONE U-Net whose encoder
starts at the full 250 Hz signal and whose bottleneck reaches one step per second, so the
morphology path itself sees the whole 10 s strip (R-R regularity, P-wave presence across
beats), and the skips bring the 40 ms and 200 ms detail back to the fused grid. The Mamba path
is unchanged: long-range context by bidirectional diagonal SSM on the same 250-step trunk.

The sub-model is named 'backbone'. Its first output is (BACKBONE_STEPS, width), like
build_backbone, so the conditioning and attention in model.build_rhythm_model are shared by
both families; its other two are the enc1 (500) and stem (2500) activations the sample head
climbs back up through. Training reads the per-sample labels (labels.sample_labels), and
predict/EC57 keep the whole-record probabilities at rc.SAMPLE_PROBS_HZ.
"""
import keras
from keras import layers

from ..models.layers import conv_bn_act, ssm_block
from . import config as rc

TRUNK_POOLS = (5, 2)          # 2500 -> 500 -> 250 (the fused grid)
UNET_POOLS = (5, 5)           # 250 -> 50 -> 10 (one step per second)


def _double_conv(x, filters, kernel, separable, name):
    x = conv_bn_act(x, filters, kernel, separable, name=f'{name}_a')
    return conv_bn_act(x, filters, kernel, separable, name=f'{name}_b')


def build_unet_backbone(input_length, in_channels, output_steps, width=64,
                        unet_filters=(32, 48, 64, 96), unet_kernels=(7, 5, 5, 3),
                        ssm_channels=64, ssm_blocks=2, state_dim=8, kernel_len=128,
                        separable=False, stem_channels=None, trunk_pools=TRUNK_POOLS,
                        name='backbone'):
    """(input_length, leads) -> [(output_steps, width), enc1, stem (2500, stem)];
    unet_filters = (f1, f2, f3, f4). trunk_pools (p0, p1) put enc1 at input_length / p0:
    (5, 2) -> enc1 at 500 steps (20 ms), (2, 5) -> 1250 (8 ms); the fused grid is 250 either
    way."""
    p0, p1 = trunk_pools
    trunk_steps = input_length // (p0 * p1)
    if input_length % (p0 * p1) or trunk_steps != output_steps or \
            output_steps % (UNET_POOLS[0] * UNET_POOLS[1]):
        raise ValueError(f"UNet-Mamba is laid out for {input_length} -> {output_steps} steps "
                         f"with pools {trunk_pools} + {UNET_POOLS}")
    f1, f2, f3, f4 = unet_filters
    k1, k2, k3, k4 = unet_kernels
    stem_channels = stem_channels or max(width // 4, 8)
    inp = keras.Input(shape=(input_length, in_channels), name='backbone_input')

    # --- shared trunk: full-rate stem, then down to the fused grid -------------------------
    # The stem stays dense in the separable sizes: on 3 input channels a separable conv saves
    # nothing and loses the cross-lead mixing.
    stem = _double_conv(inp, stem_channels, 9, False, 'stem')             # (2500, stem)
    x = layers.MaxPooling1D(p0, name='stem_pool')(stem)
    enc1 = _double_conv(x, f1, k1, separable, 'enc1')                     # (500|1250, f1)
    x = layers.MaxPooling1D(p1, name='enc1_pool')(enc1)
    trunk = _double_conv(x, f2, k2, separable, 'enc2')                    # (250, f2)

    # --- path 1: U-Net down to one step per second and back ------------------------------
    h = layers.MaxPooling1D(UNET_POOLS[0], name='enc2_pool')(trunk)
    skip3 = _double_conv(h, f3, k3, separable, 'enc3')                    # (50, f3)
    h = layers.MaxPooling1D(UNET_POOLS[1], name='enc3_pool')(skip3)
    h = _double_conv(h, f4, k4, separable, 'bottom')                      # (10, f4)
    h = layers.UpSampling1D(UNET_POOLS[1], name='dec3_up')(h)
    h = layers.Concatenate(name='dec3_skip')([h, skip3])
    h = _double_conv(h, f3, k3, separable, 'dec3')                        # (50, f3)
    h = layers.UpSampling1D(UNET_POOLS[0], name='dec2_up')(h)
    h = layers.Concatenate(name='dec2_skip')([h, trunk])
    y_u = _double_conv(h, width, k2, separable, 'dec2')                   # (250, width)

    # --- path 2: state-space, multi-beat context (unchanged from ResUMamba) --------------
    y_m = layers.Conv1D(ssm_channels, 1, use_bias=False, name='ssm_in')(trunk)
    for i in range(ssm_blocks):
        y_m = ssm_block(y_m, ssm_channels, state_dim, kernel_len, name=f'ssm{i}')

    y = layers.Concatenate(name='dual_concat')([y_u, y_m])
    y = conv_bn_act(y, width, 1, name='dual_fuse')
    return keras.Model(inp, [y, enc1, stem], name=name)


def beat_head(y, enc1, separable=False, dropout=0.15, kernel=7, width=None):
    """Beat decoder: (250, d) fused features + the enc1 skip -> 'beat' (enc1 steps, 4), a
    softmax none / N / S / V per step. The same climb as the rhythm head's first stage, with
    its own weights: beat morphology (QRS width, prematurity) is read at 8-20 ms."""
    width = width or enc1.shape[-1]
    h = layers.UpSampling1D(enc1.shape[1] // y.shape[1], name='beat_up')(y)
    h = layers.Concatenate(name='beat_skip')([h, enc1])
    h = _double_conv(h, width, kernel, separable, 'beat_dec')
    h = layers.Dropout(dropout, name='beat_drop')(h)
    return layers.Conv1D(len(rc.BEAT_CLASSES), 1, activation='softmax', name='beat')(h)


def sample_head(y, enc1, stem, num_classes, separable=False, dropout=0.15, kernels=(7, 9),
                steps=None):
    """(250, d) fused features + the two high-resolution skips -> (steps, num_classes).

    steps = 2500 (default) climbs all the way to the input rate; steps = enc1's length stops
    at enc1's level - 500 (20 ms, the rhythm_unet500 family) or 1250 (8 ms, rhythm_unet1250,
    trunk pools (2, 5)) - and never touches the stem skip. The up-sampling factors follow the
    lengths of the skips."""
    steps = steps or stem.shape[1]
    if steps not in (enc1.shape[1], stem.shape[1]):
        raise ValueError(f"sample_head emits {enc1.shape[1]} or {stem.shape[1]} steps, "
                         f"not {steps}")
    f1, f0 = enc1.shape[-1], stem.shape[-1]
    h = layers.UpSampling1D(enc1.shape[1] // y.shape[1], name='dec1_up')(y)
    h = layers.Concatenate(name='dec1_skip')([h, enc1])
    h = _double_conv(h, f1, kernels[0], separable, 'dec1')                # (enc1, f1)
    if steps == enc1.shape[1]:
        h = layers.Dropout(dropout, name='head_drop')(h)
        return layers.Conv1D(num_classes, 1, activation='softmax', name='rhythm')(h)
    h = layers.UpSampling1D(stem.shape[1] // enc1.shape[1], name='dec0_up')(h)
    h = layers.Concatenate(name='dec0_skip')([h, stem])
    h = _double_conv(h, f0, kernels[1], False, 'dec0')                    # (2500, stem)
    h = layers.Dropout(dropout, name='head_drop')(h)
    return layers.Conv1D(num_classes, 1, activation='softmax', name='rhythm')(h)
