"""Custom layers of the ResUMamba family, plus the conv helpers they are built from.

Four of these are registered Keras layers, which means their names are part of a STORAGE
FORMAT, not just a namespace: Keras writes "<package>>ClassName" into every .keras file and
looks the class up by exactly that string. `PKG` therefore stays "resumamba_seq2seq" forever,
whatever this module ends up being called - renaming it to match the import path would orphan
every checkpoint already trained.
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers

PKG = "resumamba_seq2seq"


def _pool_plan(input_length, output_steps):
    """Factor input_length/output_steps into small pooling sizes (2500/500 -> [5])."""
    if output_steps < 1 or input_length % output_steps:
        raise ValueError(f"OUTPUT_STEPS ({output_steps}) must divide SEGMENT_SAMPLES "
                         f"({input_length}) exactly")
    stride, factors = input_length // output_steps, []
    for p in (2, 3, 5, 7):
        while stride % p == 0:
            factors.append(p)
            stride //= p
    if stride != 1:
        raise ValueError(f"stride {input_length // output_steps} has a prime factor > 7")
    return factors


def conv_bn_act(x, filters, kernel_size, separable=False, dilation_rate=1, name=None):
    conv = layers.SeparableConv1D if separable else layers.Conv1D
    x = conv(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
             use_bias=False, name=None if name is None else name + '_conv')(x)
    x = layers.BatchNormalization(name=None if name is None else name + '_bn')(x)
    return layers.Activation('relu', name=None if name is None else name + '_relu')(x)


def _match_length(x, target, name=None):
    """Crop or pad the time axis to `target` - upsampling 125 -> 62 -> 124 is off by one."""
    cur = x.shape[1]
    if cur is None or cur == target:
        return x
    if cur > target:
        return layers.Cropping1D((0, cur - target), name=name)(x)
    return layers.ZeroPadding1D((0, target - cur), name=name)(x)


def res_u_block(x, filters, mid, depth, separable=False, name=None):
    """ResU block of Hwang et al. (2023), the ResUNet path's unit.

    A small U-Net (depth levels down, then up with skip concatenations) wrapped in a residual
    connection. Each level halves the time axis, so one block sees 2^depth times further than a
    plain conv of the same kernel while keeping the output at full step resolution - that is
    what "multi-scale morphology" means here: level 0 sees the QRS, the deepest level sees the
    whole T-P segment around it.
    """
    tag = (lambda s: None if name is None else f'{name}_{s}')
    entry = conv_bn_act(x, filters, 3, separable, name=tag('in'))

    h, skips, lens = entry, [], []
    for i in range(depth):
        h = conv_bn_act(h, mid, 3, separable, name=tag(f'enc{i}'))
        skips.append(h)
        lens.append(h.shape[1])
        h = layers.MaxPooling1D(2, padding='same', name=tag(f'pool{i}'))(h)

    h = conv_bn_act(h, mid, 3, separable, dilation_rate=2, name=tag('mid'))

    for i in reversed(range(depth)):
        h = layers.UpSampling1D(2, name=tag(f'up{i}'))(h)
        h = _match_length(h, lens[i], name=tag(f'fit{i}'))
        h = layers.Concatenate(name=tag(f'skip{i}'))([h, skips[i]])
        h = conv_bn_act(h, mid, 3, separable, name=tag(f'dec{i}'))

    h = conv_bn_act(h, filters, 3, separable, name=tag('out'))
    return layers.Activation('relu', name=tag('res'))(layers.Add(name=tag('add'))([entry, h]))


@keras.saving.register_keras_serializable(package=PKG)
class DiagSSM1D(layers.Layer):
    """Diagonal state-space layer evaluated as a depthwise FIR convolution.

    A diagonal SSM h_t = a*h_{t-1} + b*u_t, y_t = c*h_t has the closed-form impulse response
    k_l = c*b*a^l, so the whole recurrence equals a convolution with that kernel. Writing
    a = exp(-rate + i*omega) and summing `state_dim` of them per channel gives

        k[c, l] = sum_n mix[c,n] * exp(-softplus(lam[c,n]) * l) * cos(omega[c,n]*l + phase[c,n])

    i.e. a learnable bank of decaying oscillators - 4 scalars per (channel, state) instead of
    the LSTM's 4 gates over full matrices. The decay rates are initialised on a log-spaced grid
    of time constants from 2 up to kernel_len steps (40 ms .. kernel_len*20 ms at this project's
    20 ms/step), so a fresh layer already covers everything from QRS width to several R-R
    intervals. Half the states start at omega=0 (pure low-pass, P/T-wave scale), half oscillate.

    bidirectional adds a second, independent kernel run over the reversed sequence and sums the
    two, which turns the pair into one two-sided FIR filter: a step can then use the beats after
    it as well as the ones before it. normalize divides each kernel by its L1 norm so the layer
    cannot blow up the activation scale as the kernels grow.
    """

    def __init__(self, state_dim=8, kernel_len=128, bidirectional=True, normalize=True, **kw):
        super().__init__(**kw)
        self.state_dim = int(state_dim)
        self.kernel_len = int(kernel_len)
        self.bidirectional = bool(bidirectional)
        self.normalize = bool(normalize)

    def _init_params(self, channels, seed):
        """(log_lam, omega, phase, mix) initial values, shape (channels, state_dim) each."""
        rng = np.random.default_rng(seed)
        n = self.state_dim
        tau = np.exp(np.linspace(np.log(2.0), np.log(max(self.kernel_len, 4.0)), n))
        rate = 1.0 / tau                              # decay per step
        log_lam = np.log(np.expm1(rate))              # softplus(log_lam) == rate
        log_lam = np.tile(log_lam[None, :], (channels, 1))
        log_lam += rng.normal(0.0, 0.1, log_lam.shape)

        omega = np.zeros((channels, n))
        half = n // 2
        if n - half > 0:                              # the oscillating half
            omega[:, half:] = np.linspace(0.05, 0.9, n - half)[None, :]
        omega += rng.normal(0.0, 0.02, omega.shape)

        phase = rng.uniform(-0.1, 0.1, (channels, n))
        mix = rng.normal(0.0, 1.0 / np.sqrt(n), (channels, n))
        return [v.astype('float32') for v in (log_lam, omega, phase, mix)]

    def _add_bank(self, channels, prefix, seed):
        values = self._init_params(channels, seed)
        names = ('log_lam', 'omega', 'phase', 'mix')
        bank = []
        for value, name in zip(values, names):
            bank.append(self.add_weight(
                shape=value.shape, name=f'{prefix}_{name}', trainable=True,
                initializer=(lambda v: (lambda shape, dtype=None: tf.constant(v, dtype=dtype)))(value)))
        return bank

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.fwd = self._add_bank(channels, 'fwd', seed=0)
        self.bwd = self._add_bank(channels, 'bwd', seed=1) if self.bidirectional else None
        super().build(input_shape)

    def _kernel(self, bank):
        log_lam, omega, phase, mix = bank
        l = tf.range(self.kernel_len, dtype=self.compute_dtype)          # (L,)
        decay = tf.exp(-tf.nn.softplus(log_lam)[..., None] * l)          # (C, N, L)
        osc = tf.cos(omega[..., None] * l + phase[..., None])            # (C, N, L)
        k = tf.reduce_sum(mix[..., None] * decay * osc, axis=1)          # (C, L)
        if self.normalize:
            k = k / (tf.reduce_sum(tf.abs(k), axis=-1, keepdims=True) + 1e-6)
        return k

    @staticmethod
    def _causal_conv(x, k):
        """y[t] = sum_l k[l] * x[t-l], per channel, via FFT.

        The obvious implementation is tf.nn.depthwise_conv2d with the kernel reversed, and it
        works forward - but the kernel here is COMPUTED from the weights, so training needs the
        gradient with respect to the filter, and cuDNN rejects that for a filter this wide
        (DepthwiseConv2dNativeBackpropFilter -> CUDNN_STATUS_BAD_PARAM at width 128+). The
        frequency domain has no such limit, is differentiable end to end, and costs
        O(n log n) instead of O(T*L): at T=500, L=192 it is roughly 30x fewer multiplies.

        For an embedded export the kernel is a constant, so it can be baked back into a plain
        DepthwiseConv1D - see kernel_numpy().
        """
        length = int(k.shape[-1])
        steps = tf.shape(x)[1]
        # The FFT length is computed from the RUNTIME length, not the static one: the training
        # pipeline's time-scale augmentation resizes with a dynamic factor, so the time axis
        # reaches this layer as None. Linear (not circular) convolution needs >= T + L - 1 bins.
        total = tf.cast(steps + length - 1, tf.float32)
        exponent = tf.cast(tf.math.ceil(tf.math.log(total) / tf.math.log(2.0)), tf.int32)
        n = tf.reshape(tf.bitwise.left_shift(1, exponent), [1])

        xt = tf.transpose(x, [0, 2, 1])                                  # (B, C, T)
        spec = tf.signal.rfft(xt, fft_length=n) * \
            tf.signal.rfft(k, fft_length=n)[None]                        # (B, C, n//2+1)
        y = tf.signal.irfft(spec, fft_length=n)[:, :, :steps]            # drop the tail
        return tf.transpose(y, [0, 2, 1])

    def kernel_numpy(self):
        """The materialised impulse responses, (2, C, L) if bidirectional else (1, C, L).

        The whole layer is a fixed FIR filter once training is over; this is what an export
        script needs to replace it with a DepthwiseConv1D (reverse each kernel in time: a conv
        layer cross-correlates).
        """
        banks = [self.fwd] + ([self.bwd] if self.bidirectional else [])
        return np.stack([self._kernel(b).numpy() for b in banks])

    def call(self, inputs):
        y = self._causal_conv(inputs, self._kernel(self.fwd))
        if self.bidirectional:
            rev = tf.reverse(inputs, axis=[1])
            y += tf.reverse(self._causal_conv(rev, self._kernel(self.bwd)), axis=[1])
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'state_dim': self.state_dim,
                'kernel_len': self.kernel_len, 'bidirectional': self.bidirectional,
                'normalize': self.normalize}


def ssm_block(x, filters, state_dim, kernel_len, bidirectional=True, name=None):
    """Mamba-style block (paper Fig. 2c) with the selective scan replaced by DiagSSM1D.

    Kept from the original: the two-branch layout - one branch carries the signal through a
    short conv and the state-space layer, the other produces a SiLU gate that decides, per
    channel and per step, how much of that long-range evidence to let through. That gate is the
    part of Mamba's selectivity that survives a non-selective SSM, and it is what lets the model
    ignore the state-space output inside a noisy stretch.
    """
    tag = (lambda s: None if name is None else f'{name}_{s}')
    u = layers.Conv1D(filters, 1, use_bias=False, name=tag('in_proj'))(x)
    u = layers.DepthwiseConv1D(3, padding='same', use_bias=False, name=tag('dw'))(u)
    u = layers.BatchNormalization(name=tag('bn'))(u)
    u = layers.Activation('silu', name=tag('silu'))(u)
    u = DiagSSM1D(state_dim=state_dim, kernel_len=kernel_len, bidirectional=bidirectional,
                  name=tag('ssm'))(u)

    g = layers.Conv1D(filters, 1, use_bias=False, name=tag('gate_proj'))(x)
    g = layers.Activation('silu', name=tag('gate'))(g)

    y = layers.Multiply(name=tag('mul'))([u, g])
    y = layers.Conv1D(filters, 1, use_bias=False, name=tag('out_proj'))(y)
    y = layers.BatchNormalization(name=tag('out_bn'))(y)
    if x.shape[-1] == filters:
        y = layers.Add(name=tag('res'))([x, y])
    return layers.Activation('relu', name=tag('out_relu'))(y)


@keras.saving.register_keras_serializable(package=PKG)
class AdaIN(layers.Layer):
    """Adaptive instance normalisation: normalise over time, then re-scale from the context.

    out[b,t,c] = gamma(f)[b,c] * (x[b,t,c] - mean_t) / std_t + beta(f)[b,c]

    The normalisation is the half that matters for inter-patient variability: it strips each
    strip's own amplitude and offset per channel BEFORE the conditioning is applied, which is
    exactly what the paper shows FiLM (affine only, no normalisation) fails to do. gamma is
    parameterised as 1 + dense(f) so an untrained layer starts as plain instance norm.
    """

    def __init__(self, epsilon=1e-5, **kw):
        super().__init__(**kw)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        feat_shape, ctx_shape = input_shape
        channels = int(feat_shape[-1])
        self.to_gamma = layers.Dense(channels, name='to_gamma')
        self.to_beta = layers.Dense(channels, name='to_beta')
        self.to_gamma.build(ctx_shape)
        self.to_beta.build(ctx_shape)
        super().build(input_shape)

    def call(self, inputs):
        x, ctx = inputs
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
        gamma = 1.0 + self.to_gamma(ctx)[:, None, :]
        beta = self.to_beta(ctx)[:, None, :]
        return x * gamma + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon}


@keras.saving.register_keras_serializable(package=PKG)
class Frame1D(layers.Layer):
    """(B, T, C) -> (B, M, window, C) sliding windows.

    A plain layer rather than a Lambda around tf.signal.frame: Keras refuses to deserialize a
    Lambda holding a Python lambda unless the caller passes safe_mode=False, and the EC57 script
    loads checkpoints with a bare load_model(..., compile=False).
    """

    def __init__(self, window, hop, **kw):
        super().__init__(**kw)
        self.window = int(window)
        self.hop = int(hop)

    def call(self, x):
        return tf.signal.frame(x, self.window, self.hop, axis=1)

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        n = None if t is None else (t - self.window) // self.hop + 1
        return (b, n, self.window, c)

    def get_config(self):
        return {**super().get_config(), 'window': self.window, 'hop': self.hop}


@keras.saving.register_keras_serializable(package=PKG)
class MergeWindows(layers.Layer):
    """(B, M, W, C) -> (B*M, W, C), so one conv net can embed every window in a single call.

    The obvious spelling is TimeDistributed, but it lowers to a pfor/while loop as soon as the
    time axis is dynamic (which the training pipeline's time-scale augmentation makes it), and
    the vectorised depthwise convolution inside that loop fails outright. Folding the window
    axis into the batch instead keeps it one flat convolution - faster as well as working.
    """

    def call(self, x):
        shape = tf.shape(x)
        return tf.reshape(x, [shape[0] * shape[1], shape[2], shape[3]])

    def compute_output_shape(self, input_shape):
        b, m, w, c = input_shape
        return (None if b is None or m is None else b * m, w, c)


@keras.saving.register_keras_serializable(package=PKG)
class SplitWindows(layers.Layer):
    """(B*M, D) -> (B, M, D), the inverse of MergeWindows for the embedding vectors."""

    def __init__(self, n_win, **kw):
        super().__init__(**kw)
        self.n_win = int(n_win)

    def call(self, x):
        return tf.reshape(x, [-1, self.n_win, tf.shape(x)[-1]])

    def compute_output_shape(self, input_shape):
        return (None, self.n_win, input_shape[-1])

    def get_config(self):
        return {**super().get_config(), 'n_win': self.n_win}


@keras.saving.register_keras_serializable(package=PKG)
class RhythmDescriptor(layers.Layer):
    """Beat-rate evidence without a beat detector: envelope autocorrelation over a lag band.

    The paper's clinical vector is built from R-R intervals, which this model cannot have -
    the beats are its output. The autocorrelation of the rectified, step-pooled signal carries
    the same rhythm information: its peak lag is the dominant R-R interval, its peak height is
    how regular the rhythm is (flat and low through AF, sharp through sinus), and the ratio
    between the first and second peaks separates bigeminy from a plain fast rhythm.

    Lags cover min_bpm..max_bpm at `step_hz` steps per second and are evaluated directly - 88
    lags over ~400 steps is ~35k MACs, so the smallest model can afford it too (an FFT would be
    both heavier here and unexportable).

    `channel` selects the lead the envelope is taken from, and the default 0 - the annotated
    lead - is deliberate rather than incidental. Pooling `max |x|` over all three leads is
    more informative on a 3-lead strip, but the EC57 stage hands this layer the same lead
    three times, where the max collapses to that one lead: the descriptor would then be
    computed from a different quantity at evaluation than at training. Reading one fixed
    lead is identical in both cases. Pass channel=None for the old max-over-leads behaviour.
    """

    def __init__(self, step_hz=50.0, min_bpm=30.0, max_bpm=220.0, channel=0, **kw):
        super().__init__(**kw)
        self.step_hz = float(step_hz)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.channel = None if channel is None else int(channel)

    def build(self, input_shape):
        self.lo = max(2, int(np.floor(60.0 / self.max_bpm * self.step_hz)))
        self.hi = int(np.ceil(60.0 / self.min_bpm * self.step_hz))
        # The time axis is None under the training pipeline's time-scale augmentation, so the
        # lag band is clamped against the length only when that length is actually known. The
        # NUMBER of lags stays static either way, which is what the Dense above it needs.
        length = input_shape[1]
        if length is not None:
            self.hi = min(self.hi, max(self.lo + 1, int(length) // 2))
        super().build(input_shape)

    def call(self, x):
        if self.channel is None:
            e = tf.reduce_max(tf.abs(x), axis=-1)
        else:
            e = tf.abs(x[..., min(self.channel, x.shape[-1] - 1)])
        e = e - tf.reduce_mean(e, axis=1, keepdims=True)

        span = tf.shape(e)[1] - self.hi
        base = e[:, :span]                                               # (B, span)
        lags = tf.range(self.lo, self.hi)
        idx = tf.range(span)[None, :] + lags[:, None]                    # (n_lags, span)
        shifted = tf.gather(e, idx, axis=1)                              # (B, n_lags, span)
        num = tf.reduce_mean(base[:, None, :] * shifted, axis=-1)
        den = tf.reduce_mean(tf.square(base), axis=-1, keepdims=True) + 1e-6
        return num / den                                                 # (B, n_lags)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hi - self.lo)

    def get_config(self):
        return {**super().get_config(), 'step_hz': self.step_hz,
                'min_bpm': self.min_bpm, 'max_bpm': self.max_bpm,
                'channel': self.channel}


