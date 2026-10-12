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

    def cost(self):
        """Per-output-step hardware cost once this layer is exported as a FIR filter.

        The comparison partner of CfC1D.cost(). A diagonal SSM is a *fixed* impulse response,
        so deployment is a depthwise FIR of `kernel_len` taps per channel per direction: the
        decay rates and oscillator frequencies vanish into the coefficients and never appear
        in silicon. What does appear is one multiply-accumulate and one coefficient per tap,
        and a delay line as deep as the kernel.

        `state_words` counts that delay line, so the number is directly comparable with the
        CfC figure even though one is history and the other is state.
        """
        channels = int(self.fwd[0].shape[0])
        dirs = 2 if self.bidirectional else 1
        taps = channels * self.kernel_len * dirs
        return {'channels': channels, 'state_dim': self.state_dim,
                'bidirectional': self.bidirectional, 'taps': taps,
                'params': int(sum(np.prod(v.shape) for v in self.fwd) * dirs),
                'coeff_words': taps,            # the materialised FIR, not the 4 scalars
                'mac_per_step': taps,
                'state_words': channels * self.kernel_len * dirs,
                'nonlin_per_step': {}}

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
class StopGradient(layers.Layer):
    """Identity forward, no gradient back - a decoder can read another head's output without
    the second loss re-shaping that head."""

    def call(self, x):
        return tf.stop_gradient(x)

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package=PKG)
class BeatProbability(layers.Layer):
    """(B, T, classes) softmax with 'none' first -> (B, T, 1) probability that a step is a
    beat: 1 - p(none). A registered layer rather than a Lambda so the model reloads."""

    def call(self, x):
        return 1.0 - x[..., :1]

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)


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




# ---------------------------------------------------------------------------
# CfC: the closed-form continuous-time replacement for the state-space path
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package=PKG)
class CfC1D(layers.Layer):
    """Closed-form continuous-time recurrence, depthwise, trained by a parallel scan.

    Drop-in for DiagSSM1D - same (B, T, C) in, same (B, T, C) out, same place in the block -
    but a fundamentally different implementation, and the reason is the hardware.

    **Why replace a diagonal SSM that already works.** DiagSSM1D is exact and trains fast,
    but what it exports to is a depthwise FIR filter `kernel_len` taps long. On an FPGA or an
    ASIC that is the expensive shape: at the 2M size (192 channels, 256 taps, both
    directions) one block costs 98,304 MACs per output step and needs a 256-deep delay line
    per channel - 98,304 words of coefficient ROM and 49,152 words of line buffer, before any
    of the other three blocks. The cost is in the *memory*, and it grows with the receptive
    field you want.

    A continuous-time cell pays for the same memory with STATE instead of TAPS. One scalar
    per channel, updated in place:

        rate_t = softplus(a*u_t + a0)                 the liquid time constant, from the input
        A_t    = exp(-rate_t * dt)                    closed form of the ODE over one step
        tgt_t  = tanh(b*u_t + b0)                     what the state relaxes toward
        h_t    = A_t*h_{t-1} + (1 - A_t)*tgt_t        exact solution of dh/dt = rate*(tgt - h)
        inst_t = tanh(c*u_t + c0)                     the instantaneous branch
        y_t    = A_t*inst_t + (1 - A_t)*h_t           the CfC interpolation

    That is 6 weights and ~8 MACs per channel per step against 256, and ONE state word
    against a 256-word line buffer. Nothing is unbounded: `A_t` lives in (0, 1), so the state
    is contractive by construction and cannot blow up however long the record is.

    **It also puts back the one thing the diagonal SSM gave away.** Mamba's selectivity is
    input-dependent dynamics; DiagSSM1D dropped it to get a closed-form kernel (its decay
    rates are fixed weights, the same on clean signal and inside an artefact). Here `rate_t`
    is computed FROM the input at every step, so the cell can hold its state through a noisy
    stretch and let it go at a QRS - selectivity, at the cost of six scalars.

    **Training: a parallel scan, not a loop.** `h_t = A_t*h_{t-1} + B_t` is a first-order
    linear recurrence, so the pairs (A, B) compose associatively - (A1,B1) then (A2,B2) is
    (A1*A2, A2*B1 + B2) - and the whole sequence resolves in ceil(log2(horizon)) vectorised
    rounds instead of T sequential ones. Nine rounds cover 512 steps. A tf.scan over 500
    steps would be ~50x slower to train and would not export; this is the same recurrence,
    evaluated in a different order, and `scan_matches_loop` in the tests proves the two agree.

    `horizon` is the hard bound on how far state can propagate, and it is the direct
    analogue of DiagSSM1D's `kernel_len`: with horizon >= T the scan is exact for the whole
    strip. Unlike kernel_len it costs nothing - log2 of it, in rounds, and no extra weights.

    `state_dim` > 1 gives each channel that many independent cells with a learned readout
    mix, the analogue of DiagSSM1D's bank of oscillators. It multiplies the hardware state
    and the activation memory, so it defaults low; the input-dependent rate is doing the work
    the bank used to do.
    """

    def __init__(self, state_dim=1, horizon=512, bidirectional=True, dt=1.0,
                 recompute=True, **kw):
        super().__init__(**kw)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.bidirectional = bool(bidirectional)
        self.dt = float(dt)
        # The scan writes two (B, T, C, N) tensors per round and there are log2(horizon)
        # rounds, so keeping them all for the backward pass costs ~18x the activation of the
        # layer's own output - at the 2M size, gigabytes per block. Recomputing the scan in
        # the backward pass instead stores only its inputs. It is the difference between
        # this family fitting on one card and not; turn it off only to profile.
        self.recompute = bool(recompute)

    # --- weights ------------------------------------------------------------------------
    def _add_bank(self, channels, prefix, seed):
        """One CfC bank: (rate, target, instant) affine maps plus the readout mix."""
        rng = np.random.default_rng(seed)
        n, shape = self.state_dim, (channels, self.state_dim)

        # Rate bias on a log-spaced grid of time constants, 2 .. horizon steps, exactly as
        # DiagSSM1D seeds its decay rates: a fresh layer already spans QRS width to several
        # R-R intervals, so training starts from useful dynamics instead of finding them.
        tau = np.exp(np.linspace(np.log(2.0), np.log(max(self.horizon, 4.0)), n))
        rate0 = np.log(np.expm1(1.0 / tau))                 # softplus(rate0) == 1/tau
        rate0 = np.tile(rate0[None, :], (channels, 1)) + rng.normal(0, 0.1, shape)

        def const(v):
            v = v.astype('float32')
            return lambda s, dtype=None: tf.constant(v, dtype=dtype)

        w = {}
        w['a'] = self.add_weight(shape=shape, name=f'{prefix}_a', trainable=True,
                                 initializer=const(rng.normal(0, 0.1, shape)))
        w['a0'] = self.add_weight(shape=shape, name=f'{prefix}_a0', trainable=True,
                                  initializer=const(rate0))
        w['b'] = self.add_weight(shape=shape, name=f'{prefix}_b', trainable=True,
                                 initializer=const(rng.normal(0, 1.0, shape)))
        w['b0'] = self.add_weight(shape=shape, name=f'{prefix}_b0', trainable=True,
                                  initializer='zeros')
        w['c'] = self.add_weight(shape=shape, name=f'{prefix}_c', trainable=True,
                                 initializer=const(rng.normal(0, 1.0, shape)))
        w['c0'] = self.add_weight(shape=shape, name=f'{prefix}_c0', trainable=True,
                                  initializer='zeros')
        w['mix'] = self.add_weight(shape=shape, name=f'{prefix}_mix', trainable=True,
                                   initializer=const(rng.normal(0, 1.0 / np.sqrt(n), shape)))
        return w

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.fwd = self._add_bank(channels, 'fwd', seed=0)
        self.bwd = self._add_bank(channels, 'bwd', seed=1) if self.bidirectional else None
        super().build(input_shape)

    # --- the recurrence -----------------------------------------------------------------
    def _coeffs(self, u, bank):
        """(A, B, inst) of shape (B, T, C, N) from the input (B, T, C)."""
        x = u[..., None]                                            # (B, T, C, 1)
        rate = tf.nn.softplus(bank['a'] * x + bank['a0'])           # > 0
        a = tf.exp(-rate * self.dt)                                 # in (0, 1)
        tgt = tf.tanh(bank['b'] * x + bank['b0'])
        inst = tf.tanh(bank['c'] * x + bank['c0'])
        return a, (1.0 - a) * tgt, inst

    @staticmethod
    def _shift(x, d, fill):
        """x[:, t] -> x[:, t-d] along time, the first d steps filled with `fill`."""
        pad = tf.fill(tf.concat([tf.shape(x)[:1], [d], tf.shape(x)[2:]], axis=0),
                      tf.constant(fill, x.dtype))
        return tf.concat([pad, x[:, :-d]], axis=1)

    def _scan(self, a, b):
        """h_t = a_t*h_{t-1} + b_t, resolved by an inclusive Hillis-Steele scan.

        The pair (a, b) is the affine map h -> a*h + b; composing the map at t-d with the one
        at t gives (a_{t-d}*a_t, a_t*b_{t-d} + b_t), which is associative, so doubling the
        stride log2(horizon) times leaves every position holding the composition of the whole
        prefix. h_0 = 0, so the b half of that composition IS h_t.
        """
        rounds = max(1, int(np.ceil(np.log2(max(self.horizon, 2)))))
        for i in range(rounds):
            d = 1 << i
            a_prev = self._shift(a, d, 1.0)                 # identity map outside the window
            b_prev = self._shift(b, d, 0.0)
            b = a * b_prev + b
            a = a * a_prev
        return b

    def _direction(self, u, bank):
        a, b, inst = self._coeffs(u, bank)
        # _scan closes over no variables, only over `horizon`, so it is safe to wrap: the
        # recomputation is a pure function of (a, b).
        scan = tf.recompute_grad(self._scan) if self.recompute else self._scan
        h = scan(a, b)                                              # (B, T, C, N)
        y = a * inst + (1.0 - a) * h                                # CfC interpolation
        return tf.reduce_sum(y * bank['mix'], axis=-1)               # (B, T, C)

    def call(self, inputs):
        y = self._direction(inputs, self.fwd)
        if self.bidirectional:
            rev = tf.reverse(inputs, axis=[1])
            y += tf.reverse(self._direction(rev, self.bwd), axis=[1])
        return y

    # --- what an export / a test needs --------------------------------------------------
    def step_numpy(self, u_t, h, direction='fwd'):
        """One hardware step in numpy: (u_t, h) -> (y_t, h_next), both (C, N) for h.

        This IS the deployed arithmetic - three multiply-adds, a softplus, an exp and two
        tanh per cell, no history. An RTL or C port follows it line for line, and
        `tests/test_models.py` checks the parallel scan used in training reproduces it.
        """
        w = self.fwd if direction == 'fwd' else self.bwd
        g = {k: np.asarray(v) for k, v in w.items()}
        x = np.asarray(u_t)[..., None]
        rate = np.logaddexp(0.0, g['a'] * x + g['a0'])               # softplus
        a = np.exp(-rate * self.dt)
        tgt = np.tanh(g['b'] * x + g['b0'])
        inst = np.tanh(g['c'] * x + g['c0'])
        h_next = a * h + (1.0 - a) * tgt
        y = a * inst + (1.0 - a) * h_next
        return (y * g['mix']).sum(axis=-1), h_next

    def cost(self):
        """Per-output-step hardware cost of this layer, for the VLSI comparison.

        `state_words` is the whole runtime memory: a CfC keeps one word per cell and no
        history at all, which is the point of the swap.
        """
        channels = int(self.fwd['a'].shape[0])
        dirs = 2 if self.bidirectional else 1
        cells = channels * self.state_dim * dirs
        params = int(sum(np.prod(v.shape) for v in self.fwd.values()) * dirs)
        return {'channels': channels, 'state_dim': self.state_dim,
                'bidirectional': self.bidirectional, 'cells': cells,
                'params': params,
                # Every trained weight has to sit in ROM here too - there is no FIR to bake
                # them into, so coeff_words IS the parameter count. Reporting 0 would make
                # the comparison against the SSM's materialised kernel look free when it is
                # only small.
                'coeff_words': params,
                'mac_per_step': 8 * cells,      # 3 affine + 2 interp + mix, per cell
                'state_words': cells,
                'nonlin_per_step': {'softplus': cells, 'exp': cells, 'tanh': 2 * cells}}

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), 'state_dim': self.state_dim, 'horizon': self.horizon,
                'bidirectional': self.bidirectional, 'dt': self.dt,
                'recompute': self.recompute}


def cfc_block(x, filters, state_dim, horizon, bidirectional=True, dt=1.0,
              recompute=True, name=None):
    """The ssm_block layout with CfC1D in place of DiagSSM1D.

    Deliberately the same block around it - 1x1 in, depthwise 3, SiLU, the time mixer, a SiLU
    gate from the input, 1x1 out, residual - so a `cfc` model differs from the `ssm` model it
    is distilled from in exactly one layer per block and nothing else. Any difference in the
    EC57 tables is then attributable to the time mixer rather than to a second change riding
    along with it.
    """
    tag = (lambda s: None if name is None else f'{name}_{s}')
    u = layers.Conv1D(filters, 1, use_bias=False, name=tag('in_proj'))(x)
    u = layers.DepthwiseConv1D(3, padding='same', use_bias=False, name=tag('dw'))(u)
    u = layers.BatchNormalization(name=tag('bn'))(u)
    u = layers.Activation('silu', name=tag('silu'))(u)
    u = CfC1D(state_dim=state_dim, horizon=horizon, bidirectional=bidirectional, dt=dt,
              recompute=recompute, name=tag('cfc'))(u)

    g = layers.Conv1D(filters, 1, use_bias=False, name=tag('gate_proj'))(x)
    g = layers.Activation('silu', name=tag('gate'))(g)

    y = layers.Multiply(name=tag('mul'))([u, g])
    y = layers.Conv1D(filters, 1, use_bias=False, name=tag('out_proj'))(y)
    y = layers.BatchNormalization(name=tag('out_bn'))(y)
    if x.shape[-1] == filters:
        y = layers.Add(name=tag('res'))([x, y])
    return layers.Activation('relu', name=tag('out_relu'))(y)


# Time mixers the model family can be built with. `ssm` is the original diagonal state space
# (a long FIR once exported); `cfc` is the closed-form continuous-time cell above (a scalar
# state). The signature is shared - (x, filters, state_dim, memory, bidirectional, name) -
# so build_backbone picks one by name and nothing else in the graph changes.
TIME_MIXERS = {'ssm': ssm_block, 'cfc': cfc_block}
