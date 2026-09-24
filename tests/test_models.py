"""The four sizes: budgets, the two-output contract, the nested sub-models, and save/load."""
import keras
import numpy as np
import pytest

from ecgr import config, models

SIZES = models.list_models()
SMALL = 'resumamba_100k'          # the cheap size every other test builds


@pytest.fixture(scope='module')
def built():
    """Every size, built once - each one costs a few seconds."""
    out = {}
    for name in SIZES:
        out[name] = models.build(name)
    return out


def _x(n=2, seed=0):
    return np.random.default_rng(seed).standard_normal(
        (n, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')


@pytest.mark.parametrize('name', SIZES)
def test_the_family_is_the_four_promised_sizes(name, built):
    assert set(SIZES) == {'resumamba_5m', 'resumamba_3m', 'resumamba_1m', 'resumamba_100k'}
    assert models.BUDGETS == {'resumamba_5m': 5_000_000, 'resumamba_3m': 3_000_000,
                              'resumamba_1m': 1_000_000, 'resumamba_100k': 100_000}


@pytest.mark.parametrize('name', SIZES)
def test_parameter_budget(name, built):
    total = built[name].count_params()
    budget = models.BUDGETS[name]
    assert total < budget, f"{name} has {total:,} params, over its {budget:,} budget"
    # and not absurdly under it either - a size that drifts far below its name is a
    # different model from the one the family claims to offer
    assert total > 0.7 * budget, f"{name} has {total:,} params, far under its {budget:,}"


@pytest.mark.parametrize('name', SIZES)
def test_input_and_two_output_contract(name, built):
    """60 s x 3 leads in; beat softmax over the 20 ms grid AND a per-lead quality out."""
    model = built[name]
    assert tuple(model.input_shape[1:]) == (config.SEGMENT_SAMPLES, config.IN_CHANNELS)
    assert config.SEGMENT_SAMPLES == 60 * 250 and config.OUTPUT_STEPS == 3000
    assert len(model.outputs) == 2
    beats, quality = model.outputs
    assert models.output_names(model) == ['beat_cls', 'lead_quality']
    assert tuple(beats.shape[1:]) == (config.OUTPUT_STEPS, config.NUM_CLASSES)
    assert tuple(quality.shape[1:]) == (config.OUTPUT_STEPS, config.IN_CHANNELS)
    assert models.has_quality_output(model)


@pytest.mark.parametrize('name', SIZES)
def test_outputs_are_a_distribution_and_a_probability(name, built):
    beats, quality = models.split_outputs(built[name].predict(_x(), verbose=0))
    assert np.allclose(beats.sum(axis=-1), 1.0, atol=1e-4)
    assert (beats >= 0).all()
    assert quality.shape == (2, config.OUTPUT_STEPS, config.IN_CHANNELS)
    assert (quality >= 0).all() and (quality <= 1).all()


@pytest.mark.parametrize('name', SIZES)
def test_both_pretrainable_sub_models_exist(name, built):
    model = built[name]
    backbone = models.sub_model(model, 'backbone')
    context = models.sub_model(model, 'context_encoder')
    # the backbone is where the budget is: it is the thing worth pretraining
    assert backbone.count_params() > 0.4 * model.count_params()
    assert tuple(backbone.output_shape[1:2]) == (config.OUTPUT_STEPS,)
    assert len(context.outputs) == 3, "CPC needs (f_p, v_seq, c_seq)"
    # 60 s cut into 2 s windows at 50% overlap: the paper's own M for its 60 s calibration
    assert context.outputs[1].shape[1] == 59


def test_the_legacy_single_output_layout_is_still_buildable():
    """use_quality=False gives the 10 s family's one-output model, and every consumer that
    goes through split_outputs reads both layouts alike."""
    model = models.build(SMALL, use_quality=False)
    assert len(model.outputs) == 1 and not models.has_quality_output(model)
    beats, quality = models.split_outputs(model.predict(_x(1), verbose=0))
    assert beats.shape == (1, config.OUTPUT_STEPS, config.NUM_CLASSES) and quality is None
    assert models.split_outputs({'beat_cls': 1, 'lead_quality': 2}) == (1, 2)
    assert models.split_outputs([1, 2]) == (1, 2) and models.split_outputs(7) == (7, None)


def test_a_duplicated_lead_is_read_like_a_single_lead_record():
    """The EC57 path can feed one lead three times; that must be a valid input, not a shape
    error, and the rhythm descriptor must see the same thing either way."""
    model = models.build(SMALL)
    one = np.random.default_rng(1).standard_normal((1, config.SEGMENT_SAMPLES, 1)).astype('float32')
    tripled = np.repeat(one, config.IN_CHANNELS, axis=-1)
    beats, quality = models.split_outputs(model.predict(tripled, verbose=0))
    assert beats.shape == (1, config.OUTPUT_STEPS, config.NUM_CLASSES)
    assert np.isfinite(beats).all() and np.isfinite(quality).all()


def test_rhythm_descriptor_reads_lead_zero_only():
    """Changing leads 1 and 2 must not change the rhythm descriptor: that invariance is what
    makes the descriptor comparable between a 3-lead strip and a filled EC57 record."""
    from ecgr.models.layers import RhythmDescriptor
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 3000, 3)).astype('float32')
    b = a.copy()
    b[:, :, 1:] = rng.standard_normal((2, 3000, 2)).astype('float32')
    layer = RhythmDescriptor(step_hz=50.0, channel=0)
    assert np.allclose(layer(a).numpy(), layer(b).numpy(), atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    """A saved model must come back with identical weights - the custom layers, the AdaIN
    Dense sublayers built in build(), the two nested sub-models and the quality head all have
    to survive - and with both outputs."""
    model = models.build(SMALL)
    path = str(tmp_path / 'm.keras')
    model.save(path)
    reloaded = keras.models.load_model(path, compile=False)

    assert reloaded.count_params() == model.count_params()
    assert len(reloaded.outputs) == 2
    before = {v.path: v.numpy() for v in model.weights}
    after = {v.path: v.numpy() for v in reloaded.weights}
    assert set(before) == set(after)
    for key in before:
        assert np.array_equal(before[key], after[key]), f"{key} did not survive the round trip"


def test_diag_ssm_kernel_is_exportable():
    """The state-space layer is a fixed FIR filter once trained; kernel_numpy() is what an
    export script needs to replace it with a plain DepthwiseConv1D. Kernels of 1024 steps
    (the 5m size) cost no more parameters than kernels of 64."""
    from ecgr.models.layers import DiagSSM1D
    for length in (64, 1024):
        layer = DiagSSM1D(state_dim=6, kernel_len=length, bidirectional=True)
        layer.build((None, 3000, 8))
        kernels = layer.kernel_numpy()
        assert kernels.shape == (2, 8, length)
        assert np.isfinite(kernels).all()
        # normalize=True keeps the L1 norm at 1, so the layer cannot inflate the activations
        assert np.allclose(np.abs(kernels).sum(axis=-1), 1.0, atol=1e-3)
        assert layer.count_params() == 2 * 4 * 8 * 6


def test_every_batchnorm_tracks_fast():
    """BN momentum 0.9 everywhere: with Keras's 0.99 the 5m size's inference-mode SSL
    reconstruction diverged (val_nmse 187 vs train 0.74) because the running statistics
    lagged ~100 steps and the lag compounded over five residual SSM blocks."""
    from ecgr.models.layers import BN_MOMENTUM
    from ecgr.models.refine import attach_refinement
    assert BN_MOMENTUM <= 0.9
    for model in (models.build('resumamba_5m'), attach_refinement(models.build(SMALL))):
        stack = [model]
        while stack:
            layer = stack.pop()
            if isinstance(layer, keras.Model):
                stack.extend(layer.layers)
            elif isinstance(layer, keras.layers.BatchNormalization):
                assert layer.momentum == BN_MOMENTUM, f"{layer.name} has momentum {layer.momentum}"


def test_ssm_blocks_normalise_identically_in_both_modes():
    """The gated residual SSM stack must not depend on BatchNorm's running statistics: with
    BN the 5m size trained normally while its inference-mode output diverged to 1e9 (the
    train/inference mismatch is multiplied by the gate, block after block). LayerNorm makes
    training=True and training=False bit-for-bit the same computation."""
    from ecgr.models.layers import ssm_block
    inp = keras.Input(shape=(300, 32))
    h = inp
    for i in range(5):
        h = ssm_block(h, 32, 8, 64, name=f'b{i}')
    model = keras.Model(inp, h)
    assert not any(isinstance(l, keras.layers.BatchNormalization) for l in model.layers)
    assert sum(isinstance(l, keras.layers.LayerNormalization) for l in model.layers) == 10
    # perturb the weights hard, as training would, then compare the two modes
    rng = np.random.default_rng(0)
    for w in model.trainable_weights:
        if 'ln' in w.path or 'gamma' in w.path or 'beta' in w.path:
            w.assign(rng.uniform(0.5, 3.0, w.shape).astype('float32'))
        elif w.ndim >= 2:
            w.assign(rng.normal(0, 0.5, w.shape).astype('float32'))
    x = rng.standard_normal((4, 300, 32)).astype('float32') * 20
    train_out = model(x, training=True).numpy()
    infer_out = model(x, training=False).numpy()
    assert np.isfinite(infer_out).all()
    assert np.array_equal(train_out, infer_out)


def test_ssm_kernels_scale_with_the_window():
    """At 60 s the state-space kernels read 8-20 s each way; that is the rhythm context the
    S/N decision in atrial fibrillation needs, and it is what the longer window buys."""
    for name, size in models.SIZES.items():
        seconds = size['kernel_len'] * config.STEP_SAMPLES / config.SAMPLING_RATE
        assert 7.0 <= seconds <= 21.0, f"{name}: {seconds:.1f} s kernel"


# --- the temporal refinement head -------------------------------------------------------

def test_refinement_head_starts_as_the_identity_and_keeps_p_none():
    """models/refine.py promises two things the no-regression selection rule stands on:
    epoch 0 IS the base, and the head can never move probability into or out of None. The
    lead-quality output passes through untouched."""
    from ecgr.models.refine import attach_refinement, head_parameters
    base = models.build(SMALL)
    refined = attach_refinement(base)
    x = _x(3)
    b_base, q_base = models.split_outputs(base.predict(x, verbose=0))
    b_ref, q_ref = models.split_outputs(refined.predict(x, verbose=0))
    assert np.abs(b_base - b_ref).max() < 1e-5, "zero-initialised head must be the identity"
    assert np.array_equal(q_base, q_ref), "lead quality must pass through the head verbatim"
    assert models.output_names(refined) == ['beat_cls', 'lead_quality']
    assert all('refine' in w.path for w in refined.trainable_weights), "base must be frozen"
    assert 0 < head_parameters(refined) < 0.1 * base.count_params() + 30_000

    # kick the head hard: the beat classes move, the background does not, rows still sum to 1
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.random.default_rng(1).normal(0, 1, w.shape).astype('float32'))
    p2 = models.split_outputs(refined.predict(x, verbose=0))[0]
    assert np.array_equal(p2[..., 0], b_base[..., 0]), "p_None must be preserved exactly"
    assert np.abs(p2[..., 1:] - b_base[..., 1:]).max() > 1e-3
    assert np.allclose(p2.sum(-1), 1.0, atol=1e-5)


def test_refined_model_round_trips_through_keras(tmp_path):
    from ecgr.models.refine import attach_refinement
    refined = attach_refinement(models.build(SMALL))
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.full(w.shape, 0.3, dtype='float32'))
    path = str(tmp_path / 'r.keras')
    refined.save(path)
    again = keras.models.load_model(path, compile=False)
    x = _x(2, seed=2)
    for a, b in zip(refined.predict(x, verbose=0), again.predict(x, verbose=0)):
        assert np.array_equal(a, b)


def test_s_only_refinement_keeps_p_none_and_p_v_exactly():
    """'s_only' may move mass between N and S and nothing else: p_None and p_V verbatim,
    rows still sum to one, identity at init."""
    from ecgr.models.refine import attach_refinement
    base = models.build(SMALL)
    refined = attach_refinement(base, mode='s_only')
    x = _x(3, seed=3)
    p_base = models.split_outputs(base.predict(x, verbose=0))[0]
    assert np.abs(models.split_outputs(refined.predict(x, verbose=0))[0] - p_base).max() < 1e-5
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.random.default_rng(4).normal(0, 2, w.shape).astype('float32'))
    p2 = models.split_outputs(refined.predict(x, verbose=0))[0]
    assert np.array_equal(p2[..., 0], p_base[..., 0]), "p_None must not move"
    assert np.array_equal(p2[..., 2], p_base[..., 2]), "p_V must not move in s_only mode"
    assert np.allclose(p2[..., 1] + p2[..., 3], p_base[..., 1] + p_base[..., 3], atol=1e-6)
    assert np.abs(p2[..., 3] - p_base[..., 3]).max() > 1e-3, "S must be able to move"
    assert np.allclose(p2.sum(-1), 1.0, atol=1e-5)
