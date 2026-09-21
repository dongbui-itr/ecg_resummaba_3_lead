"""The four sizes: budgets, contract, the nested sub-models, and save/load."""
import keras
import numpy as np
import pytest

from ecgr import config, models

SIZES = models.list_models()


@pytest.fixture(scope='module')
def built():
    """Every size, built once - each one costs a few seconds."""
    out = {}
    for name in SIZES:
        out[name] = models.build(name)
    return out


@pytest.mark.parametrize('name', SIZES)
def test_the_family_is_the_four_promised_sizes(name, built):
    assert set(SIZES) == {'resumamba_2m', 'resumamba_1m', 'resumamba_100k', 'resumamba_30k'}


@pytest.mark.parametrize('name', SIZES)
def test_parameter_budget(name, built):
    total = built[name].count_params()
    budget = models.BUDGETS[name]
    assert total < budget, f"{name} has {total:,} params, over its {budget:,} budget"
    # and not absurdly under it either - a size that drifts far below its name is a
    # different model from the one the family claims to offer
    assert total > 0.7 * budget, f"{name} has {total:,} params, far under its {budget:,}"


@pytest.mark.parametrize('name', SIZES)
def test_input_output_contract(name, built):
    model = built[name]
    assert tuple(model.input_shape[1:]) == (config.SEGMENT_SAMPLES, config.IN_CHANNELS)
    assert tuple(model.output_shape[1:]) == (config.OUTPUT_STEPS, config.NUM_CLASSES)


@pytest.mark.parametrize('name', SIZES)
def test_output_is_a_distribution(name, built):
    x = np.random.default_rng(0).standard_normal(
        (2, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    y = built[name].predict(x, verbose=0)
    assert np.allclose(y.sum(axis=-1), 1.0, atol=1e-4)
    assert (y >= 0).all()


@pytest.mark.parametrize('name', SIZES)
def test_both_pretrainable_sub_models_exist(name, built):
    model = built[name]
    backbone = models.sub_model(model, 'backbone')
    context = models.sub_model(model, 'context_encoder')
    # the backbone is where the budget is: it is the thing worth pretraining
    assert backbone.count_params() > 0.4 * model.count_params()
    assert tuple(backbone.output_shape[1:2]) == (config.OUTPUT_STEPS,)
    assert len(context.outputs) == 3, "CPC needs (f_p, v_seq, c_seq)"


def test_a_duplicated_lead_is_read_like_a_single_lead_record():
    """The EC57 path feeds one lead three times; that must be a valid input, not a shape
    error, and the rhythm descriptor must see the same thing either way."""
    model = models.build('resumamba_30k')
    rng = np.random.default_rng(1)
    one = rng.standard_normal((1, config.SEGMENT_SAMPLES, 1)).astype('float32')
    tripled = np.repeat(one, config.IN_CHANNELS, axis=-1)
    y = model.predict(tripled, verbose=0)
    assert y.shape == (1, config.OUTPUT_STEPS, config.NUM_CLASSES)
    assert np.isfinite(y).all()


def test_rhythm_descriptor_reads_lead_zero_only():
    """Changing leads 1 and 2 must not change the rhythm descriptor: that invariance is what
    makes the descriptor comparable between a 3-lead strip and a duplicated EC57 record."""
    from ecgr.models.layers import RhythmDescriptor
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 500, 3)).astype('float32')
    b = a.copy()
    b[:, :, 1:] = rng.standard_normal((2, 500, 2)).astype('float32')
    layer = RhythmDescriptor(step_hz=50.0, channel=0)
    assert np.allclose(layer(a).numpy(), layer(b).numpy(), atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    """A saved model must come back with identical weights - the custom layers, the AdaIN
    Dense sublayers built in build(), and the two nested sub-models all have to survive."""
    model = models.build('resumamba_30k')
    path = str(tmp_path / 'm.keras')
    model.save(path)
    reloaded = keras.models.load_model(path, compile=False)

    assert reloaded.count_params() == model.count_params()
    before = {v.path: v.numpy() for v in model.weights}
    after = {v.path: v.numpy() for v in reloaded.weights}
    assert set(before) == set(after)
    for key in before:
        assert np.array_equal(before[key], after[key]), f"{key} did not survive the round trip"


def test_diag_ssm_kernel_is_exportable():
    """The state-space layer is a fixed FIR filter once trained; kernel_numpy() is what an
    export script needs to replace it with a plain DepthwiseConv1D."""
    from ecgr.models.layers import DiagSSM1D
    layer = DiagSSM1D(state_dim=6, kernel_len=64, bidirectional=True)
    layer.build((None, 500, 8))
    kernels = layer.kernel_numpy()
    assert kernels.shape == (2, 8, 64)
    assert np.isfinite(kernels).all()
    # normalize=True keeps the L1 norm at 1, so the layer cannot inflate the activations
    assert np.allclose(np.abs(kernels).sum(axis=-1), 1.0, atol=1e-3)


# --- the temporal refinement head -------------------------------------------------------

def test_refinement_head_starts_as_the_identity_and_keeps_p_none():
    """models/refine.py promises two things the no-regression selection rule stands on:
    epoch 0 IS the base, and the head can never move probability into or out of None."""
    from ecgr.models.refine import attach_refinement, head_parameters
    base = models.build('resumamba_30k')
    refined = attach_refinement(base)
    x = np.random.default_rng(0).standard_normal(
        (3, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    p_base, p_ref = base.predict(x, verbose=0), refined.predict(x, verbose=0)
    assert np.abs(p_base - p_ref).max() < 1e-5, "zero-initialised head must be the identity"
    assert all('refine' in w.path for w in refined.trainable_weights), "base must be frozen"
    assert 0 < head_parameters(refined) < 0.1 * base.count_params() + 30_000

    # kick the head hard: the beat classes move, the background does not, rows still sum to 1
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.random.default_rng(1).normal(0, 1, w.shape).astype('float32'))
    p2 = refined.predict(x, verbose=0)
    assert np.array_equal(p2[..., 0], p_base[..., 0]), "p_None must be preserved exactly"
    assert np.abs(p2[..., 1:] - p_base[..., 1:]).max() > 1e-3
    assert np.allclose(p2.sum(-1), 1.0, atol=1e-5)


def test_refined_model_round_trips_through_keras(tmp_path):
    from ecgr.models.refine import attach_refinement
    refined = attach_refinement(models.build('resumamba_30k'))
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.full(w.shape, 0.3, dtype='float32'))
    path = str(tmp_path / 'r.keras')
    refined.save(path)
    again = keras.models.load_model(path, compile=False)
    x = np.random.default_rng(2).standard_normal(
        (2, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    assert np.array_equal(refined.predict(x, verbose=0), again.predict(x, verbose=0))


def test_s_only_refinement_keeps_p_none_and_p_v_exactly():
    """'s_only' may move mass between N and S and nothing else: p_None and p_V verbatim,
    rows still sum to one, identity at init."""
    from ecgr.models.refine import attach_refinement
    base = models.build('resumamba_30k')
    refined = attach_refinement(base, mode='s_only')
    x = np.random.default_rng(3).standard_normal(
        (3, config.SEGMENT_SAMPLES, config.IN_CHANNELS)).astype('float32')
    p_base = base.predict(x, verbose=0)
    assert np.abs(refined.predict(x, verbose=0) - p_base).max() < 1e-5
    for w in refined.get_layer('refine_delta').weights:
        w.assign(np.random.default_rng(4).normal(0, 2, w.shape).astype('float32'))
    p2 = refined.predict(x, verbose=0)
    assert np.array_equal(p2[..., 0], p_base[..., 0]), "p_None must not move"
    assert np.array_equal(p2[..., 2], p_base[..., 2]), "p_V must not move in s_only mode"
    assert np.allclose(p2[..., 1] + p2[..., 3], p_base[..., 1] + p_base[..., 3], atol=1e-6)
    assert np.abs(p2[..., 3] - p_base[..., 3]).max() > 1e-3, "S must be able to move"
    assert np.allclose(p2.sum(-1), 1.0, atol=1e-5)
