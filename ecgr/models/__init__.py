"""The model family: the ResUMamba paper adapted to the seq2seq beat contract, in four sizes."""
from .resumamba import (BUDGETS, BUILDERS, SIZES, build_backbone, build_context_encoder,
                        build_resumamba_seq2seq)
from . import layers, refine  # noqa: F401  - both register custom layers for load_model()


def build(name, **kw):
    """Instantiate one of BUILDERS by name, with the project config."""
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}")
    return BUILDERS[name](**kw)


def list_models():
    """The family, largest first - the order the README and the launcher use."""
    return sorted(BUILDERS, key=lambda n: -BUDGETS[n])


def keras_name(name):
    """model.name of a builder, without building it - it names the output subfolders.

    The prefix is 'resumamba_seq2seq', the same string layers.PKG registers the custom
    layers under, so the name in a .keras file and the folder it lives in agree.
    """
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}")
    return f"resumamba_seq2seq_{name[len('resumamba_'):]}"


def sub_model(model, name):
    """The nested `backbone` / `context_encoder` sub-model of a built model.

    Both are pretrained without labels and loaded back by name, so every stage that touches
    them goes through this one lookup instead of indexing into model.layers.
    """
    found = next((l for l in model.layers if l.name == name), None)
    if found is None:
        raise ValueError(f"{model.name} has no {name!r} sub-model - it was built with "
                         f"use_context=False or an older layout")
    return found


def output_names(model):
    """Names of a model's outputs, i.e. of the layers that produce them ('beat_cls', ...)."""
    names = getattr(model, 'output_names', None)
    if names:
        return list(names)
    return [getattr(getattr(o, '_keras_history', None), 'operation', o).name
            for o in model.outputs]


def has_quality_output(model):
    """True when `model` emits [beat_cls, lead_quality] rather than beat_cls alone."""
    outputs = getattr(model, 'outputs', None)
    return bool(outputs) and len(outputs) >= 2


def split_outputs(outputs):
    """(beats, quality-or-None) from whatever a model call returned.

    A two-output model returns a list, a legacy one a single tensor; every consumer that
    only wants the beat softmax - decoding, metrics, ensembles - goes through here so the two
    layouts are interchangeable.
    """
    if isinstance(outputs, dict):
        return outputs['beat_cls'], outputs.get('lead_quality')
    if isinstance(outputs, (list, tuple)):
        return outputs[0], (outputs[1] if len(outputs) > 1 else None)
    return outputs, None


__all__ = ['BUDGETS', 'BUILDERS', 'SIZES', 'build', 'build_backbone',
           'build_context_encoder', 'build_resumamba_seq2seq', 'has_quality_output',
           'keras_name', 'layers', 'list_models', 'output_names', 'split_outputs', 'sub_model']
