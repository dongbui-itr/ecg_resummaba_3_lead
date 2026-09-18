"""The model family: the ResUMamba paper adapted to the seq2seq beat contract, in four sizes."""
from .resumamba import (BUDGETS, BUILDERS, SIZES, build_backbone, build_context_encoder,
                        build_resumamba_seq2seq)
from . import layers  # noqa: F401  - registers the custom layers for load_model()


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


__all__ = ['BUDGETS', 'BUILDERS', 'SIZES', 'build', 'build_backbone',
           'build_context_encoder', 'build_resumamba_seq2seq', 'keras_name', 'layers',
           'list_models', 'sub_model']
