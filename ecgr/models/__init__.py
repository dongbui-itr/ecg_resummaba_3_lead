"""The model family: the ResUMamba paper adapted to the seq2seq beat contract, in four sizes."""
from .resumamba import (BUDGETS as SSM_BUDGETS, BUILDERS as SSM_BUILDERS,
                        CFC_BUDGETS, CFC_BUILDERS, CFC_SIZES, SIZES, TEACHER_OF,
                        build_backbone, build_context_encoder, build_resumamba_seq2seq)

BUILDERS = {**SSM_BUILDERS, **CFC_BUILDERS}
BUDGETS = {**SSM_BUDGETS, **CFC_BUDGETS}
# model.name of each builder, knowable before the model is built because it names every
# output folder. The resumamba family's prefix is historical and is also the string
# layers.PKG registers the custom layers under; the CfC family uses its own key.
KERAS_NAMES = {n: f"resumamba_seq2seq_{n[len('resumamba_'):]}" for n in SSM_BUILDERS}
KERAS_NAMES.update({n: n for n in CFC_BUILDERS})
from . import layers, refine  # noqa: F401  - both register custom layers for load_model()


def build(name, **kw):
    """Instantiate one of BUILDERS by name, with the project config."""
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}")
    return BUILDERS[name](**kw)


def list_models():
    """Both families, largest first - the order the README and the launcher use."""
    return sorted(BUILDERS, key=lambda n: (n.startswith('resucfc'), -BUDGETS[n]))


def keras_name(name):
    """model.name of a builder, without building it - it names the output subfolders."""
    if name not in KERAS_NAMES:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}")
    return KERAS_NAMES[name]


def teacher_of(name):
    """The resumamba size a CfC size is distilled from, or None for the ssm family."""
    return TEACHER_OF.get(name)


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


__all__ = ['BUDGETS', 'BUILDERS', 'CFC_SIZES', 'KERAS_NAMES', 'SIZES', 'TEACHER_OF',
           'build', 'build_backbone', 'build_context_encoder', 'build_resumamba_seq2seq',
           'keras_name', 'layers', 'list_models', 'sub_model', 'teacher_of']
