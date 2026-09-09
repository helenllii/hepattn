"""Shared helpers for the Keras/HGQ2 <-> Lightning integration tests.

These cover the three integration concerns that live between keras and torch and are
not visible from either side alone: which tensors an optimizer receives, where the
keras Variables physically are, and what a checkpoint contains.
"""

from contextlib import contextmanager

from test_maskformer_parity import clic_dummy_batch  # ty: ignore [unresolved-import]

from hepattn.keras import get_keras_default_device, set_keras_default_device


def materialize(model, num_events: int = 2):
    """Build the lazy quantized layers, as MPflowHGQ.setup does before the optimizer exists.

    HGQ2 layers size their bitwidth variables from the first real batch's static shapes,
    so nothing about parameter coverage, device placement or state_dict completeness is
    even well defined until a batch has been through the model.
    """
    import torch  # noqa: PLC0415

    inputs, _ = clic_dummy_batch(num_events)
    with torch.no_grad():
        model.eval()(inputs)
    return model.train()


def expected_groups(model):
    """The trainable tensors the model SHOULD hold, derived from the model itself.

    Union of the keras Variables (which `named_parameters()` cannot see on its own) and
    the plain torch parameters (norms, register tokens, queries, linformer projections),
    keyed by object identity, minus the non-trainable ones and beta. Returned as
    (network, quantizer) so a test can compare against sets rather than hardcoded counts.
    """
    network, quantizer = {}, {}
    for layer in model.keras_layers():
        for var in layer.weights:
            if var.trainable and not var.path.endswith("/beta"):
                (quantizer if "quantizer" in var.path else network)[id(var.value)] = var.value
    keras_ids = {id(v.value) for layer in model.keras_layers() for v in layer.weights}
    for name, param in model.named_parameters():
        if param.requires_grad and not name.endswith("/beta") and id(param) not in keras_ids:
            (quantizer if "quantizer" in name else network)[id(param)] = param
    return network, quantizer


@contextmanager
def keras_device(device):
    """Pin where keras CREATES tensors, and restore it afterwards.

    The other half of MPflowHGQ._sync_keras_device: migrating the Variables is not enough
    on its own, because the quantizers build constants (STE rounding, LUT domains) through
    keras ops, which place them on the keras default device. tests/keras/conftest.py pins
    that to cpu for the whole session, so a test running on cuda must set it and put it
    back.
    """
    previous = get_keras_default_device()
    set_keras_default_device(str(device))
    try:
        yield
    finally:
        set_keras_default_device(previous)
