"""A checkpoint must carry the whole model, and must not depend on who looked at it first.

HGQ2 layers never publish their weights to the torch module tree: keras does that from
`Layer._post_build()`, and `hgq.layers.core.base.QLayerBase._post_build` overrides that hook
with assertions of its own and never chains to `super()`. Neither `layer.build(shape)` nor
`layer(x)` helps — keras' `build_wrapper` calls `_post_build` on both paths, and it is the
override that breaks it. No QDense kernel/bias tensor therefore reaches `state_dict()`, so a
checkpoint holds the quantizer state, the norms and the optimizer moments but none of the
network they belong to. The expected key set is derived from the model below rather than
hardcoded, so it tracks the architecture instead of one measurement of it.

Keras recovers lazily — `torch_params`, `.parameters()` and `.named_parameters()` on a keras
layer all call `_track_variables()` if the dict is missing — which makes the key set a
function of whether something happened to traverse submodule `.parameters()` before the save.
`KerasMaskFormer.register_keras_parameters()` does it once, eagerly, at a defined point.

Discriminating properties:
- complete: every trainable weight the model has is serialized -- catches the bug exactly.
- invariant: no incidental introspection changes the key set -- catches a fix that relies on
  the lazy path still being triggered by something.
- round trip: a fresh model restored from the state reproduces outputs, EBOPs and quantizer
  state -- catches keys that exist but are not the tensors the forward pass reads.
- identity: registration reuses the existing nn.Parameters -- catches a fix that copies
  tensors to satisfy state_dict and silently detaches the optimizer or the keras Variables.
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from integration_utils import expected_groups, materialize  # ty: ignore [unresolved-import]
from lightning.pytorch.utilities.model_summary import ModelSummary
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore [unresolved-import]
from test_optimizer_covers_weights import QUANT  # ty: ignore [unresolved-import]

REPRESENTATIVE = {
    "innet0_net_final/kernel": "input projection",
    "encoder_l0_attn_q_proj/kernel": "encoder Q projection",
    "encoder_l0_attn_k_proj/kernel": "encoder K projection",
    "encoder_l0_attn_v_proj/kernel": "encoder V projection",
    "encoder_l0_attn_out_proj/kernel": "encoder attention output",
    "encoder_l0_ffn_hidden0/kernel": "encoder FFN",
    "decoder_l0_q_sa_q_proj/kernel": "decoder self-attention",
    "decoder_l0_q_ca_q_proj/kernel": "decoder cross-attention",
    "task0_net_final/kernel": "task head",
}


@pytest.fixture(scope="module")
def registered():
    model = materialize(make_keras_model(seed=11, quant=QUANT))
    model.register_keras_parameters()
    return model


def serialized_ids(model):
    return {id(t) for t in model.state_dict(keep_vars=True).values()}


def test_every_trainable_weight_is_serialized_exactly_once(registered):
    """Test A: derive the expected set from the model, do not hardcode it."""
    network, quantizer = expected_groups(registered)
    state = registered.state_dict(keep_vars=True)
    ids = [id(t) for t in state.values()]
    covered = set(ids)

    missing_network = [p for i, p in network.items() if i not in covered]
    missing_quantizer = [p for i, p in quantizer.items() if i not in covered]
    assert not missing_network, f"{len(missing_network)}/{len(network)} network weight tensors absent from state_dict()"
    assert not missing_quantizer, f"{len(missing_quantizer)}/{len(quantizer)} quantizer tensors absent from state_dict()"

    # every trainable tensor is reachable, and no tensor object was duplicated into a copy
    assert len(covered) == len({id(t) for t in state.values()})


def test_representative_layers_are_present_by_name(registered):
    state = registered.state_dict()
    for path, label in REPRESENTATIVE.items():
        assert [k for k in state if k.endswith(path)], f"{label} ({path}) has no key in state_dict()"

    trainable = {v.path: v for layer in registered.keras_layers() for v in layer.weights if v.trainable}
    for suffix, label in (("/b", "HGQ2 /b"), ("/f", "HGQ2 /f"), ("/i", "HGQ2 /i")):
        paths = [p for p in trainable if p.endswith(suffix) and "quantizer" in p]
        assert paths, f"no trainable {label} variable — the check would be vacuous"
        assert all([k for k in state if k.endswith(p)] for p in paths[:5]), f"{label} variables missing from state_dict()"

    lut = [p for p in trainable if "softmax" in p and "quantizer" in p]
    assert lut and [k for k in state if k.endswith(lut[0])], "QSoftmax lookup-table quantizer state missing"


def test_state_dict_is_invariant_to_introspection(registered):
    """Test B: no incidental call may change what a checkpoint would contain."""
    baseline = set(registered.state_dict())
    stages = {
        "model.parameters()": lambda: list(registered.parameters()),
        "submodule .parameters()": lambda: [list(m.parameters()) for _, m in registered.named_modules()],
        "keras_layers() .parameters()": lambda: [list(layer.parameters()) for layer in registered.keras_layers()],
        "optimizer creation": lambda: torch.optim.AdamW([{"params": g} for g in registered.trainable_parameter_groups()], lr=1e-3),
        "ModelSummary": lambda: ModelSummary(_wrap_lightning(registered), max_depth=-1),
        "trainable_parameter_groups": registered.trainable_parameter_groups,
        "register_keras_parameters (again)": registered.register_keras_parameters,
    }
    for label, action in stages.items():
        action()
        assert set(registered.state_dict()) == baseline, f"{label} changed the state_dict key set"


def _wrap_lightning(model):
    import lightning.pytorch as pl  # noqa: PLC0415

    class _Wrap(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model

    return _Wrap()


def test_registration_reuses_the_existing_parameters(registered):
    """Test 4: nothing may be copied — the optimizer and the keras Variables share these objects."""
    state = registered.state_dict(keep_vars=True)
    live = {id(var.value) for var in registered.keras_variables()}
    keras_entries = [t for name, t in state.items() if "_torch_params" in name]
    assert keras_entries, "no keras variables are registered — the check would be vacuous"
    stale = [t for t in keras_entries if id(t) not in live]
    assert not stale, f"{len(stale)} serialized tensors are copies, not the Variables the forward pass reads"

    decay, quant = registered.trainable_parameter_groups()
    ids = [id(p) for p in decay + quant]
    assert len(ids) == len(set(ids)), "trainable_parameter_groups returned a tensor twice"
    assert {id(p) for p in decay + quant} <= live | {id(p) for p in registered.parameters()}


def test_save_load_round_trip_reproduces_the_model():
    """Test C: a fresh model restored from the state must be the same function."""
    source = materialize(make_keras_model(seed=12, quant=QUANT))
    source.register_keras_parameters()
    inputs, _ = clic_dummy_batch(2)
    source.eval()
    with torch.no_grad():
        before = source(inputs)
        ebops_before = float(source.quant_losses())
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    probes = {path: t.clone() for path, t in source.state_dict().items() if any(path.endswith(p) for p in REPRESENTATIVE)}
    assert probes, "no representative tensors captured"

    target = materialize(make_keras_model(seed=999, quant=QUANT))
    target.register_keras_parameters()
    incompatible = target.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys, f"missing keys: {incompatible.missing_keys[:5]}"
    assert not incompatible.unexpected_keys, f"unexpected keys: {incompatible.unexpected_keys[:5]}"

    restored = target.state_dict()
    for path, expected in probes.items():
        assert torch.equal(restored[path], expected), f"{path} differs after load_state_dict"

    target.eval()
    with torch.no_grad():
        after = target(inputs)
        ebops_after = float(target.quant_losses())
    assert ebops_before == pytest.approx(ebops_after, rel=0, abs=0), f"EBOPs differ: {ebops_before} vs {ebops_after}"

    quant_before = {v.path: v.value.detach() for layer in source.keras_layers() for v in layer.weights if "quantizer" in v.path}
    quant_after = {v.path: v.value.detach() for layer in target.keras_layers() for v in layer.weights if "quantizer" in v.path}
    assert set(quant_before) == set(quant_after)
    assert all(torch.equal(quant_before[p], quant_after[p]) for p in quant_before), "quantizer state differs after restore"

    flat_b = _flatten(before)
    flat_a = _flatten(after)
    assert set(flat_b) == set(flat_a)
    worst = max((float((flat_b[k] - flat_a[k]).abs().max()) for k in flat_b if flat_b[k].is_floating_point()), default=0.0)
    assert worst == 0.0, f"restored model outputs differ, max |delta| = {worst}"


def _flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        elif isinstance(value, torch.Tensor):
            flat[path] = value
    return flat


def test_unregistered_model_would_lose_its_kernels():
    """Non-vacuity: the same model without the fix serializes none of its kernels."""
    model = materialize(make_keras_model(seed=13, quant=QUANT))
    ids = serialized_ids(model)
    kernels = [w for layer in model.keras_layers() for w in layer.weights if w.path.endswith(("/kernel", "/bias")) and w.trainable]
    assert kernels, "fixture built no kernels"
    assert not [w for w in kernels if id(w.value) in ids], (
        "HGQ2 layers now register their own weights — upstream behaviour changed, re-derive register_keras_parameters"
    )
    assert model.register_keras_parameters() > 0
    ids = serialized_ids(model)
    assert not [w for w in kernels if id(w.value) not in ids], "registration did not cover every kernel"
