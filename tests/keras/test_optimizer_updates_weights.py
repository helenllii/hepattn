"""End-to-end: forward -> loss -> backward -> step must move the weights it claims to move.

Coverage of the optimizer's parameter list (test_optimizer_covers_weights.py) is necessary
but not sufficient: a parameter can be in a group, receive a gradient, and still never
change if the tensor the forward pass reads is a different object from the one the
optimizer holds. These tests close that loop on one representative tensor per functional
block -- Q/K/V and output projections, feed-forward, task head -- and on each kind of HGQ2
learned precision (/b integer+fractional bitwidth, /f fractional bits, /i integer bits,
plus the QSoftmax lookup-table quantizer).
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from integration_utils import keras_device, materialize  # ty: ignore [unresolved-import]
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore [unresolved-import]
from test_optimizer_covers_weights import QUANT  # ty: ignore [unresolved-import]

# path fragment -> the block it stands for; one representative tensor is checked per entry
NETWORK_PROBES = {
    "encoder_l0_attn_q_proj/kernel": "Q projection",
    "encoder_l0_attn_k_proj/kernel": "K projection",
    "encoder_l0_attn_v_proj/kernel": "V projection",
    "encoder_l0_attn_out_proj/kernel": "attention output projection",
    "encoder_l0_ffn_hidden0/kernel": "encoder feed-forward",
    "decoder_l0_q_ca_q_proj/kernel": "decoder cross-attention Q projection",
    "task0_net_final/kernel": "task head",
}


def trainable_variables(model):
    return {v.path: v for layer in model.keras_layers() for v in layer.weights if v.trainable and not v.path.endswith("/beta")}


def quantizer_probes(model):
    """One representative trainable quantizer variable of each kind."""
    variables = trainable_variables(model)
    probes = {}
    for suffix, label in (("/b", "HGQ2 bitwidth /b"), ("/f", "HGQ2 fractional bits /f"), ("/i", "HGQ2 integer bits /i")):
        matches = sorted(p for p in variables if p.endswith(suffix) and "quantizer" in p)
        assert matches, f"no trainable {suffix} quantizer variable — the probe would be vacuous"
        probes[matches[0]] = label
    lut = sorted(p for p in variables if "softmax" in p and "quantizer" in p)
    assert lut, "no QSoftmax lookup-table quantizer variable found"
    probes[lut[0]] = "QSoftmax lookup-table quantizer"
    return probes


def train_step(model, inputs, targets, opt):
    opt.zero_grad()
    outputs = model(inputs)
    _, _, loss_dict = model.loss(outputs, dict(targets))
    total = sum(v for layer in loss_dict.values() for task in layer.values() for v in task.values() if torch.isfinite(v))
    total = total + model.quant_losses()
    total.backward()
    opt.step()
    return float(total.detach())


def to_device(tree, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tree.items()}


def make_trained_pair(device, seed=71):
    """Materialize on cpu (as MPflowHGQ.setup does), then migrate exactly like _sync_keras_device."""
    torch.manual_seed(seed)
    model = materialize(make_keras_model(seed=seed, quant=QUANT))
    inputs, targets = clic_dummy_batch(2)
    model.to(device)
    model.move_keras_variables_to(device)
    model.train()
    return model, to_device(inputs, device), to_device(targets, device)


def test_one_step_moves_every_representative_parameter():
    model, inputs, targets = make_trained_pair("cpu")
    variables = trainable_variables(model)
    probes = {**NETWORK_PROBES, **quantizer_probes(model)}
    missing = [p for p in probes if p not in variables]
    assert not missing, f"probe paths absent from the model — update the probes: {missing}"

    before = {path: variables[path].value.detach().clone() for path in probes}
    decay, quant = model.trainable_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay}, {"params": quant, "weight_decay": 0.0}], lr=1e-2, weight_decay=1e-4)

    loss = train_step(model, inputs, targets, opt)
    assert torch.isfinite(torch.tensor(loss)), f"loss is not finite: {loss}"

    for path, label in probes.items():
        grad = variables[path].value.grad
        assert grad is not None, f"{label} ({path}) received no gradient"
        assert torch.isfinite(grad).all(), f"{label} ({path}) has non-finite gradients"
        assert not torch.equal(before[path], variables[path].value.detach()), f"{label} ({path}) did not change after optimizer.step()"

    network_grads = [variables[p].value.grad for p in NETWORK_PROBES]
    assert any(g.abs().sum() > 0 for g in network_grads), "every network probe had an all-zero gradient"


def test_named_parameters_optimizer_leaves_the_kernels_frozen():
    """The pre-fix optimizer, on the same graph: non-vacuity for the test above."""
    model, inputs, targets = make_trained_pair("cpu", seed=72)
    variables = trainable_variables(model)
    kernels = [p for p in NETWORK_PROBES if p in variables]
    before = {path: variables[path].value.detach().clone() for path in kernels}

    old_style = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith("/beta")]
    opt = torch.optim.AdamW(old_style, lr=1e-2)
    train_step(model, inputs, targets, opt)

    frozen = [p for p in kernels if torch.equal(before[p], variables[p].value.detach())]
    assert frozen == kernels, f"named_parameters() now reaches some kernels ({set(kernels) - set(frozen)}) — the fix's premise changed"


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")])],
)
def test_several_steps_stay_finite_and_keep_moving(device):
    # materialization must run with keras creating cpu tensors (it is fed the cpu batch),
    # so the device pin starts only once the model has been migrated — the same order
    # MPflowHGQ uses: setup() materializes, then _sync_keras_device() does both halves.
    model, inputs, targets = make_trained_pair(device, seed=73)
    with keras_device(device):
        _run_training_steps(model, inputs, targets, device)


def _run_training_steps(model, inputs, targets, device):
    variables = trainable_variables(model)
    network = {p: v for p, v in variables.items() if "quantizer" not in p}
    quantizer = {p: v for p, v in variables.items() if "quantizer" in p}
    before = {p: v.value.detach().clone() for p, v in variables.items()}

    decay, quant = model.trainable_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay}, {"params": quant, "weight_decay": 0.0}], lr=1e-3, weight_decay=1e-4)

    losses = [train_step(model, inputs, targets, opt) for _ in range(4)]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), f"non-finite loss during training: {losses}"

    for path, var in variables.items():
        assert torch.isfinite(var.value.detach()).all(), f"{path} contains NaN/inf after training"
        assert var.value.device.type == torch.device(device).type, f"{path} left {device} during training"

    moved_network = [p for p in network if not torch.equal(before[p], network[p].value.detach())]
    moved_quantizer = [p for p in quantizer if not torch.equal(before[p], quantizer[p].value.detach())]
    assert len(moved_network) > 0.5 * len(network), f"only {len(moved_network)}/{len(network)} network weights changed"
    assert moved_quantizer, "no quantizer parameter changed — quantizers are frozen"
