"""Every keras Variable must live on the device Lightning picked, not just the registered ones.

`nn.Module.to()` cannot move them all: it reaches only the variables a layer registered in
its `_torch_params`, and an HGQ2 layer registers none of its own (see
`KerasMaskFormer.register_keras_parameters` for why). Measured on Polaris, `.to("cuda")`
stranded 684 of the CLIC model's keras Variables on cpu. On a single device nothing fails
visibly, because keras' own `convert_to_tensor` copies each stranded tensor to the GPU on
every single op; under DDP `_sync_module_states` hands those cpu tensors to NCCL and the
run aborts.

`set_keras_default_device()` does not fix this: it only chooses where FUTURE tensors are
created. Already-materialized variables have to be migrated explicitly, in place, so that
an optimizer already holding references keeps pointing at the same objects.
"""

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from integration_utils import keras_device, materialize  # ty: ignore [unresolved-import]
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore [unresolved-import]
from test_optimizer_covers_weights import QUANT  # ty: ignore [unresolved-import]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def variable_devices(model):
    return [var.value.device for var in model.keras_variables()]


def on_cpu(model):
    return [var.path for var in model.keras_variables() if var.value.device.type == "cpu"]


@pytest.fixture
def built():
    return materialize(make_keras_model(seed=90, quant=QUANT))


def test_cpu_migration_is_a_no_op_that_preserves_identity(built):
    """CPU training must keep working, and nothing may be replaced by a copy."""
    before = {var.path: id(var.value) for var in built.keras_variables()}
    assert built.move_keras_variables_to("cpu") == 0, "variables reported as moved on a cpu -> cpu migration"
    after = {var.path: id(var.value) for var in built.keras_variables()}
    assert after == before, "a cpu -> cpu migration replaced Variable tensors"
    assert on_cpu(built), "fixture is not on cpu — the test would be vacuous"


@requires_cuda
@pytest.mark.gpu
def test_module_to_alone_leaves_keras_variables_behind(built):
    """Non-vacuity: this is exactly the state the fix exists to repair."""
    built.to("cuda")
    stranded = on_cpu(built)
    assert stranded, "nn.Module.to() now moves the keras Variables — re-derive the fix before removing it"


@requires_cuda
@pytest.mark.gpu
def test_no_keras_variable_remains_off_device(built):
    device = torch.device("cuda", torch.cuda.current_device())
    built.to(device)
    built.move_keras_variables_to(device)

    assert on_cpu(built) == [], f"{len(on_cpu(built))} keras Variables still on cpu"
    assert {str(d) for d in variable_devices(built)} == {str(device)}, "keras Variables are spread over several devices"

    trainable, quantizer = built.trainable_parameter_groups()
    assert {p.device for p in trainable} == {device}, "network weights in the optimizer are off-device"
    assert {p.device for p in quantizer} == {device}, "quantizer parameters in the optimizer are off-device"
    assert {p.device for p in built.parameters()} == {device}, "registered torch parameters are off-device"


@requires_cuda
@pytest.mark.gpu
def test_migration_keeps_the_optimizer_pointing_at_the_same_tensors(built):
    """The optimizer is built before on_fit_start, so migration must not swap objects out."""
    device = torch.device("cuda", torch.cuda.current_device())
    trainable, quantizer = built.trainable_parameter_groups()
    opt = torch.optim.AdamW([{"params": trainable}, {"params": quantizer, "weight_decay": 0.0}], lr=1e-3)
    before = [id(p) for group in opt.param_groups for p in group["params"]]

    built.to(device)
    built.move_keras_variables_to(device)

    after = [id(p) for group in opt.param_groups for p in group["params"]]
    assert before == after, "migration replaced parameter objects the optimizer already references"
    assert {p.device for group in opt.param_groups for p in group["params"]} == {device}, "optimizer parameters are off-device"
    live = {id(var.value) for var in built.keras_variables()}
    assert [i for i in after if i in live], "optimizer no longer references any live keras Variable"


@requires_cuda
@pytest.mark.gpu
def test_registered_parameters_still_alias_the_live_variables(built):
    """state_dict()/named_parameters() must expose the tensors the forward pass actually reads.

    Both moves preserve `nn.Parameter` identity -- torch's `_apply` takes the `set_data`
    path across devices, and the migration below is an in-place `.data` swap -- so a
    registered variable must never end up as a stale copy in `_torch_params`.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    built.to(device)
    built.move_keras_variables_to(device)

    registered = dict(built.named_parameters())
    live = {id(var.value) for var in built.keras_variables()}
    keras_entries = {name: p for name, p in registered.items() if "_torch_params" in name}
    assert keras_entries, "no keras variables are registered at all — the aliasing check would be vacuous"
    orphans = [name for name, p in keras_entries.items() if id(p) not in live]
    assert not orphans, f"{len(orphans)} registered parameters are stale copies, e.g. {orphans[:3]}"


@requires_cuda
@pytest.mark.gpu
def test_gradients_flow_to_migrated_variables(built):
    device = torch.device("cuda", torch.cuda.current_device())
    inputs, targets = clic_dummy_batch(2)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}

    # both halves of MPflowHGQ._sync_keras_device: where keras CREATES tensors, and where
    # the already-materialized Variables live
    with keras_device(device):
        built.to(device)
        built.move_keras_variables_to(device)
        built.train()

        outputs = built(inputs)
        _, _, loss_dict = built.loss(outputs, dict(targets))
        total = sum(v for layer in loss_dict.values() for task in layer.values() for v in task.values() if torch.isfinite(v))
        total = total + built.quant_losses()
        assert torch.isfinite(total), "loss is not finite on cuda"
        total.backward()

    kernels = [var for var in built.keras_variables() if var.path.endswith("/kernel") and var.trainable]
    with_grad = [var for var in kernels if var.value.grad is not None and var.value.grad.abs().sum() > 0]
    assert len(with_grad) > 0.5 * len(kernels), f"only {len(with_grad)}/{len(kernels)} kernels received a nonzero gradient"
    assert all(torch.isfinite(var.value.grad).all() for var in with_grad), "non-finite gradients on migrated variables"
    assert all(var.value.grad.device == device for var in with_grad), "gradients landed on a different device"
