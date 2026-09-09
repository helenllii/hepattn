"""A Lightning checkpoint of the HGQ2 module must round-trip, and a resume must continue the run.

Test D drives the real `MPflowHGQ` through a real `Trainer`, saves a checkpoint and restores
it into a fresh model. Test E is the discriminating one: training N+M steps continuously and
training N steps, restoring, then training M more must land on the same weights. That fails
outright on the old code -- the kernels are not in the checkpoint, so the resumed run carries
on from a freshly initialized network with restored Adam moments, and nothing reports an error.

Step 6 (callback independence) is covered here too: what a checkpoint contains must not depend
on which callbacks were configured.
"""

import copy

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore [unresolved-import]
from test_optimizer_covers_weights import QUANT  # ty: ignore [unresolved-import]

from hepattn.experiments.clic.lightning_module_hgq import MPflowHGQ
from hepattn.keras.callbacks import BitwidthMonitor, EBOPsMonitor

LRS = {"initial": 1e-3, "max": 1e-3, "end": 1e-4, "pct_start": 0.3, "weight_decay": 1e-4, "skip_scheduler": False}
PROBES = ("encoder_l0_attn_q_proj/kernel", "encoder_l0_ffn_hidden0/kernel", "task0_net_final/kernel")


def fixed_batch():
    """One dummy CLIC batch drawn from a pinned RNG state.

    CLICDataset's dummy generator draws from the global torch RNG, so two datamodules built
    at different points in a test would serve DIFFERENT data — which looks exactly like a
    broken resume.
    """
    torch.manual_seed(7)
    return clic_dummy_batch(2)


class FixedBatchData(pl.LightningDataModule):
    """One deterministic batch, served `steps` times — makes two runs comparable step for step."""

    def __init__(self, steps: int = 8, batch=None):
        super().__init__()
        self.steps = steps
        self.batch = batch if batch is not None else fixed_batch()

    def train_dataloader(self):
        batches = [self.batch] * self.steps
        return torch.utils.data.DataLoader(batches, batch_size=None, num_workers=0, shuffle=False)

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader


def make_module(seed: int) -> MPflowHGQ:
    torch.manual_seed(seed)
    return MPflowHGQ(name="ckpt-test", model=make_keras_model(seed=seed, quant=QUANT), lrs_config=dict(LRS), optimizer="AdamW")


class SaveAt(pl.Callback):
    """Write a checkpoint mid-run, without disturbing the run.

    Test E compares a resumed leg against the SAME trajectory that produced the checkpoint,
    rather than against a separately seeded re-run. Two runs of one config do not start from
    the same weights: the encoder/decoder attention QDense layers have no torch module to port
    from, so keras initializes them itself, and the initializer's seed is drawn when the layer
    is constructed — just after KerasMaskFormer's `clear_session()` — which neither
    `seed_everything` nor `keras.utils.set_random_seed` reaches. Branching off one trajectory
    keeps that (real, separate) reproducibility gap out of this measurement.
    """

    def __init__(self, steps: int, path):
        self.steps = steps
        self.path = path
        self.saved = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.saved and trainer.global_step >= self.steps:
            trainer.save_checkpoint(self.path)
            self.saved = True


def make_trainer(tmp_path, max_steps: int, callbacks=None, limit_val_batches=0):
    return pl.Trainer(
        max_steps=max_steps,
        max_epochs=-1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10_000,  # keep predict()/metrics out of the compared trajectory
        default_root_dir=str(tmp_path),
        callbacks=callbacks or [],
    )


def probe_values(module):
    variables = {v.path: v for layer in module.model.keras_layers() for v in layer.weights}
    return {path: variables[path].value.detach().clone() for path in PROBES if path in variables}


def quantizer_values(module):
    return {v.path: v.value.detach().clone() for layer in module.model.keras_layers() for v in layer.weights if "quantizer" in v.path and v.trainable}


def test_lightning_checkpoint_round_trip(tmp_path):
    """Test D: save from a real fit, restore into a fresh model, outputs must match."""
    data = FixedBatchData(steps=4)
    module = make_module(31)
    trainer = make_trainer(tmp_path, max_steps=3)
    trainer.fit(module, datamodule=data)

    ckpt = tmp_path / "d.ckpt"
    trainer.save_checkpoint(ckpt)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["global_step"] == 3

    inputs, _ = data.batch
    module.model.eval()
    with torch.no_grad():
        before = module.model(inputs)
        ebops_before = float(module.model.quant_losses())
    weights_before, quant_before = probe_values(module), quantizer_values(module)

    restored = make_module(999)
    restored_trainer = make_trainer(tmp_path, max_steps=3)
    restored_trainer.fit(restored, datamodule=data, ckpt_path=str(ckpt))  # restores, then has nothing left to run

    assert restored.trainer.global_step == 3, "global_step was not restored"
    for path, expected in weights_before.items():
        got = probe_values(restored)[path]
        assert torch.equal(expected, got), f"{path} not restored (max delta {float((expected - got).abs().max())})"
    after_quant = quantizer_values(restored)
    assert set(after_quant) == set(quant_before)
    assert all(torch.equal(quant_before[p], after_quant[p]) for p in quant_before), "quantizer state not restored"

    opt_state = saved["optimizer_states"][0]
    assert sum(len(g["params"]) for g in opt_state["param_groups"]) == sum(len(g) for g in module.model.trainable_parameter_groups())
    assert saved["lr_schedulers"], "no LR scheduler state saved"

    restored.model.eval()
    with torch.no_grad():
        after = restored.model(inputs)
        ebops_after = float(restored.model.quant_losses())
    assert ebops_before == ebops_after, f"EBOPs differ after restore: {ebops_before} vs {ebops_after}"
    worst = _max_delta(before, after)
    assert worst == 0.0, f"restored model outputs differ, max |delta| = {worst}"


def test_resume_equals_continuous_training(tmp_path):
    """Test E: a run checkpointed at step N and resumed into a fresh model must land where it would have."""
    n, m = 3, 3
    batch = fixed_batch()
    ckpt = tmp_path / "e.ckpt"

    control = make_module(41)
    saver = SaveAt(n, ckpt)
    make_trainer(tmp_path / "control", max_steps=n + m, callbacks=[saver]).fit(control, datamodule=FixedBatchData(n + m + 2, batch))
    assert saver.saved and ckpt.exists(), "no mid-run checkpoint was written"
    assert control.trainer.global_step == n + m
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["global_step"] == n

    resumed = make_module(999)  # deliberately different init: everything must come from the checkpoint
    at_start = {}
    make_trainer(tmp_path / "resumed", max_steps=n + m, callbacks=[_Capture(at_start)]).fit(
        resumed, datamodule=FixedBatchData(n + m + 2, batch), ckpt_path=str(ckpt)
    )
    assert resumed.trainer.global_step == n + m, "resumed run did not continue to the configured horizon"
    assert at_start, "resume never reached on_train_start"

    control_probes, resumed_probes = probe_values(control), probe_values(resumed)
    assert control_probes, "no probe tensors found"
    for path, expected in control_probes.items():
        delta = float((expected - resumed_probes[path]).abs().max())
        rel = delta / max(float(expected.abs().max()), 1e-12)
        assert delta <= 1e-5 and rel <= 1e-4, f"{path}: resumed run diverged, max |delta| = {delta:.3e} (rel {rel:.3e})"

    cq, rq = quantizer_values(control), quantizer_values(resumed)
    assert set(cq) == set(rq)
    qdelta = max((float((cq[p] - rq[p]).abs().max()) for p in cq if cq[p].is_floating_point()), default=0.0)
    assert qdelta <= 1e-5, f"quantizer state diverged, max |delta| = {qdelta:.3e}"

    inputs, _ = batch
    for module in (control, resumed):
        module.model.eval()
    with torch.no_grad():
        worst = _max_delta(control.model(inputs), resumed.model(inputs))
    assert worst <= 1e-4, f"outputs of the resumed run differ, max |delta| = {worst:.3e}"

    control_opt = control.trainer.optimizers[0].state_dict()
    resumed_opt = resumed.trainer.optimizers[0].state_dict()
    assert control_opt["param_groups"][0]["lr"] == pytest.approx(resumed_opt["param_groups"][0]["lr"]), "LR schedule diverged"
    exp_delta = max(float((control_opt["state"][k]["exp_avg"] - resumed_opt["state"][k]["exp_avg"]).abs().max()) for k in control_opt["state"])
    assert exp_delta <= 1e-5, f"optimizer moments diverged, max |delta| = {exp_delta:.3e}"


class _Capture(pl.Callback):
    """Record the restored weights at train start, to separate a bad restore from a bad resume."""

    def __init__(self, into: dict):
        self.into = into

    def on_train_start(self, trainer, pl_module):
        self.into.update(probe_values(pl_module))


def test_checkpoint_keys_do_not_depend_on_callbacks(tmp_path):
    """Step 6: checkpoint coverage must be a property of the model, not of the callback list.

    The callbacks chosen are the ones that traverse the module: ModelSummary walks
    `named_parameters()` at fit start, BitwidthMonitor walks it again each validation
    epoch, EBOPsMonitor walks `keras_layers()`. Any of those can trip keras' lazy
    `_track_variables()` recovery, which is exactly how the saved key set used to become a
    function of the callback list. Validation is enabled here so the two epoch-end
    callbacks actually run.
    """
    configurations = {
        "none": [],
        "ModelSummary": [ModelSummary(max_depth=-1)],
        "monitors": [BitwidthMonitor(), EBOPsMonitor()],
        "both": [ModelSummary(max_depth=-1), BitwidthMonitor(), EBOPsMonitor()],
    }
    saved = {}
    for label, callbacks in configurations.items():
        module = make_module(51)
        trainer = make_trainer(tmp_path / label, max_steps=2, callbacks=copy.copy(callbacks), limit_val_batches=1)
        trainer.fit(module, datamodule=FixedBatchData(steps=3))
        ckpt = tmp_path / f"{label}.ckpt"
        module.trainer.save_checkpoint(ckpt)
        saved[label] = set(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])

    reference = saved["none"]
    assert [k for k in reference if k.endswith(("/kernel", "/bias"))], "no keras kernels in the checkpoint at all"
    for label, keys in saved.items():
        assert keys == reference, (
            f"callbacks={label} changed the saved model state: {len(keys - reference)} extra, {len(reference - keys)} missing keys"
        )


def _max_delta(a, b):
    fa, fb = _flatten(a), _flatten(b)
    assert set(fa) == set(fb)
    return max((float((fa[k] - fb[k]).abs().max()) for k in fa if fa[k].is_floating_point()), default=0.0)


def _flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        elif isinstance(value, torch.Tensor):
            flat[path] = value
    return flat


def test_incomplete_checkpoint_is_refused_loudly(tmp_path):
    """Step 9: a checkpoint written before the fix must fail with its reason, not resume silently.

    Reproduces the shape of the real production checkpoint (quantizer state, norms and
    optimizer moments, no kernels) by stripping the keras weight keys from a good one.
    """
    data = FixedBatchData(steps=3)
    module = make_module(61)
    make_trainer(tmp_path / "guard", max_steps=1).fit(module, datamodule=data)
    good = tmp_path / "good.ckpt"
    module.trainer.save_checkpoint(good)

    checkpoint = torch.load(good, map_location="cpu", weights_only=False)
    stripped = {k: v for k, v in checkpoint["state_dict"].items() if not k.endswith(("/kernel", "/bias"))}
    assert len(stripped) < len(checkpoint["state_dict"]), "nothing was stripped — the guard would be untested"

    module.on_load_checkpoint({"state_dict": checkpoint["state_dict"]})  # complete: must not raise
    with pytest.raises(RuntimeError, match=r"missing .* of the model's state keys"):
        module.on_load_checkpoint({"state_dict": stripped})
