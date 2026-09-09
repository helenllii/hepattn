"""Lightning module for the Keras/HGQ2 CLIC pflow model.

Kept separate from lightning_module.py so torch-only runs never import keras
(importing keras pins its backend process-wide).
"""

import torch
from lion_pytorch import Lion
from torch import Tensor
from torch.optim import AdamW

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer


class MPflowHGQ(MPflow):
    """MPflow driving a KerasMaskFormer (float reference or HGQ2 quantization-aware).

    Adds on top of MPflow:
    - the HGQ2 EBOPs regularization term in the aggregated loss,
    - materialization of lazily-built quantized layers before the optimizer is
      created and before checkpoint state is restored (HGQ2 layers size their
      bitwidth variables from the first real batch's static shapes),
    - a quantizer parameter group without weight decay (decaying learned bitwidths
      would silently shrink precision), with the non-trainable beta excluded,
    - registration of the keras weights on the torch module tree, without which they
      are absent from state_dict() and therefore from every checkpoint
      (see KerasMaskFormer.register_keras_parameters),
    - migration of already-materialized keras Variables onto Lightning's device, which
      nn.Module.to() cannot do for the ones no layer registered
      (see KerasMaskFormer.move_keras_variables_to).
    """

    def setup(self, stage: str) -> None:
        super().setup(stage)
        assert isinstance(self.model, KerasMaskFormer), "MPflowHGQ requires a KerasMaskFormer model"
        # Create keras variables on the RANK'S device, not cpu.
        #
        # The old comment here said variables are "created on cpu and moved with the module
        # by Lightning". Measured (polaris/10_ddp_device_probe.py): they are NOT. Module.to()
        # moves all 2027 registered parameters and buffers, and leaves the keras Variables
        # behind on cpu. Single-device runs survive that because keras reads them wherever
        # they are; DDP does not, because torch's _sync_module_states walks a wider set than
        # named_parameters()+named_buffers() and hands NCCL 684 cpu tensors, which fails with
        # "No backend type associated with device type cpu" (measured, run 7598422).
        #
        # self.device is still cpu at setup() -- Lightning has not moved the module yet -- so
        # take the device from the strategy, which setup_environment() has already resolved.
        set_keras_default_device(self._target_device())
        self._materialize_keras_layers(stage)
        # Every lazy keras Variable now exists. Publish them to the torch module tree here,
        # before anything can save a checkpoint or introspect the module: HGQ2 layers never
        # do it themselves, and keras' lazy recovery would otherwise make the state_dict key
        # set depend on whether something happened to traverse submodule .parameters(). It
        # also has to precede on_load_checkpoint(), which checks a restored state against
        # exactly the key set this call fixes.
        self.model.register_keras_parameters()

    def _target_device(self) -> str:
        strategy = getattr(self.trainer, "strategy", None)
        root = getattr(strategy, "root_device", None)
        return str(root) if root is not None else str(self.device)

    def _sync_keras_device(self) -> None:
        # Two distinct things have to follow Lightning's device:
        # 1. where keras creates NEW tensors -- quantizer internals (STE rounding, LUT
        #    domains) otherwise mix cpu constants with cuda activations;
        # 2. where the ALREADY-materialized keras Variables live. set_keras_default_device
        #    does not touch those, and nn.Module.to() only reaches the ones a layer
        #    registered on the torch module tree.
        #
        # setup() now builds on the strategy's root device, so both are normally no-ops.
        # Kept because self.device is authoritative once Lightning has moved the module, and
        # because test/predict can run without a fit having gone through setup() first --
        # in which case the Variables really are somewhere else and have to be migrated.
        set_keras_default_device(str(self.device))
        self.model.move_keras_variables_to(self.device)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._sync_keras_device()

    def on_validation_start(self) -> None:
        self._sync_keras_device()

    def on_test_start(self) -> None:
        self._sync_keras_device()

    def on_predict_start(self) -> None:
        self._sync_keras_device()

    def _materialize_keras_layers(self, stage: str) -> None:
        datamodule = self.trainer.datamodule
        loader_fn = {
            "fit": datamodule.train_dataloader,
            "validate": datamodule.val_dataloader,
        }.get(stage, datamodule.test_dataloader)
        inputs, _ = next(iter(loader_fn()))
        # the loader yields cpu tensors; the layers are being built on the target device
        device = self._target_device()
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.model.to(device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)
        self.model.train(was_training)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Refuse a checkpoint that predates the keras-weight registration fix.

        Such a checkpoint carries the quantizer state, the norms and the optimizer moments
        but none of the network's kernels, so Lightning would restore it onto freshly
        initialized weights and report a successful resume. Fail here, with the reason,
        rather than let strict=True print thousands of missing keys or -- worse -- let a
        non-strict load through.

        Runs after setup(), so register_keras_parameters() has already fixed the key set
        this compares against.

        Raises:
            RuntimeError: If the checkpoint does not carry every key the model expects.
        """
        state = checkpoint.get("state_dict")
        if state is None:
            return
        missing = self.model.missing_state_keys({k.removeprefix("model."): v for k, v in state.items()})
        if missing:
            raise RuntimeError(
                f"this checkpoint is missing {len(missing)} of the model's state keys, including "
                f"{sum(1 for k in missing if k.endswith(('/kernel', '/bias')))} keras kernel/bias tensors "
                f"(e.g. {missing[:3]}). It was written before KerasMaskFormer.register_keras_parameters "
                "existed, so it does not contain the trained network weights and cannot be resumed "
                "faithfully -- only its quantizer state and norms are recoverable."
            )

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str | None = None) -> Tensor:
        total_loss = super().aggregate_losses(losses, stage=stage)
        quant_loss = self.model.quant_losses()
        self.log(f"{stage}/quant_ebops_loss", quant_loss, sync_dist=True)
        return total_loss + quant_loss

    def configure_optimizers(self):
        # Mirrors ModelWrapper.configure_optimizers with quantizer-aware param groups.
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # NOT self.model.named_parameters(): keras layer weights are not registered on
        # the nn.Module, so that list omits every Dense kernel in the model. See
        # KerasMaskFormer.trainable_parameter_groups.
        decay_params, quantizer_params = self.model.trainable_parameter_groups()

        # A tensor in two groups would take its update twice, with two weight decays.
        # Cheap, and the two groups are built from overlapping traversals.
        ids = [id(p) for p in decay_params + quantizer_params]
        assert len(ids) == len(set(ids)), "duplicate parameter objects across optimizer groups"

        param_groups = [{"params": decay_params}, {"params": quantizer_params, "weight_decay": 0.0}]
        opt = optimizer(param_groups, lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
