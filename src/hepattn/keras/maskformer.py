"""KerasMaskFormer: the hepattn MaskFormer with its network rebuilt from Keras/HGQ2 layers.

Orchestration (forward/loss/predict, per-layer matching, mask attention) is inherited
from MaskFormer. Input nets and tasks are the SAME torch modules as the YAML config
describes, with their Dense sub-nets swapped for factory-built Keras twins; the encoder
and decoder are keras mirrors constructed from plain config dicts so that every keras
layer is created inside the HGQ2 config scopes regardless of YAML instantiation order.

With quant=None this is the float parity-reference model; with a quant spec it is the
HGQ2 quantization-aware model on the identical graph (float weights warm-start QAT).
"""

import torch
from torch import Tensor, nn

from hepattn.keras import get_keras_default_device, keras, set_keras_default_device
from hepattn.keras.decoder import KerasMaskFormerDecoder
from hepattn.keras.encoder import KerasEncoder
from hepattn.keras.factory import LayerFactory
from hepattn.keras.tasks import kerasify_module
from hepattn.models.maskformer import MaskFormer


class KerasMaskFormer(MaskFormer):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: dict,
        decoder: dict,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        matcher: nn.Module | None = None,
        encoder_tasks: nn.ModuleList | None = None,
        quant: dict | None = None,
    ):
        """Build the keras-backed MaskFormer.

        Args:
            input_nets: Instantiated hepattn InputNet modules (their Dense nets are swapped in place).
            encoder: KerasEncoder configuration dict (same keys as the torch Encoder YAML section).
            decoder: KerasMaskFormerDecoder configuration dict (same keys as the torch decoder YAML section).
            tasks: Instantiated hepattn task modules (their Dense nets are swapped in place).
            dim: Embedding dimension.
            target_object: Target object name used during matching.
            matcher: Hungarian matcher module (reused unchanged, training-only).
            encoder_tasks: Optional tasks run on post-encoder features (Dense nets swapped in place).
            quant: None for the float reference model, or an HGQ2 QuantSpec dict
                (keys: weight, datalane, ebops) for the quantization-aware model.
        """
        factory = LayerFactory(quant)

        # Reset keras's process-global name counters so every layer name — including the
        # quantizer sub-layers HGQ2 creates internally, which cannot be named explicitly —
        # is determined by construction order alone. Keras embeds these names in the torch
        # state_dict keys; without the reset, a checkpoint written by the first model built
        # in a process cannot be loaded into the second (e.g. Lightning's `test` restore).
        device = get_keras_default_device()
        keras.utils.clear_session(free_memory=False)
        set_keras_default_device(device)

        with factory.scopes():
            keras_encoder = KerasEncoder(dim=dim, factory=factory, **encoder)
            keras_decoder = KerasMaskFormerDecoder(factory=factory, **decoder)
            for i, input_net in enumerate(input_nets):
                kerasify_module(input_net, factory, name=f"innet{i}")
            for i, task in enumerate(tasks):
                kerasify_module(task, factory, name=f"task{i}")
            for i, task in enumerate(encoder_tasks or []):
                kerasify_module(task, factory, name=f"enctask{i}")

        super().__init__(
            input_nets=input_nets,
            encoder=keras_encoder,
            decoder=keras_decoder,
            tasks=tasks,
            dim=dim,
            target_object=target_object,
            matcher=matcher,
            encoder_tasks=encoder_tasks,
        )
        self.factory = factory

    def keras_layers(self):
        """Yield the top-level keras layers of the model (without descending into their sublayers)."""

        def walk(module: nn.Module):
            for child in module.children():
                if isinstance(child, keras.layers.Layer):
                    yield child
                else:
                    yield from walk(child)

        yield from walk(self)

    def quant_losses(self) -> Tensor:
        """Sum of the HGQ2 EBOPs regularization terms collected from all keras layers.

        Keras layers register their EBOPs*beta losses via add_loss during training-mode
        calls; collecting from top-level layers only avoids double counting (Layer.losses
        already aggregates sublayers such as quantizers).
        """
        terms = [loss for layer in self.keras_layers() for loss in layer.losses]
        if not terms:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack([torch.as_tensor(t) for t in terms]).sum()

    def trainable_parameter_groups(self) -> tuple[list, list]:
        """Every trainable tensor, split into (non-quantizer, quantizer) groups.

        MUST be used instead of `named_parameters()` to build an optimizer.

        Keras 3 on the torch backend keeps layer weights as keras `Variable`s. Each
        `.value` IS an `nn.Parameter` with `requires_grad=True`, and gradients do reach
        them -- but they are never registered on the `nn.Module`, so `named_parameters()`
        does not list them. An optimizer built from that list therefore contains **none**
        of the model's Dense kernels: measured on the CLIC config, 282 kernel/bias
        tensors totalling 11.6M elements, all receiving gradients, none optimized. What
        remained was 35.3M quantizer parameters plus 77k of LayerNorm/register/query
        tensors -- which is why the loss stalled around 29-30 instead of approaching the
        float reference's 3.79.

        `/beta` is excluded: it is the regularization strength HGQ2's add_loss reads, and
        its 'gradient' is just the EBOPs magnitude. Deduplicated by object identity,
        since `keras_layers()` aggregates sublayer weights and can yield the same
        variable more than once.
        """
        decay: list = []
        quant: list = []
        seen: set[int] = set()

        def add(param, path: str) -> None:
            if id(param) in seen or not getattr(param, "requires_grad", False) or path.endswith("/beta"):
                return
            seen.add(id(param))
            (quant if "quantizer" in path else decay).append(param)

        for name, param in self.named_parameters():
            add(param, name)
        for layer in self.keras_layers():
            for var in layer.weights:
                if getattr(var, "trainable", True):
                    add(var.value, var.path)
        return decay, quant

    def keras_variables(self):
        """Yield every keras Variable owned by the model exactly once (trainable or not).

        `Layer.variables` is recursive and includes seed/metric state, so it also reaches
        sublayers that keras tracks in plain lists rather than as torch submodules.
        """
        seen: set[int] = set()
        for layer in self.keras_layers():
            for var in layer.variables:
                if id(var) not in seen:
                    seen.add(id(var))
                    yield var

    @torch.no_grad()
    def move_keras_variables_to(self, device: torch.device | str) -> int:
        """Move every already-materialized keras Variable onto ``device``. Returns the number moved.

        `nn.Module.to()` -- and therefore Lightning's device migration -- reaches only the
        variables a layer registered in its `_torch_params`, and an HGQ2 layer registers
        none of its own (see register_keras_parameters). Measured on Polaris, `.to(cuda)`
        left 684 of the CLIC model's keras Variables behind on cpu -- the own weights
        (kernel, bias, beta, ebops) of the layers that were built directly. On a single
        device nothing fails visibly, because keras' `convert_to_tensor` copies each
        stranded tensor to the GPU on every single op; under DDP it aborts the run.

        `set_keras_default_device()` is not an alternative: it only chooses where keras
        creates FUTURE tensors. Both are needed, and MPflowHGQ._sync_keras_device does both.

        The move is an in-place `.data` swap on the existing `nn.Parameter`, so parameter
        object identity -- and hence any optimizer already holding a reference -- survives,
        and the Variable is never replaced by a detached tensor. That also keeps a
        registered variable aliased to its `_torch_params` entry, so `named_parameters()`
        and `state_dict()` keep exposing the tensors the forward pass actually reads.
        """
        target = torch.empty(0, device=device).device
        moved = 0
        for var in self.keras_variables():
            value = getattr(var, "_value", None)
            if value is None or value.device == target:
                continue  # never built, or already there
            value.data = value.data.to(target)
            if value.grad is not None:
                value.grad = value.grad.to(target)
            moved += 1
        return moved

    def register_keras_parameters(self) -> int:
        """Register every keras layer's weights on the torch module tree. Returns layers newly tracked.

        MUST be called once after the lazy layers are materialized, before anything saves a
        checkpoint. `MPflowHGQ.setup` does it; a driver that materializes the model itself
        has to as well.

        A keras layer publishes its weights to torch by building
        `layer._torch_params = ParameterDict({var.path: var.value})` from `_post_build()`.
        `hgq.layers.core.base.QLayerBase._post_build` OVERRIDES that hook with assertions
        of its own and never calls `super()._post_build()`, so no HGQ2 layer ever tracks
        its weights -- not via `layer.build(shape)`, and not via `layer(x)` either
        (keras' `build_wrapper` calls `_post_build` on both paths; it is the override, not
        the call path, that breaks it). None of the QDense kernel/bias tensors therefore
        reach `state_dict()`, and a checkpoint carries the quantizer state and the norms
        but none of the network they belong to.

        Keras recovers lazily: `TorchLayer.torch_params`, `.parameters()` and
        `.named_parameters()` all call `_track_variables()` if the dict is missing. That is
        the second half of the problem -- whether a checkpoint is complete depends on
        whether something happened to traverse submodule `.parameters()` first, so the key
        set is a function of the callback list. Doing it here, once, at a defined point,
        makes the state_dict deterministic and complete; afterwards those lazy paths are
        no-ops. `torch_params` is keras' own public accessor for this, and the tracking it
        performs reuses the existing `nn.Parameter` objects, so no tensor is copied and
        nothing the optimizer already references is replaced.
        """
        tracked = 0
        for module in self.modules():
            if isinstance(module, keras.layers.Layer) and getattr(module, "_torch_params", None) is None:
                _ = module.torch_params
                tracked += 1
        return tracked

    def missing_state_keys(self, state: dict) -> list[str]:
        """Keys this model needs that ``state`` does not carry (checkpoint completeness check)."""
        return sorted(set(self.state_dict().keys()) - set(state.keys()))
