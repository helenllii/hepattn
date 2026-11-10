"""An Encoder subclass that selects which layers run GMP, and which GMP forward they run.

WHY A SUBCLASS AND NOT AN EDIT TO encoder.py
The production `Encoder` applies GMP in every layer and has no way to select a subset. Adding the
argument to `Encoder` itself would change a class that several live trainings import, so the
selection lives here instead. Production code is not modified.

EQUIVALENCE TO BUILDING THE ENCODER WITHOUT THOSE MODULES
Dropping a layer's GMP submodule after construction is equivalent to never building it:
`PatchEncoderLayer.forward` guards every use with `if self.gmp is not None`, and assigning None
over an nn.Module attribute removes it from `_modules`, so its parameters leave `state_dict()`,
`parameters()` and therefore the optimizer. It is not merely bypassed.

The GMP geometry is still built once per forward by `Encoder._patch_forward` whenever `self.gmp`
is set, and is shared by every layer, so a subset removes only the per-layer GMP call.

SELECTING THE GMP IMPLEMENTATION
`gmp_impl` chooses which GMP forward the kept layers run. It retypes each GMP submodule in place
(`layer.gmp.__class__ = ...`) rather than rebuilding it. That is deliberate, and it is what makes
a comparison against production controlled: every candidate in `gmp_impl.GMP_IMPLS` is a strict
subclass of `GeometricMessagePassing` that declares no `__init__` and no new state, so retyping
changes the forward arithmetic and nothing else -- not the parameters, not their creation order,
not the global RNG stream, not the `state_dict` keys. Constructing a replacement module and
copying weights into it would instead advance the RNG and perturb every later random
initialization.
"""

from __future__ import annotations

from hepattn.experiments.clic.a4.gmp_impl import GMP_IMPLS
from hepattn.models.encoder import Encoder


class PlacementEncoder(Encoder):
    """Production Encoder with GMP restricted to the layers named in `gmp_layers`.

    Args:
        gmp_layers: 0-based layer indices that keep a GMP module. None (the default) keeps GMP in
            every layer, i.e. exactly the production PHAT+GMP architecture. Requires `gmp`.
        gmp_impl: Which GMP forward the kept layers run, a key of `gmp_impl.GMP_IMPLS`.
            "production" (the default) leaves the production module exactly as built.
        **kwargs: Passed straight to `hepattn.models.encoder.Encoder`.

    Raises:
        ValueError: If `gmp_layers` repeats or goes out of range, if it is empty, if it is given
            without `gmp`, or if `gmp_impl` is not a key of `GMP_IMPLS`.
        RuntimeError: If the pruning or the retyping did not take effect.
    """

    def __init__(self, gmp_layers: list[int] | None = None, gmp_impl: str = "production", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gmp_layers = sorted(gmp_layers) if gmp_layers is not None else None
        self.gmp_impl = gmp_impl
        if gmp_layers is None:
            # No placement restriction: apply the implementation to every layer and stop.
            self._apply_gmp_impl(gmp_impl)
            return
        if self.gmp is None:
            raise ValueError("gmp_layers selects which layers get GMP, so it needs gmp to be set")
        if len(set(gmp_layers)) != len(gmp_layers):
            raise ValueError(f"gmp_layers must not repeat an index, got {gmp_layers}")
        if any(i < 0 or i >= self.num_layers for i in gmp_layers):
            raise ValueError(f"gmp_layers indices must be in [0, {self.num_layers - 1}], got {gmp_layers}")
        if not gmp_layers:
            # forward() would still demand x_coords while no layer could consume them
            raise ValueError("gmp_layers is empty: unset gmp instead to build an encoder without GMP")
        keep = set(self.gmp_layers)
        for i, layer in enumerate(self.layers):
            if i not in keep:
                layer.gmp = None
        kept = [i for i, layer in enumerate(self.layers) if layer.gmp is not None]
        if kept != self.gmp_layers:
            raise RuntimeError(f"GMP ended up in layers {kept}, expected {self.gmp_layers}")
        # After pruning, so the provenance line names the layers that actually keep a GMP module.
        self._apply_gmp_impl(gmp_impl)

    def _apply_gmp_impl(self, gmp_impl: str) -> None:
        """Retype every GMP submodule to the requested implementation. See the module docstring.

        Raises:
            ValueError: If `gmp_impl` is unknown, or needs `gmp` and `gmp` is unset.
            RuntimeError: If any kept layer did not take the new class.
        """
        if gmp_impl not in GMP_IMPLS:
            raise ValueError(f"gmp_impl must be one of {sorted(GMP_IMPLS)}, got {gmp_impl!r}")
        cls = GMP_IMPLS[gmp_impl]
        if gmp_impl != "production" and self.gmp is None:
            raise ValueError("gmp_impl selects a GMP forward, so it needs gmp to be set")
        self.gmp_impl = gmp_impl
        for layer in self.layers:
            if getattr(layer, "gmp", None) is not None:
                layer.gmp.__class__ = cls
        wrong = [i for i, layer in enumerate(self.layers) if getattr(layer, "gmp", None) is not None and type(layer.gmp) is not cls]
        if wrong:
            raise RuntimeError(f"layers {wrong} did not take gmp_impl {gmp_impl!r}")
        # Provenance: every training log states which GMP forward actually ran, so a run can be
        # audited from its log alone. The class is not recoverable from the checkpoint (state_dict
        # keys are production's by design), which is exactly why this is printed.
        retyped = [i for i, layer in enumerate(self.layers) if getattr(layer, "gmp", None) is not None]
        print(f"[a4] gmp_impl={gmp_impl} ({cls.__module__}.{cls.__qualname__}) in encoder layers {retyped}", flush=True)
