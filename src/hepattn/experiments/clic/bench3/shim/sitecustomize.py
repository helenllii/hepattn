"""Make `import hepattn` load from $HEPATTN_SRC, inside an env whose editable install points at another checkout.

WHY THIS EXISTS
The production `clic` pixi env (hepattn-phat/.pixi/envs/clic) has hepattn installed editable with scikit-build-core.
Its `_editable_skbc_hepattn.pth` puts a redirecting finder at sys.meta_path[0] that maps every `hepattn.*` module to
hepattn-phat/src. That finder runs BEFORE sys.path, so PYTHONPATH=<bench>/src is silently ignored and the bench
would time hepattn-phat's code (and fail to import modules that exist only here, e.g. `hepattn.experiments.clic.a4`).

WHAT IT DOES
Python imports `sitecustomize` after processing .pth files, so the finder installed here goes in front of the
scikit-build one. It answers ONLY for `hepattn` and `hepattn.*`, resolving them under $HEPATTN_SRC with the stock
PathFinder. Everything else -- in particular `lap1015`, whose compiled `_core` lives in the hepattn-phat build --
falls through to the env's own finders unchanged. Nothing in the env is modified.

It is strict: a `hepattn.*` module that is missing under $HEPATTN_SRC raises ImportError instead of falling through
to the other checkout, so the two code bases can never be mixed in one process.

Activated by putting this directory on PYTHONPATH and setting HEPATTN_SRC; inherited by Lightning's DDP subprocesses
and the DataLoader workers through the environment. Inert when HEPATTN_SRC is unset.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys

_SRC = os.environ.get("HEPATTN_SRC")


class HepattnFromSrc:
    """Meta-path finder that resolves `hepattn` and its submodules under one source root only."""

    def __init__(self, src: str) -> None:
        self.src = os.path.abspath(src)

    def find_spec(self, name, path=None, target=None):
        if name != "hepattn" and not name.startswith("hepattn."):
            return None
        search = [self.src] if name == "hepattn" else path
        spec = importlib.machinery.PathFinder.find_spec(name, search, target)
        if spec is None:
            raise ImportError(f"[bench3 shim] {name} not found under HEPATTN_SRC={self.src}; refusing to fall back to another checkout", name=name)
        return spec


if _SRC and not any(isinstance(f, HepattnFromSrc) for f in sys.meta_path):
    sys.meta_path.insert(0, HepattnFromSrc(_SRC))
    # A module already imported from elsewhere (only possible if something imported hepattn during .pth
    # processing) would defeat the finder: drop it so the next import goes through us.
    for _mod in [m for m in sys.modules if m == "hepattn" or m.startswith("hepattn.")]:
        del sys.modules[_mod]
