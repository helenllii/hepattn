"""Gates for the opt-in DataLoader options on PflowDataModule.

`persistent_workers` and `prefetch_factor` are the one change to production code in this branch,
so the thing to pin is that they are genuinely opt-in: an existing config that sets neither must
build exactly the DataLoader it built before, and setting either must not make a `num_workers: 0`
run raise -- torch rejects both arguments outright when there are no worker processes.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from hepattn.experiments.clic.pflow_data import PflowDataModule

TORCH_DEFAULT_PREFETCH = 2


def _module(**kwargs) -> PflowDataModule:
    """A datamodule that is never `setup()`, so no ROOT file is opened."""
    defaults = {
        "train_path": "/nonexistent/train.root",
        "valid_path": "/nonexistent/val.root",
        "batch_size": 8,
        "num_workers": 2,
        "num_train": -1,
        "num_val": -1,
        "num_test": -1,
        "scale_dict_path": "/nonexistent/scale.yaml",
    }
    return PflowDataModule(**(defaults | kwargs))


def _loader(dm: PflowDataModule):
    return dm.get_dataloader(stage="fit", dataset=TensorDataset(torch.zeros(16, 2)), shuffle=False)


def test_defaults_are_unchanged():
    """No config change -> the DataLoader torch would have built before."""
    dl = _loader(_module())
    assert dl.num_workers == 2
    assert dl.persistent_workers is False
    assert dl.prefetch_factor == TORCH_DEFAULT_PREFETCH


def test_options_are_forwarded_when_requested():
    dl = _loader(_module(persistent_workers=True, prefetch_factor=4))
    assert dl.persistent_workers is True
    assert dl.prefetch_factor == 4


def test_only_persistent_workers_leaves_prefetch_at_the_torch_default():
    """Setting one must not drag the other off its default."""
    dl = _loader(_module(persistent_workers=True))
    assert dl.persistent_workers is True
    assert dl.prefetch_factor == TORCH_DEFAULT_PREFETCH


@pytest.mark.parametrize("opts", [{}, {"persistent_workers": True}, {"prefetch_factor": 4}, {"persistent_workers": True, "prefetch_factor": 4}])
def test_num_workers_zero_never_raises(opts):
    """Torch raises for either option when num_workers == 0, so neither may be passed there."""
    dl = _loader(_module(num_workers=0, **opts))
    assert dl.num_workers == 0
    assert dl.persistent_workers is False


def test_batch_size_and_shuffle_still_reach_the_loader():
    dm = _module(batch_size=4)
    dl = dm.get_dataloader(stage="fit", dataset=TensorDataset(torch.zeros(16, 2)), shuffle=True)
    assert dl.batch_size == 4
    assert dl.sampler is not None
