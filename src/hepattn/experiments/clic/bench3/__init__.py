"""bench3: training-time-per-epoch comparison of MaskFormer, MaskFormer+Linformer and MaskFormer+PHAT/GMP on Polaris.

See README.md in this directory. Nothing here modifies production code.
"""

from hepattn.experiments.clic.bench3.callback import TrainingTimeBenchmark

__all__ = ["TrainingTimeBenchmark"]
