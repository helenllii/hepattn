"""A4: GMP placement and GMP implementation variants for the CLIC PHAT encoder.

Nothing here modifies production code. `PlacementEncoder` is a subclass of
`hepattn.models.encoder.Encoder`, and every entry of `GMP_IMPLS` is a strict subclass of
`hepattn.models.encoder.GeometricMessagePassing`.
"""

from hepattn.experiments.clic.a4.encoder import PlacementEncoder
from hepattn.experiments.clic.a4.gmp_impl import GMP_IMPLS, GMPEinsumOpt
from hepattn.experiments.clic.a4.gmp_weights_first import GMPWeightsFirst, GMPWeightsFirstLoop

__all__ = [
    "GMP_IMPLS",
    "GMPEinsumOpt",
    "GMPWeightsFirst",
    "GMPWeightsFirstLoop",
    "PlacementEncoder",
]
