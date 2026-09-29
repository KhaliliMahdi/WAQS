"""WAQS: Weight Absorption for Quadratic probing and affine Steering."""

from .extract import (
    ActivationExtractor,
    MultiPointActivationExtractor,
    NormActivationExtractor,
    load_activations,
    load_model,
    save_activations,
)
from .probes import (
    DiagonalQuadraticProbe,
    DiagonalQuadraticProbeTrainer,
    GDAQuadraticProbe,
    LinearProbe,
    QuadraticProbe,
    QuadraticProbeTrainer,
    load_probe,
    save_probe,
)
from .steering import (
    AbsorbedRMSNormSteering,
    AngularSteeringHook,
    DynamicQuadraticSteeringHook,
    SphericalSteeringHook,
    SteeringHook,
    compute_mean_diff,
    get_linear_probe_direction,
    get_quadratic_probe_direction,
)
from .writeback import (
    inject_linear_direction,
    inject_quadratic_probe,
    save_model,
    write_absorbed_rmsnorm,
)

__all__ = [
    "ActivationExtractor",
    "MultiPointActivationExtractor",
    "NormActivationExtractor",
    "load_activations",
    "load_model",
    "save_activations",
    "DiagonalQuadraticProbe",
    "DiagonalQuadraticProbeTrainer",
    "GDAQuadraticProbe",
    "LinearProbe",
    "QuadraticProbe",
    "QuadraticProbeTrainer",
    "load_probe",
    "save_probe",
    "AbsorbedRMSNormSteering",
    "AngularSteeringHook",
    "DynamicQuadraticSteeringHook",
    "SphericalSteeringHook",
    "SteeringHook",
    "compute_mean_diff",
    "get_linear_probe_direction",
    "get_quadratic_probe_direction",
    "inject_linear_direction",
    "inject_quadratic_probe",
    "save_model",
    "write_absorbed_rmsnorm",
]
