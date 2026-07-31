"""pinhole_effect: simulate the pinhole effect in XAFS.

This package models how the average x-ray transmission through a sample is
distorted by thickness inhomogeneity -- the "pinhole effect" in x-ray
absorption fine structure (XAFS) -- and compares the result against the
spectrum of an equivalent uniform-thickness sample.

Public API
----------
Source
    A point x-ray source emitting a uniformly illuminated cone; supplies
    ray angles drawn from its angular distribution.
Sample
    A specimen described by its element, absorption edge, mu(E) table, and
    a thickness probability distribution p(t).
Experiment
    Factory that couples a Source and a Sample and simulates transmission.
    Returns a SingleRayExperiment (n_rays == 1) or a FullSampleExperiment
    (n_rays > 1).
SingleRayExperiment, FullSampleExperiment
    The two concrete experiment types (usually obtained via Experiment).

Example
-------
>>> import numpy as np, pandas as pd
>>> from scipy import stats
>>> from pinhole_effect import Source, Sample, Experiment
>>> E = np.linspace(8900, 9100, 401)
>>> mu = 1.0 + 3.0 / (1 + np.exp(-(E - 8979) / 2.0))
>>> sample = Sample("Cu", "K", pd.DataFrame({0: E, 1: mu}),
...                 thickness_pdf=stats.lognorm(s=0.4, scale=0.8))
>>> source = Source(energy=8979.0, alpha=1e-4)
>>> exp = Experiment(source, sample, energy_range=(8950, 9050), n_rays=100_000)
>>> spectrum = exp.run()            # doctest: +SKIP
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .source import Source
from .sample import Sample
from .experiment import (
    Experiment,
    SingleRayExperiment,
    FullSampleExperiment,
)

try:
    __version__ = version("pinhole-effect")
except PackageNotFoundError:
    # Package is not installed (e.g. running from a source checkout without
    # an editable install); fall back to a placeholder.
    __version__ = "0.0.0"

__all__ = [
    "Source",
    "Sample",
    "Experiment",
    "SingleRayExperiment",
    "FullSampleExperiment",
    "__version__",
]