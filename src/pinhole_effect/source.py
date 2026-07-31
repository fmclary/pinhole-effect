"""X-ray source description for pinhole-effect simulations.

The Source is a point source emitting a cone-shaped beam of half-angle
`alpha`. The cone illuminates a circular cross-section on the sample.
Assuming that cross-section is uniformly illuminated, and (in the
small-angle regime relevant to x-rays) that a ray's radial position is
proportional to its angle, the probability of a ray landing in an
annulus scales with the annulus area ~ theta d(theta). The ray-angle
distribution is therefore a linear ramp:

    p(theta) = 2 * theta / alpha**2,   for theta in [0, alpha],

zero outside [0, alpha]. The density is zero on-axis and rises linearly
to its maximum at the cone edge theta = alpha, so larger-angle rays are
weighted more heavily.

Naming
------
* `alpha` : the user-supplied maximum ray angle (the cone half-angle,
  i.e. the angular spread of the source), in radians.
* `theta` : the variable ray angle drawn from the distribution, in
  radians, with 0 <= theta <= alpha.

Conventions
-----------
* Photon energy is in eV.
* Angles are in radians, measured from the nominal beam axis.
"""

from __future__ import annotations

import numpy as np


class Source:
    """A point x-ray source emitting a uniformly illuminated cone.

    Parameters
    ----------
    energy : float
        Photon energy in eV. Must be positive.
    alpha : float
        Maximum ray angle (cone half-angle / angular spread) in radians.
        Must be positive.
    rng : numpy.random.Generator, optional
        Random generator for reproducible simulations. If omitted, a
        fresh default generator is used.

    Attributes
    ----------
    energy : float
        Photon energy in eV.
    alpha : float
        Maximum ray angle (radians).
    """

    def __init__(self, energy: float, alpha: float,
                rng: np.random.Generator | None = None):
        if energy <= 0:
            raise ValueError(f"energy must be positive (got {energy} eV)")
        if alpha <= 0:
            raise ValueError(f"half-angle must be positive (got {alpha} rad)")
        if alpha >= np.pi/2:
            raise ValueError(f'half-angle must be less than pi/2 (got {alpha} rad)')
    
        self.energy = float(energy)
        self.alpha = float(alpha)
        self._rng = rng if rng is not None else np.random.default_rng()

    def generate_direction(self, size=None):
        """Randomly generate ray angle(s) theta in [0, alpha].

        Uses exact inverse-transform sampling of p(theta) = 2 theta/alpha^2:
        with u uniform on [0, 1), theta = alpha * sqrt(u).

        Parameters
        ----------
        size : int or None, optional
            Number of angles to draw. None (default) returns a single
            float; an integer returns an ndarray of that length.

        Returns
        -------
        float or ndarray
            Ray angle(s) theta in radians, in [0, alpha].
        """
        u = self._rng.random(size=size)
        theta = self.alpha * np.sqrt(u)
        return float(theta) if size is None else theta

    def angular_pdf(self, theta):
        """Evaluate the normalized angular density p(theta) = 2 theta/alpha^2.

        Returns zero outside [0, alpha]. Accepts scalar or array input.
        Useful for plotting or deterministic (quadrature) integration.
        """
        theta = np.asarray(theta, dtype=float)
        pdf = np.where((theta >= 0.0) & (theta <= self.alpha),
                       2.0 * theta / self.alpha**2, 0.0)
        return float(pdf) if pdf.ndim == 0 else pdf

    def __repr__(self) -> str:
        return (f"Source(energy={self.energy:.1f} eV, "
                f"alpha={self.alpha:.4g} rad)")
