"""Experiment classes for XAFS pinhole-effect simulations.

An Experiment couples a Source (which supplies ray angles) and a Sample
(which supplies mu(E) and the thickness distribution p(t)) and simulates
the transmitted intensity over a range of photon energies.

Physics
-------
Beer-Lambert law for a single ray of angle theta passing through a slab
of nominal thickness t:

    I(E) = I0 * exp(-mu(E) * t / cos(theta))

with I0 = 1, so the transmitted intensity is a transmittance in [0, 1].
The factor 1/cos(theta) is the path-length enhancement for an inclined
ray. (At true x-ray divergences this correction is ~theta^2/2 and hence
negligible, but it is retained so that larger angles -- or a future
exact, non-small-angle Source distribution -- are handled correctly.)

The pinhole effect appears when averaging over an inhomogeneous sample:

    <I>(E) = (1/N) sum_i exp(-mu(E) * t_i / cos(theta_i))

which is *not* equal to exp(-mu(E) * <t>), the spectrum an ideal uniform
sample of the same mean thickness would give.

Class structure
---------------
`Experiment` is the base class holding the shared attributes and the
private ray primitives. Calling `Experiment(...)` acts as a factory and
returns one of:

  * SingleRayExperiment  (n_rays == 1) -- exposes the single-ray methods
    generate_ray, generate_thickness, transmission_table. Intended for
    debugging and exploration: a single ray does NOT produce a pinhole
    spectrum, only Beer-Lambert for one random (t, theta) pair.

  * FullSampleExperiment (n_rays > 1)  -- exposes run / transmission_table
    over many rays, plus compare_spectra.

Both subclasses share plot_spectrum and plot_absorbance.

Conventions / units
-------------------
* Photon energy in eV; energy_range and energy_step in eV.
* mu and thickness must have reciprocal units so that mu * t is
  dimensionless.
* Angles in radians.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Cap on the number of elements in the (n_rays x n_energies) work array,
# so large runs are chunked instead of allocating enormous matrices.
_MAX_BLOCK_ELEMENTS = 4_000_000


class Experiment:
    """Base class: simulates transmission through an inhomogeneous sample.

    Calling ``Experiment(...)`` directly returns a SingleRayExperiment or
    a FullSampleExperiment according to `n_rays`.

    Parameters
    ----------
    source : Source
        Supplies ray angles via source.generate_direction().
    sample : Sample
        Supplies mu(E) and the thickness distribution p(t).
    energy_range : tuple(float, float)
        (Emin, Emax) of the output spectrum, in eV. Must lie within the
        energy range over which the sample's mu is defined.
    energy_step : float
        Energy step size in eV. Must be positive and no larger than the
        width of energy_range.
    n_rays : int, optional
        Number of rays. 1 selects SingleRayExperiment; >1 selects
        FullSampleExperiment. Default 1.
    rng : numpy.random.Generator or int, optional
        Random generator or seed. Threaded into both the angle and the
        thickness draws so that a whole run is reproducible. If given,
        it overrides the source's own generator for this experiment.

    Attributes
    ----------
    source, sample : objects as supplied
    energy_range : tuple(float, float)
    energy_step : float
    n_rays : int
    energies : ndarray
        The energy grid of the output spectrum (eV).
    rng : numpy.random.Generator
    """

    # ------------------------------------------------------------------
    # Factory dispatch
    # ------------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        if cls is Experiment:
            if "n_rays" in kwargs:
                n_rays = kwargs["n_rays"]
            elif len(args) > 4:
                n_rays = args[4]
            else:
                n_rays = 1
            n_rays_int = _validate_n_rays(n_rays)
            cls = SingleRayExperiment if n_rays_int == 1 else FullSampleExperiment
        return super().__new__(cls)

    def __init__(self, source, sample, energy_range, energy_step,
                 n_rays: int = 1, rng=None):
        # --- n_rays ---
        n_rays = _validate_n_rays(n_rays)

        # --- energy grid ---
        try:
            emin, emax = energy_range
        except (TypeError, ValueError):
            raise ValueError("energy_range must be a (Emin, Emax) pair")
        emin, emax = float(emin), float(emax)
        if emax <= emin:
            raise ValueError(
                f"energy_range must have Emax > Emin (got {emin}, {emax})")
        energy_step = float(energy_step)
        if energy_step <= 0:
            raise ValueError(
                f"energy_step must be positive (got {energy_step})")
        if energy_step > (emax - emin):
            raise ValueError(
                f"energy_step ({energy_step} eV) exceeds the width of "
                f"energy_range ({emax - emin} eV)")

        # --- energy range must lie inside the domain of mu ---
        mu_range = getattr(sample, "mu_energy_range", None)
        if mu_range is not None:
            mu_min, mu_max = mu_range
            if emin < mu_min or emax > mu_max:
                raise ValueError(
                    f"energy_range [{emin:.1f}, {emax:.1f}] eV lies outside "
                    f"the range over which the sample's mu is defined "
                    f"[{mu_min:.1f}, {mu_max:.1f}] eV")

        self.source = source
        self.sample = sample
        self.energy_range = (emin, emax)
        self.energy_step = energy_step
        self.n_rays = n_rays

        # Inclusive endpoint (within floating-point tolerance).
        self.energies = np.arange(emin, emax + 0.5 * energy_step, energy_step)

        # --- rng threading ---
        if rng is None:
            self.rng = getattr(source, "_rng", None) or np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)
        # Share the generator with the source so angles come from the
        # same reproducible stream.
        if hasattr(source, "_rng"):
            source._rng = self.rng

        # mu(E) is identical for every ray: evaluate it once.
        self._mu_grid = np.asarray(self.sample.mu(self.energies), dtype=float)

    # ------------------------------------------------------------------
    # Private ray primitives (shared by both subclasses)
    # ------------------------------------------------------------------
    def _draw_angles(self, size=None):
        """Draw ray angle(s) theta from the source's angular distribution."""
        return self.source.generate_direction(size=size)

    def _draw_thicknesses(self, size=None):
        """Draw sample thickness(es) t from the sample's p(t)."""
        t = self.sample.thickness_pdf.rvs(size=size, random_state=self.rng)
        if np.any(np.asarray(t) < 0):
            raise ValueError(
                "thickness distribution produced a negative thickness; its "
                "support should be restricted to t >= 0 (consider lognorm, "
                "gamma, or truncnorm)")
        return t

    def _transmitted_intensity(self, thickness, theta):
        """Beer-Lambert transmitted intensity I(E) = exp(-mu(E) t / cos theta).

        Vectorized over the stored energy grid. Scalars give a length-M
        spectrum; arrays of length N give an (N, M) matrix, one row per ray.
        """
        t = np.asarray(thickness, dtype=float)
        th = np.asarray(theta, dtype=float)
        t_eff = t / np.cos(th)                      # effective path length
        if t_eff.ndim == 0:
            return np.exp(-self._mu_grid * t_eff)
        return np.exp(-self._mu_grid[None, :] * t_eff[:, None])

    def _mean_thickness(self):
        """Mean thickness <t> of the sample's thickness distribution."""
        return float(self.sample.thickness_pdf.mean())

    def _ideal_intensity(self, thickness=None):
        """Ideal uniform-sample spectrum: exp(-mu(E) * t), theta = 0."""
        t = self._mean_thickness() if thickness is None else float(thickness)
        return np.exp(-self._mu_grid * t)

    # ------------------------------------------------------------------
    # Shared plotting (returns figures; never calls plt.show())
    # ------------------------------------------------------------------
    def plot_spectrum(self, ax=None, **plot_kw):
        """Plot transmitted intensity vs energy.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes
        """
        table = self.transmission_table()
        energies, intensity = table[:, 0], table[:, 1]
        fig, ax = _get_axes(ax)
        ax.plot(energies, intensity, **plot_kw)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Transmitted intensity  $I/I_0$")
        ax.set_title(self._plot_title())
        fig.tight_layout()
        return fig, ax

    def plot_absorbance(self, ax=None, **plot_kw):
        """Plot absorbance ln(I0/I) vs energy.

        The pinhole distortion (edge damping, reduced oscillation
        amplitude) is usually clearer in absorbance than in intensity.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes
        """
        table = self.transmission_table()
        energies, intensity = table[:, 0], table[:, 1]
        fig, ax = _get_axes(ax)
        ax.plot(energies, _absorbance(intensity), **plot_kw)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"Absorbance  $\ln(I_0/I)$")
        ax.set_title(self._plot_title())
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    def _plot_title(self):
        return (f"{self.sample.element} {self.sample.edge}-edge "
                f"({self.n_rays} ray{'s' if self.n_rays != 1 else ''})")

    def __repr__(self):
        return (f"{type(self).__name__}(element={self.sample.element!r}, "
                f"edge={self.sample.edge!r}, "
                f"energy_range={self.energy_range}, "
                f"energy_step={self.energy_step:g}, n_rays={self.n_rays})")


class SingleRayExperiment(Experiment):
    """A single-ray experiment (n_rays == 1).

    Exposes the individual ray primitives. Note that a single ray gives
    Beer-Lambert transmission for one random (thickness, angle) pair --
    it is NOT a pinhole-averaged spectrum. Use FullSampleExperiment for
    that. This class is intended for debugging and exploration.
    """

    def __init__(self, source, sample, energy_range, energy_step,
                 n_rays: int = 1, rng=None):
        if _validate_n_rays(n_rays) != 1:
            raise ValueError(
                f"SingleRayExperiment requires n_rays == 1 (got {n_rays}); "
                f"use FullSampleExperiment for multiple rays")
        super().__init__(source, sample, energy_range, energy_step, 1, rng)
        self._theta = None
        self._thickness = None

    def generate_ray(self):
        """Generate and store this ray's angle theta (radians)."""
        self._theta = self._draw_angles()
        return self._theta

    def generate_thickness(self):
        """Generate and store the sample thickness at one point."""
        self._thickness = float(self._draw_thicknesses())
        return self._thickness

    @property
    def theta(self):
        """The generated ray angle, or None if generate_ray() not yet called."""
        return self._theta

    @property
    def thickness(self):
        """The generated thickness, or None if not yet generated."""
        return self._thickness

    def transmission_table(self, regenerate: bool = False):
        """Build the (energy, transmitted intensity) table for this ray.

        Generates the angle and thickness if they have not yet been drawn.

        Parameters
        ----------
        regenerate : bool, optional
            If True, draw a fresh angle and thickness first.

        Returns
        -------
        ndarray of shape (M, 2)
            Column 0: energy (eV). Column 1: transmitted intensity I/I0.
        """
        if regenerate or self._theta is None:
            self.generate_ray()
        if regenerate or self._thickness is None:
            self.generate_thickness()
        intensity = self._transmitted_intensity(self._thickness, self._theta)
        return np.column_stack((self.energies, intensity))


class FullSampleExperiment(Experiment):
    """A many-ray experiment (n_rays > 1) producing a pinhole spectrum.

    For each of `n_rays` rays a random angle and a random thickness are
    drawn, the Beer-Lambert transmitted intensity is evaluated over the
    energy grid, and the results are averaged:

        <I>(E) = (1/N) sum_i exp(-mu(E) t_i / cos theta_i)
    """

    def __init__(self, source, sample, energy_range, energy_step,
                 n_rays: int, rng=None):
        if _validate_n_rays(n_rays) < 2:
            raise ValueError(
                f"FullSampleExperiment requires n_rays > 1 (got {n_rays}); "
                f"use SingleRayExperiment for a single ray")
        super().__init__(source, sample, energy_range, energy_step,
                         int(n_rays), rng)
        self._intensity = None

    def run(self, force: bool = False):
        """Simulate all rays and return the averaged transmitted intensity.

        Rays are processed in blocks so that the (n_rays x n_energies)
        work array stays within a bounded memory budget.

        Parameters
        ----------
        force : bool, optional
            Re-run even if a result is already cached.

        Returns
        -------
        ndarray of shape (M,)
            Average transmitted intensity on the energy grid.
        """
        if self._intensity is not None and not force:
            return self._intensity

        n_energies = self.energies.size

        # Draw every ray's angle and thickness up front (two length-N
        # vectors, cheap). Doing this before any chunking makes the result
        # independent of the block size chosen below.
        thetas = np.atleast_1d(self._draw_angles(size=self.n_rays))
        thicknesses = np.atleast_1d(self._draw_thicknesses(size=self.n_rays))

        # The (n_rays x n_energies) matrix is the memory-hungry part, so
        # accumulate it in blocks bounded by _MAX_BLOCK_ELEMENTS.
        block = max(1, min(self.n_rays,
                           _MAX_BLOCK_ELEMENTS // max(1, n_energies)))

        total = np.zeros(n_energies, dtype=float)
        for start in range(0, self.n_rays, block):
            stop = min(start + block, self.n_rays)
            # (k x M) matrix: one row per ray, one column per energy.
            total += self._transmitted_intensity(
                thicknesses[start:stop], thetas[start:stop]).sum(axis=0)

        self._intensity = total / self.n_rays
        return self._intensity

    def transmission_table(self):
        """Return the averaged (energy, transmitted intensity) table.

        Returns
        -------
        ndarray of shape (M, 2)
            Column 0: energy (eV). Column 1: mean transmitted intensity.
        """
        return np.column_stack((self.energies, self.run()))

    def compare_spectra(self, assumed_thickness=None, ax=None,
                        absorbance: bool = False):
        """Plot the simulated spectrum against an ideal uniform sample.

        The ideal curve uses a single assumed thickness (by default the
        mean <t> of the sample's thickness distribution) and theta = 0,
        so the only difference between the curves is the sample
        inhomogeneity -- i.e. the pinhole effect itself.

        Parameters
        ----------
        assumed_thickness : float, optional
            Thickness for the ideal curve. Defaults to <t>.
        ax : matplotlib Axes, optional
        absorbance : bool, optional
            If True, compare in absorbance ln(I0/I) instead of intensity.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes
        """
        t_assumed = (self._mean_thickness() if assumed_thickness is None
                     else float(assumed_thickness))
        simulated = self.run()
        ideal = self._ideal_intensity(t_assumed)

        if absorbance:
            simulated, ideal = _absorbance(simulated), _absorbance(ideal)
            ylabel = r"Absorbance  $\ln(I_0/I)$"
        else:
            ylabel = "Transmitted intensity  $I/I_0$"

        fig, ax = _get_axes(ax)
        ax.plot(self.energies, ideal, "k--", lw=1.5,
                label=fr"ideal uniform sample ($t={t_assumed:.4g}$)")
        ax.plot(self.energies, simulated, "C0-", lw=1.8,
                label=f"inhomogeneous sample ({self.n_rays} rays)")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pinhole effect: {self.sample.element} "
                     f"{self.sample.edge}-edge")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _validate_n_rays(n_rays):
    """Return n_rays as an int, rejecting non-integer or sub-1 values."""
    try:
        n_int = int(n_rays)
    except (TypeError, ValueError):
        raise ValueError(f"n_rays must be an integer (got {n_rays!r})")
    if n_int != n_rays:
        raise ValueError(f"n_rays must be an integer (got {n_rays!r})")
    if n_int < 1:
        raise ValueError(f"n_rays must be >= 1 (got {n_rays!r})")
    return n_int


def _get_axes(ax):
    """Return (fig, ax), creating them if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    return fig, ax


def _absorbance(intensity):
    """Absorbance ln(I0/I) = -ln(I) for I0 = 1, guarded against I <= 0."""
    intensity = np.asarray(intensity, dtype=float)
    return -np.log(np.clip(intensity, np.finfo(float).tiny, None))
