"""Experiment classes for XAFS pinhole-effect simulations.

An Experiment couples a Source (which supplies ray angles) and a Sample
(which supplies mu(E) as a two-column table and the thickness
distribution p(t)) and simulates the transmitted intensity over the
sample's tabulated energies.

Energy grid
-----------
The energy axis of the simulation IS the energy column of the sample's
mu table -- no interpolation and no resampling. mu at each energy is
therefore read directly from the table's second column, aligned by
construction. An optional `energy_range` trims the spectrum to a
sub-window of the table (see the constructor).

Physics
-------
Beer-Lambert law for a single ray of angle theta passing through a slab
of nominal thickness t:

    I(E) = I0 * exp(-mu(E) * t / cos(theta))

with I0 = 1, so the transmitted intensity is a transmittance in [0, 1].
The 1/cos(theta) factor is the path-length enhancement for an inclined
ray. (At true x-ray divergences this correction is ~theta^2/2 and hence
negligible, but it is retained so that larger angles -- or a future
exact, non-small-angle Source distribution -- are handled correctly.)

The pinhole effect appears when averaging over an inhomogeneous sample:

    <I>(E) = (1/N) sum_i exp(-mu(E) * t_i / cos(theta_i))

which is *not* equal to exp(-mu(E) * <t>), the spectrum an ideal uniform
sample of the same mean thickness would give.

Class structure
---------------
`Experiment` is the base class and acts as a factory: calling
`Experiment(...)` returns one of

  * SingleRayExperiment  (n_rays == 1) -- exposes generate_ray,
    generate_thickness, transmission_table. Intended for debugging and
    exploration: a single ray is NOT a pinhole-averaged spectrum, only
    Beer-Lambert for one random (t, theta) pair.

  * FullSampleExperiment (n_rays > 1)  -- exposes run / transmission_table
    over many rays, plus compare_spectra.

Both subclasses share plot_spectrum and plot_absorbance.

Conventions / units
-------------------
* Photon energy is in eV.
* mu and thickness must have reciprocal units so that mu * t is
  dimensionless.
* Angles are in radians.
* Source.energy is treated as metadata and does not affect the spectrum.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
        Supplies mu (a two-column energy/mu DataFrame) and the thickness
        distribution p(t).
    energy_range : tuple(float, float) or None, optional
        (Emin, Emax) sub-window of the spectrum, in eV. If None (default)
        the full tabulated energy column is used. If given, it must be a
        subset of the table's energy span; Emin and Emax are each snapped
        to the nearest tabulated energy (via argmin), and the simulation
        uses exactly the table rows in that window -- no new energies are
        created.
    n_rays : int, optional
        Number of rays. 1 selects SingleRayExperiment; >1 selects
        FullSampleExperiment. Default 1.
    rng : numpy.random.Generator or int, optional
        Random generator or seed, threaded into both the angle and the
        thickness draws so that a whole run is reproducible. If given, it
        overrides the source's own generator for this experiment.

    Attributes
    ----------
    source, sample : objects as supplied
    energy_range : tuple(float, float)
        The (snapped) energy window actually used, in eV.
    n_rays : int
    energies : ndarray
        The energy grid of the output spectrum (eV) -- a view of the
        table's energy column, possibly trimmed.
    rng : numpy.random.Generator
    """

    # ------------------------------------------------------------------
    # Factory dispatch
    # ------------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        if cls is Experiment:
            if "n_rays" in kwargs:
                n_rays = kwargs["n_rays"]
            elif len(args) > 3:            # source, sample, energy_range, n_rays
                n_rays = args[3]
            else:
                n_rays = 1
            n_rays_int = _validate_n_rays(n_rays)
            cls = SingleRayExperiment if n_rays_int == 1 else FullSampleExperiment    # noqa: PLW0642 -- intentional factory dispatch
        return super().__new__(cls)

    def __init__(self, source, sample, energy_range=None,
                 n_rays: int = 1, rng=None):
        n_rays = _validate_n_rays(n_rays)

        self.source = source
        self.sample = sample
        self.n_rays = n_rays

        # Pull the mu table (two columns: energy, mu) as plain arrays.
        mu_df = sample.mu
        table_E = np.asarray(mu_df.iloc[:, 0], dtype=float)
        table_mu = np.asarray(mu_df.iloc[:, 1], dtype=float)

        # Select the energy grid: full table, or a snapped sub-window.
        i0, i1 = self._resolve_window(table_E, energy_range)
        self.energies = table_E[i0:i1 + 1]
        self._mu_grid = table_mu[i0:i1 + 1]
        self.energy_range = (float(self.energies[0]), float(self.energies[-1]))

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

    # ------------------------------------------------------------------
    # Energy-window resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_window(table_E, energy_range):
        """Return (i0, i1) index bounds into table_E for the requested window.

        None -> the full table. Otherwise (Emin, Emax) must be a subset of
        [table_E[0], table_E[-1]]; each endpoint is snapped to the nearest
        tabulated energy. Raises ValueError on a malformed, out-of-range,
        or degenerate (<2 point) window.
        """
        n = table_E.size
        if n < 2:
            raise ValueError(
                "the sample's mu table must contain at least two energies")
        if energy_range is None:
            return 0, n - 1

        try:
            emin, emax = energy_range
        except (TypeError, ValueError):
            raise ValueError("energy_range must be a (Emin, Emax) pair or None")
        emin, emax = float(emin), float(emax)
        if emax <= emin:
            raise ValueError(
                f"energy_range must have Emax > Emin (got {emin}, {emax})")

        lo, hi = float(table_E[0]), float(table_E[-1])
        if emin < lo or emax > hi:
            raise ValueError(
                f"energy_range [{emin:.1f}, {emax:.1f}] eV must be a subset of "
                f"the table's energy range [{lo:.1f}, {hi:.1f}] eV")

        # Snap each endpoint to the nearest tabulated energy.
        i0 = int(np.argmin(np.abs(table_E - emin)))
        i1 = int(np.argmin(np.abs(table_E - emax)))
        if i1 < i0:
            i0, i1 = i1, i0
        if i1 - i0 < 1:
            raise ValueError(
                f"energy_range [{emin:.1f}, {emax:.1f}] eV selects fewer than "
                f"two tabulated points; widen the range or use a finer table")
        return i0, i1

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
    # Shared table export method
    # ------------------------------------------------------------------
    def save_table(table: np.ndarray, filename: str):
        """Save transmission table to local directory as `.csv` file."""
        return table.tofile(filename, sep=',')
        
    # ------------------------------------------------------------------
    # Shared plotting (returns figures; never calls plt.show())
    # ------------------------------------------------------------------
    def plot_spectrum(self, ax=None, **plot_kw):
        """Plot transmitted intensity vs energy. Returns (fig, ax)."""
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
        """Plot absorbance ln(I0/I) vs energy. Returns (fig, ax)."""
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
                f"energy_range=({self.energy_range[0]:.1f}, "
                f"{self.energy_range[1]:.1f}) eV, "
                f"n_points={self.energies.size}, n_rays={self.n_rays})")


class SingleRayExperiment(Experiment):
    """A single-ray experiment (n_rays == 1).

    Exposes the individual ray primitives. A single ray gives
    Beer-Lambert transmission for one random (thickness, angle) pair --
    it is NOT a pinhole-averaged spectrum. Use FullSampleExperiment for
    that. Intended for debugging and exploration.
    """

    def __init__(self, source, sample, energy_range=None,
                 n_rays: int = 1, rng=None):
        if _validate_n_rays(n_rays) != 1:
            raise ValueError(
                f"SingleRayExperiment requires n_rays == 1 (got {n_rays}); "
                f"use FullSampleExperiment for multiple rays")
        super().__init__(source, sample, energy_range, 1, rng)
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

        Returns
        -------
        ndarray of shape (M, 2): column 0 energy (eV), column 1 I/I0.
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

    def __init__(self, source, sample, energy_range=None,
                 n_rays: int = 2, rng=None):
        if _validate_n_rays(n_rays) < 2:
            raise ValueError(
                f"FullSampleExperiment requires n_rays > 1 (got {n_rays}); "
                f"use SingleRayExperiment for a single ray")
        super().__init__(source, sample, energy_range, int(n_rays), rng)
        self._intensity = None

    def run(self, force: bool = False):
        """Simulate all rays and return the averaged transmitted intensity.

        Rays are drawn up front, then the (n_rays x n_energies) work array
        is accumulated in blocks bounded by _MAX_BLOCK_ELEMENTS, so the
        result is independent of the block size.

        Returns
        -------
        ndarray of shape (M,): average transmitted intensity on the grid.
        """
        if self._intensity is not None and not force:
            return self._intensity

        n_energies = self.energies.size

        # Draw every ray's angle and thickness up front (two length-N
        # vectors, cheap). Doing this before any chunking makes the result
        # independent of the block size chosen below.
        thetas = np.atleast_1d(self._draw_angles(size=self.n_rays))
        thicknesses = np.atleast_1d(self._draw_thicknesses(size=self.n_rays))

        block = max(1, min(self.n_rays,
                           _MAX_BLOCK_ELEMENTS // max(1, n_energies)))

        total = np.zeros(n_energies, dtype=float)
        for start in range(0, self.n_rays, block):
            stop = min(start + block, self.n_rays)
            total += self._transmitted_intensity(
                thicknesses[start:stop], thetas[start:stop]).sum(axis=0)

        self._intensity = total / self.n_rays
        return self._intensity

    def transmission_table(self):
        """Return the averaged (energy, transmitted intensity) table.

        Returns
        -------
        ndarray of shape (M, 2): column 0 energy (eV), column 1 mean I/I0.
        """
        return np.column_stack((self.energies, self.run()))

    def compare_spectra(self, assumed_thickness=None, ax=None,
                        absorbance: bool = False):
        """Plot the simulated spectrum against an ideal uniform sample.

        The ideal curve uses a single assumed thickness (by default the
        mean <t> of the sample's thickness distribution) and theta = 0,
        so the only difference between the curves is the sample
        inhomogeneity -- i.e. the pinhole effect itself.

        Returns
        -------
        (fig, ax)
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
