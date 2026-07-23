"""Sample description for pinhole-effect simulations.

The Sample describes the specimen whose transmission is being simulated:
which element and absorption edge are under study, the sample's x-ray
absorption coefficient mu(E), and a probability density describing the
distribution of thicknesses across the sample.

The pinhole effect arises because transmission is exponential in
thickness: for an inhomogeneous sample the *average* transmission,
<T>(E) = <exp(-mu(E) * t)> averaged over the thickness distribution, is
not equal to exp(-mu(E) * <t>). This class supplies the two ingredients
of that average -- mu(E) and the thickness probability density.

Note
----
The Sample only *stores* the thickness probability distribution p(t). It
deliberately performs no evaluation or sampling of thickness: thickness
is nonuniform across the sample surface, so both evaluating p(t) and
drawing thicknesses for particular ray paths are the responsibility of
the Experiment class that simulates a measurement.

Conventions / units
-------------------
* Photon energy is in eV (matching Source).
* mu(E) is a linear absorption coefficient. Its units must be the
  reciprocal of the thickness units so that mu * t is dimensionless
  (e.g. mu in cm^-1 with thickness in cm).
* Thickness is a path length through the sample and should be >= 0
  (the thickness density should have support only on t >= 0).
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Light validation tables
# ----------------------------------------------------------------------
_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

_EDGES = {
    "K",
    "L1", "L2", "L3",
    "M1", "M2", "M3", "M4", "M5",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
}


class Sample:
    """A specimen for a pinhole-effect XAFS transmission simulation.

    Parameters
    ----------
    element : str
        Chemical symbol of the element under study, e.g. 'Cu'.
    edge : str
        Absorption edge under study, e.g. 'K', 'L3'.
    mu : callable or tuple(array, array)
        The sample's linear absorption coefficient as a function of
        energy. Either:
          * a callable mu(E) returning the coefficient at energy E (eV), or
          * a pair (energies, mus) of tabulated values (energies in eV,
            strictly increasing) that will be linearly interpolated.
    thickness_pdf : scipy.stats frozen distribution
        The probability distribution p(t) of sample thickness, e.g.
        scipy.stats.lognorm(s=0.3, scale=1e-3). Must provide .pdf and
        .rvs. Its support should be restricted to t >= 0. The Sample only
        *stores* this distribution; evaluating and sampling thickness is
        the responsibility of the Experiment class.

    Attributes
    ----------
    element : str
    edge : str
    thickness_pdf : object
        The thickness distribution p(t) exactly as supplied.
    """

    def __init__(self, element, edge, mu, thickness_pdf):
        # --- element / edge validation ---
        element = str(element)
        if element not in _ELEMENTS:
            raise ValueError(f"unknown element symbol {element!r}")
        edge = str(edge)
        if edge not in _EDGES:
            raise ValueError(
                f"unrecognized edge {edge!r}; expected one of "
                f"{sorted(_EDGES)}")
        self.element = element
        self.edge = edge

        # --- mu(E) ---
        self._mu_func, self._mu_range = self._build_mu(mu)

        # --- thickness probability density (stored, not evaluated) ---
        self.thickness_pdf = self._validate_thickness_pdf(thickness_pdf)

    # ------------------------------------------------------------------
    # mu(E)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mu(mu):
        """Return (mu_func, (Emin, Emax)-or-None) from the supplied mu."""
        if callable(mu):
            return mu, None
        # otherwise expect a (energies, mus) pair
        try:
            energies, mus = mu
        except (TypeError, ValueError):
            raise ValueError(
                "mu must be a callable or a (energies, mus) pair")
        energies = np.asarray(energies, dtype=float)
        mus = np.asarray(mus, dtype=float)
        if energies.shape != mus.shape or energies.ndim != 1:
            raise ValueError("energies and mus must be 1-D arrays of equal "
                             "length")
        if np.any(np.diff(energies) <= 0):
            raise ValueError("energies must be strictly increasing")
        if np.any(mus < 0):
            raise ValueError("absorption coefficient mu must be nonnegative")
        Emin, Emax = float(energies[0]), float(energies[-1])

        def mu_func(E):
            E = np.asarray(E, dtype=float)
            if np.any(E < Emin) or np.any(E > Emax):
                raise ValueError(
                    f"energy outside tabulated range "
                    f"[{Emin:.1f}, {Emax:.1f}] eV")
            return np.interp(E, energies, mus)

        return mu_func, (Emin, Emax)

    @property
    def mu_energy_range(self):
        """(Emin, Emax) in eV over which mu is defined, or None if mu is a
        callable with no declared domain."""
        return self._mu_range

    def mu(self, E):
        """Linear absorption coefficient at photon energy E (eV).

        Returns a float for scalar E, or an array for array-like E.
        """
        result = self._mu_func(E)
        return float(result) if np.isscalar(E) or np.ndim(E) == 0 else \
            np.asarray(result, dtype=float)

    # ------------------------------------------------------------------
    # Thickness probability density (stored only -- never evaluated here)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_thickness_pdf(dist):
        """Check that `dist` is a usable probability distribution p(t).

        The Sample only stores p(t). All evaluation and sampling of
        thickness is done by the Experiment class, which needs both the
        density (.pdf) and a sampler (.rvs) -- so a scipy.stats frozen
        distribution is required here.
        """
        if hasattr(dist, "pdf") and hasattr(dist, "rvs"):
            return dist
        raise ValueError(
            "thickness_pdf must be a probability distribution object with "
            ".pdf and .rvs methods, e.g. a scipy.stats frozen distribution "
            "such as scipy.stats.lognorm(s=0.3, scale=1e-3). A bare "
            "callable cannot be sampled by the Experiment class.")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        mu_txt = ("tabulated " + f"[{self._mu_range[0]:.0f}-"
                  f"{self._mu_range[1]:.0f} eV]"
                  if self._mu_range else "callable")
        return (f"Sample(element={self.element!r}, edge={self.edge!r}, "
                f"mu={mu_txt})")
