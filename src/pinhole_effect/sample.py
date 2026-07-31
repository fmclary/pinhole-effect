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
deliberately performs no evaluation or sampling of thickness: Thickness
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
import pandas as pd

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
    mu : whitespace-delimited text file, numpy array, or pandas DataFrame
        The sample's linear absorption coefficient as a function of
        energy. A two-column data structure whose first column is energy
        and second column is mu. Either:
          * a pandas DataFrame,  
          * a whitespace-delimited text file that is converted into a 
          pandas DataFrame, or
          * a numpy array that is converted into a pandas DataFrame.
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

    def __init__(self, element: str, edge: str, mu, thickness_pdf):
        # --- element / edge validation ---
        if element not in _ELEMENTS:
            raise ValueError(f"unknown element symbol {element!r}")
        if edge not in _EDGES:
            raise ValueError(
                f"unrecognized edge {edge!r}; expected one of "
                f"{sorted(_EDGES)}")
        self.element = element
        self.edge = edge

        # --- mu(E) ---
        self.mu = self._build_mu(mu)

        # --- thickness probability density (stored, not evaluated) ---
        self.thickness_pdf = self._validate_thickness_pdf(thickness_pdf)

    # ------------------------------------------------------------------
    # mu(E)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mu(mu, comments=None):
        '''Return pandas DataFrame where the first column is energy and the 
        second column is the absorption coefficient. Comments argument is 
        optional and should only be used if inputting a whitespace-delimited 
        text file with prefixed metadata.'''
        if isinstance(mu, pd.DataFrame):
            df_mu = mu
        elif isinstance(mu, np.ndarray):
            df_mu = pd.DataFrame(mu)
        elif isinstance(mu, str):
            df_mu = pd.read_csv(mu, sep=r'\s+', comment=comments)
        else:
            raise TypeError('argument mu must be a pandas DataFrame, a numpy '
                            'array, or a whitespace-delimited text file')
        
        if len(df_mu.columns) != 2:
            raise ValueError(f'got a table with {len(df_mu.columns)} '
                             f'column{'s' if len(df_mu.columns) != 1 else ''}; '
                             'can only build mu(E) from a table with two '
                             'columns (energy and mu)')
        energies = pd.to_numeric(df_mu.iloc[:, 0], errors="coerce")
        mus = pd.to_numeric(df_mu.iloc[:, 1], errors="coerce")
        # non-numeric entries become NaN, which the next check reports:
        if energies.isna().any() or mus.isna().any():
            raise ValueError('energy and mu columns must be numeric and not '
                             'contain missing values')
        if np.any(np.diff(energies) <= 0):
            raise ValueError("energies must be strictly increasing")
        if np.any(mus < 0):
            raise ValueError("absorption coefficient mu must be nonnegative")
        
        return df_mu

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
        mu_txt = ("tabulated " + f"[{self.mu.iloc[0,0]:.0f}-"
                  f"{self.mu.iloc[-1,0]:.0f} eV]")
        return (f"Sample(element={self.element!r}, edge={self.edge!r}, "
                f"mu={mu_txt})")
