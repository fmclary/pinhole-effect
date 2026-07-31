'''
Unit tests for the pinhole_effect package.
Covers all modules.

Runs using pytest.
'''

import os
import tempfile

import matplotlib
matplotlib.use("Agg")            # headless backend for plot tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pinhole_effect import (
    Source,
    Sample,
    Experiment,
    SingleRayExperiment,
    FullSampleExperiment,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def energies():
    return np.linspace(8900.0, 9100.0, 401)


@pytest.fixture
def mu_values(energies):
    # Smooth edge step near 8979 eV; strictly positive.
    return 1.0 + 3.0 / (1.0 + np.exp(-(energies - 8979.0) / 2.0))


@pytest.fixture
def mu_table(energies, mu_values):
    return pd.DataFrame({0: energies, 1: mu_values})


@pytest.fixture
def thick_pdf():
    # Lognormal keeps thickness strictly positive.
    return stats.lognorm(s=0.35, scale=0.5)


@pytest.fixture
def sample(mu_table, thick_pdf):
    return Sample("Cu", "K", mu_table, thickness_pdf=thick_pdf)


@pytest.fixture
def source():
    return Source(8979.0, 1e-4, rng=np.random.default_rng(7))


# ======================================================================
# Source
# ======================================================================
class TestSource:

    def test_stores_attributes(self):
        '''Ensure source is properly identifying energy and alpha
        arguments.'''
        s = Source(8979.0, 1e-4)
        assert s.energy == 8979.0
        assert s.alpha == 1e-4

    def test_rejects_nonpositive_energy(self):
        '''Raise proper ValueError upon receiving nonpositive energy
        input.'''
        # Match a stable substring of the message, not the whole (formatted)
        # string -- see test_rejects_nonpositive_alpha for why.
        with pytest.raises(ValueError, match="energy must be positive"):
            Source(0.0, 1e-4)

    def test_rejects_nonpositive_alpha(self):
        '''Raise proper ValueError upon receiving nonpositive alpha
        input.'''
        # `match` is a regex searched against str(exc). Use a stable
        # substring: the exact wording recently changed ("alpha" ->
        # "half-angle") and the message interpolates the value, so
        # asserting the full string would be brittle. Avoid regex
        # metacharacters like the parentheses in "(got ...)".
        with pytest.raises(ValueError, match="half-angle must be positive"):
            Source(8979.0, -1e-4)

    def test_rejects_alpha_at_or_above_half_pi(self):
        '''Raise proper ValueError upon receiving too large (unphysical) 
        alpha input.'''
        with pytest.raises(ValueError, match="less than pi/2"):
            Source(8979.0, np.pi / 2)

    def test_generate_direction_scalar(self):
        '''Ensure rng generates a scalar (float) theta.'''
        s = Source(8979.0, 1e-4, rng=np.random.default_rng(0))
        theta = s.generate_direction()
        assert isinstance(theta, float)

    def test_generate_direction_array_shape(self):
        '''Ensure rng generates an array of directions with the given 
        dimension.'''
        s = Source(8979.0, 1e-4, rng=np.random.default_rng(0))
        arr = s.generate_direction(size=1000)
        assert arr.shape == (1000,)

    def test_generated_angles_within_bounds(self):
        '''Ensure generated angles are between zero and alpha.'''
        alpha = 1e-4
        s = Source(8979.0, alpha, rng=np.random.default_rng(1))
        arr = s.generate_direction(size=100_000)
        assert arr.min() >= 0.0
        assert arr.max() <= alpha

    def test_angular_pdf_endpoints(self):
        '''Ensure endpoints of the angular pdf are consisent 
        with the linear function we expect using the small angle 
        approximation.'''
        alpha = 1e-4
        s = Source(8979.0, alpha)
        assert s.angular_pdf(0.0) == pytest.approx(0.0)
        assert s.angular_pdf(alpha) == pytest.approx(2.0 / alpha)

    def test_angular_pdf_zero_outside_support(self):
        '''Ensure angular pdf is zero outside the zero to alpha 
        angle range.'''
        alpha = 1e-4
        s = Source(8979.0, alpha)
        assert s.angular_pdf(-1e-6) == 0.0
        assert s.angular_pdf(alpha * 1.01) == 0.0

    def test_angular_pdf_is_linear_ramp(self):
        '''Ensure angular pdf has the correct shape within the 
        zero to alpha angle range.'''
        alpha = 1e-4
        s = Source(8979.0, alpha)
        assert s.angular_pdf(alpha / 2) == pytest.approx(0.5 * s.angular_pdf(alpha))

    def test_angular_pdf_normalized(self):
        '''Ensure angular pdf is normalized.'''
        alpha = 1e-4
        s = Source(8979.0, alpha)
        grid = np.linspace(0.0, alpha, 2001)
        integral = np.trapezoid(s.angular_pdf(grid), grid)
        assert integral == pytest.approx(1.0, abs=1e-6)

    def test_sampling_moments(self):
        '''Ensure rng direction result has the expected mean and stdev 
        relative to the analytic form.'''
        alpha = 1e-4
        s = Source(8979.0, alpha, rng=np.random.default_rng(42))
        draws = s.generate_direction(size=200_000)
        # Analytic: mean = 2a/3, std = a/sqrt(18).
        assert draws.mean() / alpha == pytest.approx(2.0 / 3.0, abs=0.01)
        assert draws.std() / alpha == pytest.approx(np.sqrt(1.0 / 18.0), abs=0.01)

    def test_reproducible_with_seed(self):
        '''Ensure reproducibility by getting equal results with the same seed 
        on two different runs.'''
        a = Source(8979.0, 1e-4, rng=np.random.default_rng(5)).generate_direction(size=500)
        b = Source(8979.0, 1e-4, rng=np.random.default_rng(5)).generate_direction(size=500)
        assert np.array_equal(a, b)

    def test_repr_does_not_raise(self):
        '''Catch exceptions in _repr_ method.'''
        assert "Source" in repr(Source(8979.0, 1e-4))


# ======================================================================
# Sample
# ======================================================================
class TestSample:

    def test_stores_element_and_edge(self, sample):
        assert sample.element == "Cu"
        assert sample.edge == "K"

    def test_rejects_unknown_element(self, mu_table, thick_pdf):
        with pytest.raises(ValueError):
            Sample("Xx", "K", mu_table, thickness_pdf=thick_pdf)

    def test_rejects_unknown_edge(self, mu_table, thick_pdf):
        with pytest.raises(ValueError):
            Sample("Cu", "Q", mu_table, thickness_pdf=thick_pdf)

    def test_mu_stored_as_dataframe(self, sample):
        assert isinstance(sample.mu, pd.DataFrame)

    def test_accepts_numpy_array(self, energies, mu_values, thick_pdf):
        arr = np.column_stack([energies, mu_values])
        s = Sample("Cu", "K", arr, thickness_pdf=thick_pdf)
        assert isinstance(s.mu, pd.DataFrame)

    def test_rejects_one_column(self, energies, thick_pdf):
        with pytest.raises(ValueError):
            Sample("Cu", "K", pd.DataFrame({0: energies}), thickness_pdf=thick_pdf)

    def test_rejects_three_columns(self, energies, mu_values, thick_pdf):
        df = pd.DataFrame({0: energies, 1: mu_values, 2: mu_values})
        with pytest.raises(ValueError):
            Sample("Cu", "K", df, thickness_pdf=thick_pdf)

    def test_rejects_missing_values(self, energies, mu_values, thick_pdf):
        mu = mu_values.copy()
        mu[10] = np.nan
        df = pd.DataFrame({0: energies, 1: mu})
        with pytest.raises(ValueError):
            Sample("Cu", "K", df, thickness_pdf=thick_pdf)

    def test_rejects_nonincreasing_energy(self, energies, mu_values, thick_pdf):
        df = pd.DataFrame({0: energies[::-1], 1: mu_values})
        with pytest.raises(ValueError):
            Sample("Cu", "K", df, thickness_pdf=thick_pdf)

    def test_rejects_negative_mu(self, energies, mu_values, thick_pdf):
        df = pd.DataFrame({0: energies, 1: mu_values - 5.0})
        with pytest.raises(ValueError):
            Sample("Cu", "K", df, thickness_pdf=thick_pdf)

    def test_rejects_bare_callable_pdf(self, mu_table):
        with pytest.raises(ValueError):
            Sample("Cu", "K", mu_table, thickness_pdf=lambda t: np.exp(-t))

    def test_stores_frozen_pdf(self, sample, thick_pdf):
        assert sample.thickness_pdf is thick_pdf

    def test_reads_whitespace_file(self, thick_pdf):
        E = np.linspace(8900.0, 9100.0, 50)
        mu = np.linspace(1.0, 4.0, 50)
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
            fh.write("energy mu\n")
            for e, m in zip(E, mu):
                fh.write(f"{e} {m}\n")
            path = fh.name
        try:
            s = Sample("Fe", "K", path, thickness_pdf=thick_pdf)
            assert isinstance(s.mu, pd.DataFrame)
            assert len(s.mu.columns) == 2
        finally:
            os.remove(path)

    def test_repr_does_not_raise(self, sample):
        assert "Cu" in repr(sample)


# ======================================================================
# Experiment -- construction & dispatch
# ======================================================================
class TestExperimentDispatch:

    def test_single_ray_dispatch(self, source, sample):
        e = Experiment(source, sample, None, 1)
        assert isinstance(e, SingleRayExperiment)

    def test_full_sample_dispatch(self, source, sample):
        e = Experiment(source, sample, None, 5000)
        assert isinstance(e, FullSampleExperiment)

    def test_both_are_experiments(self, source, sample):
        assert isinstance(Experiment(source, sample, None, 1), Experiment)
        assert isinstance(Experiment(source, sample, None, 10), Experiment)

    def test_keyword_nrays_dispatch(self, source, sample):
        e = Experiment(source, sample, n_rays=3000)
        assert isinstance(e, FullSampleExperiment)

    def test_grid_matches_table(self, source, sample, energies):
        e = Experiment(source, sample, None, 1)
        assert np.array_equal(e.energies, energies)

    def test_mu_grid_matches_table(self, source, sample, mu_values):
        e = Experiment(source, sample, None, 10)
        assert np.allclose(e._mu_grid, mu_values)

    def test_energy_range_snaps_to_nearest(self, source, sample, energies):
        e = Experiment(source, sample, (8975.3, 8984.7), 10)
        assert e.energies[0] == energies[np.argmin(np.abs(energies - 8975.3))]
        assert e.energies[-1] == energies[np.argmin(np.abs(energies - 8984.7))]

    def test_energy_range_is_contiguous_subset(self, source, sample, energies):
        e = Experiment(source, sample, (8950.0, 9050.0), 10)
        expected = energies[(energies >= e.energies[0]) & (energies <= e.energies[-1])]
        assert np.array_equal(e.energies, expected)

    def test_rejects_range_below_table(self, source, sample):
        with pytest.raises(ValueError):
            Experiment(source, sample, (8800.0, 9050.0), 10)

    def test_rejects_range_above_table(self, source, sample):
        with pytest.raises(ValueError):
            Experiment(source, sample, (8950.0, 9200.0), 10)

    def test_rejects_reversed_range(self, source, sample):
        with pytest.raises(ValueError):
            Experiment(source, sample, (9050.0, 8950.0), 10)

    def test_rejects_too_narrow_range(self, source, sample):
        # Two adjacent-ish energies that snap to the same point.
        with pytest.raises(ValueError):
            Experiment(source, sample, (8979.01, 8979.02), 10)

    def test_rejects_nrays_zero(self, source, sample):
        with pytest.raises(ValueError):
            Experiment(source, sample, None, 0)

    def test_rejects_noninteger_nrays(self, source, sample):
        with pytest.raises(ValueError):
            Experiment(source, sample, None, 2.5)

    def test_single_ray_guard(self, source, sample):
        with pytest.raises(ValueError):
            SingleRayExperiment(source, sample, None, 5)

    def test_full_sample_guard(self, source, sample):
        with pytest.raises(ValueError):
            FullSampleExperiment(source, sample, None, 1)


# ======================================================================
# Experiment -- single ray
# ======================================================================
class TestSingleRay:

    def test_angle_within_bounds(self, source, sample):
        e = Experiment(source, sample, None, 1, rng=3)
        theta = e.generate_ray()
        assert 0.0 <= theta <= source.alpha

    def test_thickness_positive(self, source, sample):
        e = Experiment(source, sample, None, 1, rng=3)
        assert e.generate_thickness() > 0.0

    def test_table_shape(self, source, sample):
        e = Experiment(source, sample, None, 1, rng=3)
        table = e.transmission_table()
        assert table.shape == (e.energies.size, 2)

    def test_beer_lambert_exact(self, source, sample):
        e = Experiment(source, sample, None, 1, rng=3)
        theta = e.generate_ray()
        thickness = e.generate_thickness()
        table = e.transmission_table()
        mu_col = sample.mu.iloc[:, 1].to_numpy(dtype=float)
        expected = np.exp(-mu_col * thickness / np.cos(theta))
        assert np.allclose(table[:, 1], expected)


# ======================================================================
# Experiment -- full sample
# ======================================================================
class TestFullSample:

    def test_run_shape(self, source, sample):
        e = Experiment(source, sample, None, 2000, rng=1)
        assert e.run().shape == (e.energies.size,)

    def test_jensen_inequality(self, source, sample, mu_values, thick_pdf):
        # <exp(-mu t)> >= exp(-mu <t>) everywhere (pinhole effect).
        e = Experiment(source, sample, None, 20000, rng=1)
        sim = e.run()
        ideal = np.exp(-mu_values * float(thick_pdf.mean()))
        assert np.all(sim >= ideal - 1e-12)

    def test_reproducible(self, source, sample):
        a = Experiment(source, sample, None, 2000, rng=99).run()
        b = Experiment(source, sample, None, 2000, rng=99).run()
        assert np.allclose(a, b)

    def test_transmission_table(self, source, sample):
        e = Experiment(source, sample, None, 1000, rng=2)
        table = e.transmission_table()
        assert table.shape == (e.energies.size, 2)
        assert np.array_equal(table[:, 0], e.energies)


# ======================================================================
# Experiment -- plotting
# ======================================================================
class TestPlotting:

    def test_plot_spectrum_returns_axes(self, source, sample):
        e = Experiment(source, sample, None, 1000, rng=1)
        fig, ax = e.plot_spectrum()
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_plot_absorbance_returns_axes(self, source, sample):
        e = Experiment(source, sample, None, 1000, rng=1)
        fig, ax = e.plot_absorbance()
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_compare_spectra_has_legend(self, source, sample):
        e = Experiment(source, sample, None, 1000, rng=1)
        fig, ax = e.compare_spectra()
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_compare_spectra_absorbance(self, source, sample):
        e = Experiment(source, sample, None, 1000, rng=1)
        fig, ax = e.compare_spectra(absorbance=True)
        assert fig is not None
        plt.close(fig)