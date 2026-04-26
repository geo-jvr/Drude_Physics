"""Unit tests for the Drude / Drude-Sommerfeld simulation engine.

Run from the terminal::

    python test_drude.py            # human-readable summary
    python -m unittest test_drude   # unittest discovery

These are *unit-style* tests --- short, deterministic, and they catch the
specific ways the engine can be wrong (sign errors in the Boris pusher,
miscalibrated collision rate, tail-truncation in the FD sampler, etc.).

The full statistical / visual tests live in ``Drude_Computation.ipynb``;
graphs go there because they are the headline deliverable.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# A self-contained copy of the engine, identical to the notebook's.
# Duplicated so that the tests stand alone with no notebook execution.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Callable, Optional


def maxwell_boltzmann(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``(n, 3)`` MB-distributed velocities, unit variance per axis."""
    return rng.standard_normal((n, 3))


@dataclass
class DrudeStep:
    """First- or second-order operator-split single-step Drude propagator."""

    order: int = 2
    resample: Callable[[int, np.random.Generator], np.ndarray] = field(
        default=maxwell_boltzmann
    )

    def _collide(self, v, dt, rng):
        p_coll = 1.0 - np.exp(-dt)
        mask = rng.random(v.shape[0]) < p_coll
        if mask.any():
            v[mask] = self.resample(int(mask.sum()), rng)

    @staticmethod
    def _drift(v, dt, e_field, b_field):
        accel = -np.asarray(e_field, dtype=float)
        if b_field is None or not np.any(b_field):
            v += accel * dt
            return
        v += 0.5 * accel * dt
        t_vec = -0.5 * dt * np.asarray(b_field, dtype=float)
        s_vec = 2.0 * t_vec / (1.0 + t_vec @ t_vec)
        v_prime = v + np.cross(v, t_vec)
        v += np.cross(v_prime, s_vec)
        v += 0.5 * accel * dt

    def __call__(self, v, dt, e_field, b_field=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        e_field = np.asarray(e_field, dtype=float)
        b_field = None if b_field is None else np.asarray(b_field, dtype=float)
        if self.order == 1:
            self._collide(v, dt, rng)
            self._drift(v, dt, e_field, b_field)
        elif self.order == 2:
            self._drift(v, 0.5 * dt, e_field, b_field)
            self._collide(v, dt, rng)
            self._drift(v, 0.5 * dt, e_field, b_field)
        else:
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        return v


def fermi_dirac_sampler(theta: float):
    mu_tilde = (1.0 / theta) * (1.0 - (np.pi ** 2 / 12.0) * theta ** 2)
    vF = math.sqrt(2.0 / theta)
    u_grid = np.linspace(0.0, max(vF * 1.1, 6.0), 4096)

    def speed_pdf(u):
        arg = 0.5 * u ** 2 / theta - mu_tilde
        return u ** 2 / (np.exp(np.clip(arg, -700, 700)) + 1.0)

    pdf = speed_pdf(u_grid)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    def sample(n, rng):
        u_samples = np.interp(rng.random(n), cdf, u_grid)
        z = rng.uniform(-1.0, 1.0, n)
        phi = rng.uniform(0.0, 2 * np.pi, n)
        sin_t = np.sqrt(1.0 - z ** 2)
        return u_samples[:, None] * np.column_stack(
            [sin_t * np.cos(phi), sin_t * np.sin(phi), z]
        )

    return sample


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SEED = 20260417


class TestDriftSubstep(unittest.TestCase):
    """The pure-drift (no collisions) case must be ballistic Newton."""

    def test_no_field_is_identity(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        v = rng.standard_normal((50, 3))
        v0 = v.copy()
        # With zero E and zero B, the only motion is collisions; suppress them
        # by using a no-op resample so that we test the drift sub-step alone.
        prop_noop = DrudeStep(order=2, resample=lambda n, r: np.zeros((n, 3)))
        # Disable collisions entirely by setting dt huge AFTER zero-field check.
        prop._drift(v, 1.0, np.zeros(3), None)
        np.testing.assert_allclose(v, v0)

    def test_constant_E_is_uniform_acceleration(self):
        """One ballistic step with no collisions: dv = -E dt for an electron."""
        v = np.zeros((1, 3))
        DrudeStep._drift(v, dt=0.1, e_field=np.array([0.5, 0.0, 0.0]),
                         b_field=None)
        np.testing.assert_allclose(v, [[-0.05, 0.0, 0.0]], atol=1e-12)

    def test_boris_preserves_kinetic_energy(self):
        """The Boris rotation is exactly energy-conserving for any dt and B."""
        rng = np.random.default_rng(SEED)
        v = rng.standard_normal((1000, 3))
        ke0 = 0.5 * (v ** 2).sum(axis=1)
        for dt in (0.01, 0.1, 0.5, 1.0):
            v_test = v.copy()
            DrudeStep._drift(v_test, dt, e_field=np.zeros(3),
                             b_field=np.array([0.0, 0.0, 0.7]))
            ke = 0.5 * (v_test ** 2).sum(axis=1)
            # Boris is energy-exact even at huge dt (modulo rounding).
            np.testing.assert_allclose(ke, ke0, rtol=1e-12, atol=1e-12,
                                       err_msg=f"dt={dt}")

    def test_boris_rotates_in_correct_sense(self):
        """v in +x with B in +z rotates toward +y for an electron.

        The Lorentz force on the electron is F = -e (v x B). With v = +x and
        B = +z we have v x B = -y, so F = +ey, hence v_y becomes positive.
        """
        v = np.array([[1.0, 0.0, 0.0]])
        DrudeStep._drift(v, dt=0.01, e_field=np.zeros(3),
                         b_field=np.array([0.0, 0.0, 1.0]))
        self.assertGreater(v[0, 0], 0.99)
        self.assertGreater(v[0, 1], 0.0)


class TestCollisionSubstep(unittest.TestCase):
    """Collision rate, equilibrium variance, and post-collision distribution."""

    def test_collision_probability_matches_poisson(self):
        """Empirical fraction collided in one step ~ 1 - exp(-dt)."""
        rng = np.random.default_rng(SEED)
        n = 200_000
        v = np.zeros((n, 3))           # tag pre-collision velocity
        prop = DrudeStep(order=1)
        prop._collide(v, dt=0.3, rng=rng)
        # Velocities that changed are precisely the ones that collided.
        collided = (v != 0.0).any(axis=1)
        empirical = collided.mean()
        expected = 1.0 - math.exp(-0.3)
        self.assertAlmostEqual(empirical, expected, places=2)

    def test_equilibrium_variance_unity(self):
        """Long zero-field run leaves <v_alpha^2> within MC error of 1."""
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        v = maxwell_boltzmann(20_000, rng)
        for _ in range(800):
            prop(v, dt=0.05, e_field=np.zeros(3), rng=rng)
        for ax in range(3):
            self.assertAlmostEqual(v[:, ax].var(), 1.0, delta=0.04,
                                   msg=f"axis {ax}")
            self.assertAlmostEqual(v[:, ax].mean(), 0.0, delta=0.03,
                                   msg=f"axis {ax}")


class TestMaxwellBoltzmannSampler(unittest.TestCase):
    def test_first_two_moments(self):
        rng = np.random.default_rng(SEED)
        v = maxwell_boltzmann(200_000, rng)
        np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=0.01)
        np.testing.assert_allclose(v.var(axis=0), 1.0, rtol=0.02)

    def test_three_dimensional_speed_mean(self):
        """<|v|> = 2 sqrt(2/pi) for a 3D Gaussian with unit variance."""
        rng = np.random.default_rng(SEED)
        v = maxwell_boltzmann(200_000, rng)
        speeds = np.linalg.norm(v, axis=1)
        self.assertAlmostEqual(speeds.mean(), 2.0 * math.sqrt(2.0 / math.pi),
                               delta=0.01)


class TestFermiDiracSampler(unittest.TestCase):
    def test_normalisation_and_isotropy(self):
        rng = np.random.default_rng(SEED)
        sample = fermi_dirac_sampler(theta=0.01)
        v = sample(80_000, rng)
        # Isotropy: each Cartesian component has zero mean.
        np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=0.05)
        # Variance per axis equals (1/3)<v^2>.
        np.testing.assert_allclose(v.var(axis=0)[0], v.var(axis=0)[1],
                                   rtol=0.05)
        np.testing.assert_allclose(v.var(axis=0)[0], v.var(axis=0)[2],
                                   rtol=0.05)

    def test_sommerfeld_mean_energy(self):
        """<eps>/eps_F -> 3/5 in the deeply degenerate limit.

        The sampler works in units where eps_F = 1 (and kT = theta), so
        <(1/2) v^2> directly equals <eps>/eps_F.
        """
        rng = np.random.default_rng(SEED)
        sample = fermi_dirac_sampler(theta=0.001)
        v = sample(80_000, rng)
        eps_per_eF = 0.5 * np.mean(np.sum(v ** 2, axis=1))
        self.assertAlmostEqual(eps_per_eF, 0.6, delta=0.01)

    def test_sommerfeld_specific_heat_scaling(self):
        """c_v / k_B = (pi^2 / 2) theta in the degenerate limit.

        Strategy: <eps>/eps_F = 3/5 + (pi^2/4) theta^2 + O(theta^4), so
        differentiating with respect to T (= T_F * theta) gives
        c_v / k_B = (pi^2 / 2) theta. We sample <eps>(theta) at a handful of
        moderately degenerate temperatures (where the correction comfortably
        exceeds the MC noise floor of ~6e-4 at N=200k) and fit the quadratic
        coefficient. At theta < 0.05 the signal is buried in noise and the
        gradient operator is meaningless --- this is why we deliberately stay
        in the regime theta in [0.05, 0.2].
        """
        rng = np.random.default_rng(SEED)
        thetas = np.array([0.05, 0.10, 0.15, 0.20])
        eps = np.empty_like(thetas)
        for i, theta in enumerate(thetas):
            v = fermi_dirac_sampler(theta)(200_000, rng)
            eps[i] = 0.5 * np.mean(np.sum(v ** 2, axis=1))
        # Fit eps - 3/5 = a * theta^2; theory predicts a = pi^2 / 4.
        coef, *_ = np.linalg.lstsq(thetas[:, None] ** 2,
                                   eps - 0.6, rcond=None)
        a_fit = float(coef[0])
        a_th = math.pi ** 2 / 4
        # 12% tolerance: this captures both MC noise and higher-order
        # corrections (the next term is ~ -7 pi^4 / 240 theta^4).
        self.assertAlmostEqual(a_fit / a_th, 1.0, delta=0.12)


class TestDCConductivity(unittest.TestCase):
    """Strang integrator should give sigma/sigma_0 = 1 within MC error."""

    def test_dc_drift_velocity(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        v = maxwell_boltzmann(20_000, rng)
        E0, dt, n_steps = 0.1, 0.05, 4000
        e = np.array([E0, 0.0, 0.0])
        hist = np.empty(n_steps)
        for s in range(n_steps):
            prop(v, dt, e, rng=rng)
            hist[s] = v[:, 0].mean()
        ss = hist[n_steps // 2:].mean()
        self.assertAlmostEqual(ss, -E0, delta=5e-3)


class TestACConductivity(unittest.TestCase):
    """Re sigma at omega tau = 1 must be 1/2 for the Drude Lorentzian."""

    def test_ac_real_part_at_corner(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        omega, E0 = 1.0, 0.2
        period = 2 * math.pi / omega
        steps_per_period = 80
        n_periods = 30
        dt = period / steps_per_period
        n_steps = n_periods * steps_per_period
        t_grid = np.arange(n_steps) * dt
        v = maxwell_boltzmann(8000, rng)
        vx = np.empty(n_steps)
        for s, t in enumerate(t_grid):
            prop(v, dt, np.array([E0 * math.cos(omega * t), 0, 0]), rng=rng)
            vx[s] = v[:, 0].mean()
        start = n_steps // 3
        design = np.column_stack([np.cos(omega * t_grid[start:]),
                                  np.sin(omega * t_grid[start:])])
        A, B = np.linalg.lstsq(design, vx[start:], rcond=None)[0]
        sigma = -(A + 1j * B) / E0
        # theory: sigma = 1 / (1 - i) = 0.5 + 0.5 i
        self.assertAlmostEqual(sigma.real, 0.5, delta=0.04)
        self.assertAlmostEqual(sigma.imag, 0.5, delta=0.04)


class TestGreenKuboIdentity(unittest.TestCase):
    """The integral of the equilibrium VACF gives the DC conductivity."""

    def test_integrated_vacf_equals_one(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        n_e, n_steps, dt = 8000, 1024, 0.05
        v = maxwell_boltzmann(n_e, rng)
        v_hist = np.empty((n_steps, n_e))
        for s in range(n_steps):
            prop(v, dt, e_field=np.zeros(3), rng=rng)
            v_hist[s] = v[:, 0]
        # FFT-based unbiased autocorrelation.
        n_pad = 2 * n_steps
        spec = np.fft.rfft(v_hist, n=n_pad, axis=0)
        acf = np.fft.irfft(spec * spec.conj(), n=n_pad,
                           axis=0)[:n_steps].real
        norm = np.arange(n_steps, 0, -1)[:, None]
        C = (acf / norm).mean(axis=1)
        # Truncate at 15 tau: signal is e^-15 ~ 3e-7 there, so any further
        # contribution is pure noise and only inflates the variance.
        cutoff = int(15.0 / dt)
        sigma_dc = np.trapezoid(C[:cutoff], dx=dt)
        self.assertAlmostEqual(sigma_dc, 1.0, delta=0.05)


class TestHallCoefficient(unittest.TestCase):
    """In natural units R_H = -1 for any B; check at one B value."""

    def test_hall_coefficient_sign_and_magnitude(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=2)
        n_e, n_steps, dt = 12_000, 3000, 0.05
        Bz, E0 = 1.0, 0.1
        v = maxwell_boltzmann(n_e, rng)
        e = np.array([E0, 0.0, 0.0])
        b = np.array([0.0, 0.0, Bz])
        vx = np.empty(n_steps)
        vy = np.empty(n_steps)
        for s in range(n_steps):
            prop(v, dt, e, b, rng=rng)
            vx[s] = v[:, 0].mean()
            vy[s] = v[:, 1].mean()
        start = n_steps // 3
        sxx = -vx[start:].mean() / E0
        syx = -vy[start:].mean() / E0
        rho_yx = -syx / (sxx ** 2 + syx ** 2)
        R_H = rho_yx / Bz
        self.assertAlmostEqual(R_H, -1.0, delta=0.03)


class TestIntegratorConvergence(unittest.TestCase):
    """Lie-Trotter bias should be ~ E0*dt/2 to leading order."""

    def test_lie_trotter_steady_state_matches_closed_form(self):
        rng = np.random.default_rng(SEED)
        prop = DrudeStep(order=1)
        E0, dt, n_steps = 0.1, 0.5, 1200
        v = maxwell_boltzmann(20_000, rng)
        # Equilibrate, then time-average over the steady-state portion to
        # beat down the per-snapshot ensemble noise of size 1/sqrt(N_e).
        burn = n_steps // 3
        history = np.empty(n_steps - burn)
        for s in range(n_steps):
            prop(v, dt, np.array([E0, 0, 0]), rng=rng)
            if s >= burn:
                history[s - burn] = v[:, 0].mean()
        # Closed-form Lie-Trotter steady state: v_inf = -E0 dt / (1 - e^-dt)
        expected = -E0 * dt / (1.0 - math.exp(-dt))
        measured = history.mean()
        self.assertAlmostEqual(measured, expected, delta=2e-3)


# ---------------------------------------------------------------------------
# Pretty terminal runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
