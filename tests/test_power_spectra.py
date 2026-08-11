import numpy as np

from kspace import PowerLaw, PowerLawBetaModel


def test_power_law_renormalize():
    ps = PowerLaw(alpha=-11.0 / 3.0, k0=1.0)
    f_rms = 5.0
    ps.renormalize(f_rms, kmin=0.01, kmax=10.0)
    integral = ps.integrate_E(0.01, 10.0)
    assert np.isclose(integral, f_rms**2)


def test_power_law_beta_model_positive():
    ps = PowerLawBetaModel(l_min=10.0, l_max=200.0, alpha=-11.0 / 3.0)
    k = np.geomspace(1e-3, 10.0, 50)
    assert np.all(ps(k) > 0)
