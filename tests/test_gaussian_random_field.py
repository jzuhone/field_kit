import numpy as np

from kspace import GaussianRandomField, PowerLawBetaModel


def _make_field(ndim, seed=10):
    power_spec = PowerLawBetaModel(
        l_min=10.0, l_max=200.0, alpha=-11.0 / 3.0, ndim=ndim
    )
    power_spec.renormalize(f_rms=1.0)
    le = np.zeros(ndim)
    re = np.full(ndim, 100.0)
    ddims = [32] * ndim
    return GaussianRandomField(le, re, ddims, power_spec, seed=seed), ddims


def test_scalar_field_shape_and_reality_3d():
    grf, ddims = _make_field(3)
    field = grf.generate_scalar_field_realization()
    assert field.shape == tuple(ddims)
    assert np.isrealobj(field)


def test_vector_field_shape_3d():
    grf, ddims = _make_field(3)
    field = grf.generate_vector_field_realization()
    assert field.shape == (3, *ddims)


def test_reproducible_with_seed():
    grf1, _ = _make_field(2, seed=42)
    grf2, _ = _make_field(2, seed=42)
    field1 = grf1.generate_scalar_field_realization()
    field2 = grf2.generate_scalar_field_realization()
    np.testing.assert_array_equal(field1, field2)
