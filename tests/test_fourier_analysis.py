import numpy as np

from field_kit import FourierAnalysis, GaussianRandomField, PowerLawBetaModel
from field_kit.fourier_analysis import FFTArray


def _spectral_curl(fa, a_real):
    a_fft = fa.fftn(a_real)
    k, _ = fa.generate_waves("continuum")
    if fa.ndim == 2:
        b_hat = 1j * np.stack([k[1] * a_fft, -k[0] * a_fft])
    else:
        b_hat = 1j * np.cross(k, a_fft, axis=0)
    return fa.ifftn(FFTArray(b_hat, delta=fa.delta))


def test_divergence_component_recovers_purely_longitudinal_field():
    # A field constructed as a gradient is 100% longitudinal, so projecting
    # out its longitudinal part should return it unchanged (regression test
    # for a kvec/fftn layout mismatch that used to corrupt this projection).
    width = np.array([100.0, 100.0])
    ddims = [64, 64]
    fa = FourierAnalysis(width, ddims)
    x = fa.delta[0] * (np.arange(ddims[0]) + 0.5)
    k0 = 2 * np.pi * 5 / width[0]
    fx = (k0 * np.cos(k0 * x))[:, None] * np.ones((1, ddims[1]))
    field = np.stack([fx, np.zeros_like(fx)])

    longitudinal = fa.divergence_component(field, diff_type="continuum")
    np.testing.assert_allclose(longitudinal, field, atol=1e-10)


def test_make_binned_powerspec_recovers_known_mode():
    width = np.array([100.0, 100.0])
    ddims = [64, 64]
    fa = FourierAnalysis(width, ddims)
    x = fa.delta[0] * (np.arange(ddims[0]) + 0.5)
    k0 = 2 * np.pi * 5 / width[0]
    field = np.cos(k0 * x)[:, None] * np.ones((1, ddims[1]))

    kbins, pk = fa.make_binned_powerspec(field, 40)
    kmid = np.sqrt(kbins[1:] * kbins[:-1])
    i_peak = np.ma.argmax(pk)
    assert abs(kmid[i_peak] - k0) < (kbins[1] - kbins[0])


def test_potential_of_field_inverts_spectral_curl():
    # Use a realistic, smooth (decaying) power spectrum rather than white
    # noise: white noise puts an unrealistically large fraction of its
    # power at the Nyquist frequency, which is the one mode this inversion
    # can't recover exactly (see docstring on potential_of_field).
    for ndim, ddims in [(2, [48, 48]), (3, [32, 32, 32])]:
        width = np.full(ndim, 200.0)
        ps = PowerLawBetaModel(l_min=10.0, l_max=200.0, alpha=-11.0 / 3.0, ndim=ndim)
        ps.renormalize(f_rms=1.0)
        grf = GaussianRandomField(np.zeros(ndim), width, ddims, ps, seed=7)
        fa = FourierAnalysis(width, ddims)

        if ndim == 2:
            a = grf.generate_scalar_field_realization()
        else:
            a = grf.generate_vector_field_realization(divergence_free=False)
        b = _spectral_curl(fa, a)
        a_recovered = fa.potential_of_field(b)
        b_recovered = _spectral_curl(fa, a_recovered)

        # A small residual is expected: naive ik-multiplication spectral
        # derivatives can't preserve realness at the Nyquist frequency on
        # an even-sized grid, so this isn't an exact round trip -- modes
        # at/near the Nyquist frequency can be off by a large relative
        # amount, so check the typical (median) error rather than the max.
        assert np.median(np.abs(b_recovered - b)) < 0.05 * b.std()


def test_average_symmetric_k_preserves_hermitian_real_signal():
    # For any real-valued input, f_hat[-k] == conj(f_hat[k]) (Hermitian
    # symmetry), so averaging the +k and -k modes must give exactly the
    # real part of f_hat at each +k mode -- this only holds if the fold
    # pairs indices using the same (unshifted) layout as fftn.
    width = np.array([100.0, 100.0])
    ddims = [8, 8]
    fa = FourierAnalysis(width, ddims)
    x = fa.delta[0] * (np.arange(ddims[0]) + 0.5)
    k0 = 2 * np.pi * 2 / width[0]
    field = np.cos(k0 * x)[:, None] * np.ones((1, ddims[1]))

    f_hat = fa.fftn(field)
    folded = f_hat.average_symmetric_k(axis=0)

    n_pos = ddims[0] - ddims[0] // 2
    assert folded.shape == (n_pos, ddims[1])
    np.testing.assert_allclose(folded, f_hat[:n_pos].real, atol=1e-8)


def test_solenoidal_component_complements_divergence_component():
    # solenoidal + compressional must reconstruct the original field, and
    # the result must agree regardless of whether the input is passed as
    # a real-space array or an already-transformed FFTArray (regression
    # test for a bug where FFTArray input was silently re-FFT'd/mismatched
    # in space).
    width = np.array([100.0, 100.0])
    ddims = [32, 32]
    fa = FourierAnalysis(width, ddims)
    field = np.random.default_rng(0).normal(size=(2, *ddims))

    vc = fa.divergence_component(field)
    vs = fa.solenoidal_component(field)
    np.testing.assert_allclose(vc + vs, field, atol=1e-10)

    field_hat = fa.fftn(field)
    vs_hat = fa.solenoidal_component(field_hat, return_fft=True)
    np.testing.assert_allclose(fa.ifftn(vs_hat), vs, atol=1e-10)


def test_periodic_gradient_removes_boundary_artifact():
    # curl_of_field/divergence_of_field default to non-periodic (one-sided
    # at the domain edge) finite differences, since data need not be
    # periodic in general. But the compressive/solenoidal projections are
    # inherently periodic (they're computed via FFT), so checking them
    # with the *default* real-space curl/div leaves a boundary artifact
    # from that mismatch, not a real residual. periodic=True removes it.
    width = np.array([100.0, 100.0, 100.0])
    ddims = [32, 32, 32]
    fa = FourierAnalysis(width, ddims)
    v = np.random.default_rng(0).normal(size=(3, *ddims))

    vc = fa.divergence_component(v)
    vs = fa.solenoidal_component(v)

    assert np.abs(fa.curl_of_field(vc)).mean() > 1e-8
    assert np.abs(fa.curl_of_field(vc, periodic=True)).max() < 1e-10

    assert np.abs(fa.divergence_of_field(vs)).mean() > 1e-8
    assert np.abs(fa.divergence_of_field(vs, periodic=True)).max() < 1e-10


def test_curl_of_field_2d_returns_scalar():
    width = np.array([100.0, 100.0])
    ddims = [32, 32]
    fa = FourierAnalysis(width, ddims)
    data_vec = np.random.default_rng(0).normal(size=(2, *ddims))
    curl = fa.curl_of_field(data_vec)
    assert curl.shape == tuple(ddims)
    assert np.isrealobj(curl)


def test_curl_of_field_3d_returns_vector():
    width = np.array([100.0, 100.0, 100.0])
    ddims = [16, 16, 16]
    fa = FourierAnalysis(width, ddims)
    data_vec = np.random.default_rng(0).normal(size=(3, *ddims))
    curl = fa.curl_of_field(data_vec)
    assert curl.shape == (3, *ddims)
