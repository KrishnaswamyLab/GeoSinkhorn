import pytest
import numpy as np
from geosink.heat_kernel import HeatFilter, laplacian_from_data
import scipy
from scipy.sparse import rand

def gt_heat_kernel_data(
    data,
    t,
    sigma,
    alpha=20,
):
    L = laplacian_from_data(data, sigma, alpha=alpha)
    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    # compute the heat kernel
    heat_kernel = eigvecs @ np.diag(np.exp(-t * eigvals)) @ eigvecs.T
    heat_kernel = (heat_kernel + heat_kernel.T) / 2
    heat_kernel[heat_kernel < 0] = 0.0
    return heat_kernel


def test_laplacian():
    data = np.random.normal(0, 1, (100, 5))
    sigma = 1.0
    L = laplacian_from_data(data, sigma)
    assert np.allclose(L, L.T)
    # compute the largest eigenvalue
    eigvals, _ = np.linalg.eigh(L)
    max_eigval = eigvals.max()
    assert max_eigval <= 2.0


@pytest.mark.parametrize("t", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("method", ["cheb", "exact"])
def test_heat_kernel_gaussian(t, order, method):
    data = np.random.normal(0, 1, (100, 5))
    lap = laplacian_from_data(data, sigma=1.0)
    heat_op = HeatFilter(lap=lap, tau=t, order=order, method=method)
    heat_kernel = heat_op.heat_kernel

    # test if symmetric
    assert np.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert np.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_data(data, t=t, sigma=1.0)
    assert np.allclose(heat_kernel, gt_heat_kernel, atol=1e-1, rtol=1e-1)

@pytest.mark.limit_memory("800 MB")
def test_sparse_heat_kernel():
    t = 5.0
    order = 10
    adj = rand(10_000, 10_000, density=0.0001, format="coo", dtype=np.float32)
    lap = scipy.sparse.csgraph.laplacian(adj, symmetrized=False)

    heat_op = HeatFilter(lap=lap, tau=t, order=order, method="cheb")
    signal = np.random.uniform(0, 1, (10_000,1))
    signal = signal / signal.sum()
    diffused_signal = heat_op(signal)

    np.testing.assert_array_equal(np.isfinite(diffused_signal), True)

if __name__ == "__main__":
    pytest.main([__file__])