from geosink.conv_sinkhorn import ConvSinkhorn 
from geosink.heat_kernel import laplacian_from_data
import numpy as np
import pytest
import scipy
from scipy.sparse import rand


@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("num_iter", [100, 200, 300])
@pytest.mark.parametrize("tau", [0.1, 1.0, 10.0])
def test_conv_sinkhorn(order, num_iter, tau):
    data = np.random.normal(0, 1, (1000, 5))
    lap = laplacian_from_data(data, sigma=1.0)
    conv_sinkhorn = ConvSinkhorn(tau=tau, order=order, method="cheb", lap=lap)
    m_0 = np.random.rand(1000,)
    m_0[:5 00] = 0
    m_0 = m_0 / np.sum(m_0)
    m_1 = np.random.rand(1000,)
    m_1[500:] = 0
    m_1 = m_1 / np.sum(m_1)
    kl, v, w = conv_sinkhorn(m_0, m_1, return_vw=True, max_iter=num_iter) 
    assert np.isfinite(kl)
    assert kl >= 0
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))
 
def test_sparse_conv_sinkhorn():
    adj = rand(10_000, 10_000, density=0.0001, format="coo", dtype=np.float32)
    lap = scipy.sparse.csgraph.laplacian(adj, symmetrized=False)
    conv_sinkhorn = ConvSinkhorn(tau=1.0, order=10, method="cheb", lap=lap)
    m_0 = np.random.rand(10_000,)
    m_0[:5_000] = 0
    m_0 = m_0 / np.sum(m_0)
    m_1 = np.random.rand(10_000,)
    m_1[5_000:] = 0
    m_1 = m_1 / np.sum(m_1)
    kl, v, w = conv_sinkhorn(m_0, m_1, return_vw=True, max_iter=100)
    assert np.isfinite(kl)
    assert kl >= 0
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))
