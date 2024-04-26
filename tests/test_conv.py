from geosink.sinkhorn import GeoSinkhorn 
from geosink.heat_kernel import laplacian_from_data
import numpy as np
import pytest
import scipy
from scipy.sparse import rand


@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("num_iter", [100, 200, 300])
@pytest.mark.parametrize("tau", [0.1, 1.0, 10.0])
def test_geo_sinkhorn(order, num_iter, tau):
    data = np.random.normal(0, 1, (1000, 5))
    lap = laplacian_from_data(data, sigma=1.0)
    conv_sinkhorn = GeoSinkhorn(tau=tau, order=order, method="cheb", lap=lap)
    m_0 = np.random.rand(1000,)
    m_0[:500] = 0
    m_0 = m_0 / np.sum(m_0)
    m_1 = np.random.rand(1000,)
    m_1[500:] = 0
    m_1 = m_1 / np.sum(m_1)
    kl, v, w = conv_sinkhorn(m_0, m_1, return_vw=True, max_iter=num_iter) 
    assert np.isfinite(kl)
    assert kl >= 0
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(w))
 
def test_sparse_geo_sinkhorn():
    adj = rand(10_000, 10_000, density=0.0001, format="coo", dtype=np.float32)
    lap = scipy.sparse.csgraph.laplacian(adj, symmetrized=False)
    conv_sinkhorn = GeoSinkhorn(tau=1.0, order=10, method="cheb", lap=lap)
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

def test_gaussian():
    # random normal data
    data0 = np.random.normal(0, 1, (100, 5))
    data1 = np.random.normal(5, 1, (100, 5))
    data2 = np.random.normal(10, 1, (100, 5))
    data = np.concatenate([data0, data1, data2], axis=0)
    lap = laplacian_from_data(data, sigma=1.0)
    conv_sinkhorn = GeoSinkhorn(tau=1.0, order=10, method="cheb", lap=lap)
    m_0 = np.zeros(300,)
    m_0[:100] = 1
    m_0 = m_0 / np.sum(m_0)
    m_1 = np.zeros(300,)
    m_1[100:200] = 1
    m_1 = m_1 / np.sum(m_1)
    m_2 = np.zeros(300,)
    m_2[200:] = 1
    kl = conv_sinkhorn(m_0, m_1, max_iter=500)
    kl2 = conv_sinkhorn(m_0, m_2, max_iter=500)
    assert kl2 > kl

if __name__ == "__main__":
    pytest.main([__file__])