import numpy as np

from geosink.heat_kernel import HeatFilter


class GeoSinkhorn:
    def __init__(self, tau, order, method, graph=None, lap=None):
        assert (graph is not None) or (
            lap is not None
        ), "Either graph or lap must be provided."
        self.tau = tau
        kwargs_filter = {"lap": lap} if graph is None else {"graph": graph}
        self.heat_filter = HeatFilter(
            tau=tau, order=order, method=method, **kwargs_filter
        )

    def __call__(
        self,
        m_0,
        m_1,
        stopThr=1e-4,
        max_iter=1e3,
        verbose=False,
        return_vw=False,
        stopping_crit=False,
        stopping_fn=lambda x: False,
    ):
        eps = 1e-8
        assert m_0.shape == m_1.shape, "m_0 and m_1 must have the same shape"
        N = m_0.shape[0]
        v = np.ones(N)
        w = np.ones(N)
        a = np.ones(N) / N
        for i in range(1, int(max_iter) + 1):
            v_prev = v
            w_prev = w
            v = m_0 / (self.heat_filter(a * w) + eps)
            w = m_1 / (self.heat_filter(a * v) + eps)
            if (
                np.any(np.isnan(v))
                or np.any(np.isnan(w))
                or np.any(np.isinf(v))
                or np.any(np.isinf(w))
            ):
                v = v_prev
                w = w_prev
                break
            if i % 100 == 0:
                if verbose:
                    print(i, np.sum(np.abs(v - v_prev)))
                if np.sum(np.abs(v - v_prev)) < stopThr:
                    if verbose:
                        print("converged at iteration %d" % i)
                    break
            if stopping_crit:
                kl = np.sum(
                    4
                    * self.tau
                    * a
                    * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps))
                )
                if stopping_fn(kl):
                    print("converged at iteration %d" % i)
                    break
        kl = np.sum(
            4 * self.tau * a * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps))
        )
        if return_vw:
            return kl, v, w
        return kl
