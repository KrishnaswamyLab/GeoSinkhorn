from typing import Any, Callable, Optional

import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh

from geosink.cheb_approx import compute_chebychev_coeff_all, expm_multiply

try:
    import pygsp
except ImportError:
    pygsp = None

EPS_LOG = 1e-6
EPS_HEAT = 1e-4


def norm_sym_laplacian(A: np.array):
    deg = A.sum(axis=1)
    deg_sqrt_inv = np.diag(1.0 / np.sqrt(deg + EPS_LOG))
    return deg_sqrt_inv @ A @ deg_sqrt_inv


def laplacian_from_data(data: np.array, sigma: float, alpha: int = 20):
    affinity = np.exp(
        -((scipy.spatial.distance.cdist(data, data) / (2 * sigma)) ** alpha)
    )
    return norm_sym_laplacian(affinity)


class HeatFilter:
    """Wrapper for the approximation of the heat kernel."""

    _valid_methods = ["cheb_pygsp", "cheb", "lowrank", "exact"]

    def __init__(
        self,
        tau: float,  # Diffusion time
        order: int,  # Degree or numver of steps
        method: str,  # filter `" cheb_pygsp"`, `"cheb"`, `"lowrank"`, `"exact"`.
        graph: Optional[Any] = None,  # Graph object
        lap: Optional[np.array] = None,  # Laplacian matrix
    ) -> Callable:
        self.graph = graph
        self.tau = tau
        self.order = order
        self.method = method

        assert (graph is not None) or (
            lap is not None
        ), "Either graph or lap must be provided."
        self.lap = graph.L if lap is None else lap

        if method not in self._valid_methods:
            raise ValueError(
                "method must be one of {}".format(self._valid_methods)
            )

        if method == "cheb_pygsp":
            assert (
                graph is not None
            ), "graph must be provided for method cheb_pygsp"
            graph.estimate_lmax()
            self._filter = pygsp.filters.Heat(graph, tau)

        elif method == "cheb":
            self.phi = eigsh(self.lap, k=1, return_eigenvectors=False)[0] / 2
            self.coeff = compute_chebychev_coeff_all(
                self.phi, self.tau, self.order
            )

        elif method == "lowrank":
            self.lap.shape[0]
            eval, evec = eigsh(self.lap, k=order, which="SM")
            self._filter = evec @ np.diag(np.exp(-self.tau * eval)) @ evec.T

        elif method == "exact":
            eval, evec = scipy.linalg.eigh(self.lap)
            self._filter = evec @ np.diag(np.exp(-self.tau * eval)) @ evec.T

    @property
    def heat_kernel(self):
        heat_kernel = self(np.eye(self.lap.shape[0]))
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        heat_kernel[heat_kernel < 0] = 0.0
        return heat_kernel

    def __call__(self, b, safe_zeros=True):

        if self.method == "cheb_pygsp":
            diff = self._filter.filter(b, order=self.order)

        elif self.method == "cheb":
            diff = expm_multiply(self.lap, b, self.coeff, self.phi)

        elif self.method in ["lowrank", "exact"]:
            diff = self._filter @ b

        if safe_zeros:
            diff[diff < 0.0] = 0.0
        return diff
