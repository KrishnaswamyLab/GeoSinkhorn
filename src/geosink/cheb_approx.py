import typing as T

import numpy as np
from scipy.special import ive

try:
    import torch
except ImportError:
    torch = None


def compute_chebychev_coeff_all_torch(eigval, t, K):
    with torch.no_grad():
        eigval = eigval.detach().cpu()
        out = 2.0 * ive(torch.arange(0, K + 1), -t * eigval)
    return out


def compute_chebychev_coeff_all(eigval, t, K):
    return 2.0 * ive(
        np.arange(
            0,
            K + 1,
        ),
        -t * eigval,
    )


def expm_multiply(
    L: np.ndarray,
    X: np.ndarray,
    coeff: np.ndarray,
    eigval: np.ndarray,
):
    """Matrix exponential with Chebyshev polynomial approximation."""

    def body(carry, c):
        T0, T1, Y = carry
        T2 = (2.0 / eigval) * (L @ T1) - 2.0 * T1 - T0
        Y = Y + c * T2
        return (T1, T2, Y)

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = (1.0 / eigval) * (L @ X) - T0
    Y = Y + coeff[1] * T1

    initial_state = (T0, T1, Y)
    for c in coeff[2:]:
        initial_state = body(initial_state, c)

    _, _, Y = initial_state

    return Y
