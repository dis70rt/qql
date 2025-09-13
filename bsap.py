# bsap.py
import math
from statistics import NormalDist
from typing import Optional


def z_from_quantile(q: float) -> float:
    """
    Return z such that Phi(z) = q (standard normal inverse CDF).
    Uses statistics.NormalDist for robust inverse CDF.
    """
    if q <= 0.0:
        return -10.0
    if q >= 1.0:
        return 10.0
    return NormalDist().inv_cdf(q)


def blocks_uv(B: int, n: int, block_var: float) -> float:
    """
    SRSWOR variance formula for the HT sum/mean estimator at block granularity.

    U_V[Theta] approx = B^2 * (1 - n/B) * (sigma_b^2 / n)
    where sigma_b^2 is the population variance of block totals.
    """
    if n <= 0:
        return float("inf")
    return (float(B) ** 2) * (1.0 - (float(n) / float(B))) * (float(block_var) / float(n))



def L_mu_lower(mu_hat_p: float, var_mu_hat_p: float, delta1: float) -> float:
    """
    Compute L_mu = mu_hat_p - z_{1-delta1} * sqrt(var_mu_hat_p)
    Uses normal approx (CLT).
    """
    z = z_from_quantile(1.0 - float(delta1))
    return float(mu_hat_p) - z * math.sqrt(max(0.0, float(var_mu_hat_p)))

