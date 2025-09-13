import math
from statistics import NormalDist

def z_from_quantile(q: float) -> float:
    if q <= 0.0:
        return -10.0
    if q >= 1.0:
        return 10.0
    return NormalDist().inv_cdf(q)

def blocks_uv(B: int, n: int, block_var: float) -> float:
    if n <= 0:
        return float("inf")
    return (float(B) ** 2) * (1.0 - (float(n) / float(B))) * (float(block_var) / float(n))

def L_mu_lower(mu_hat_p: float, var_mu_hat_p: float, delta1: float) -> float:
    z = z_from_quantile(1.0 - float(delta1))
    return float(mu_hat_p) - z * math.sqrt(max(0.0, float(var_mu_hat_p)))

