"""
Return / Covariance Estimation

"""

from pypfopt import expected_returns, risk_models  # type: ignore[reportMissingImports]


def estimate_mu_cov(prices):
    """Turn a price-history DataFrame into annualized expected returns (mu)
    and a shrunk covariance matrix (cov).

    Ledoit-Wolf shrinkage (“A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices”) is used for the covariance rather than the raw
    sample covariance: with a few years of daily data against ~50 assets,
    the raw sample covariance is poorly conditioned (near-singular
    directions from noise, not real structure).
    """
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    return mu, cov
