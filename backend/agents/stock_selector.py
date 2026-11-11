"""
Stock Selector

"""

import numpy as np
import pandas as pd


def select_top_n(posterior_mu: pd.Series, cov: pd.DataFrame, n: int) -> list:
    if n >= len(posterior_mu):
        return list(posterior_mu.index)

    vol = np.sqrt(np.diag(cov.loc[posterior_mu.index, posterior_mu.index].values))
    vol = np.where(vol > 1e-9, vol, 1e-9)
    score = pd.Series(posterior_mu.values / vol, index=posterior_mu.index)
    return list(score.sort_values(ascending=False).index[:n])
