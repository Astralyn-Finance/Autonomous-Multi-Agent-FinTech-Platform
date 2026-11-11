from pydantic import BaseModel

class PortfolioOptimizationResult(BaseModel):
    optimizer_name: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    objective_value: float
    sector_exposures: Dict[str, float]
    active_constraints: List[str]
    turnover: Optional[float] = None
    converged: bool
    iterations: int
    solver_status: str
    candidate_weights: Optional[Dict[str, float]] = None
    diagnostics: Dict[str, str] = {}