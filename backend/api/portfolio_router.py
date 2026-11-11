from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

from backend.schemas.constraint_schema import PortfolioConstraints
from backend.schemas.view_schema import StructuredInvestmentView
from backend.agents.coordinator import MultiAgentPortfolioCoordinator

router = APIRouter(prefix="/api/v2/portfolio", tags=["Portfolio Optimization"])

class OptimizationRequest(BaseModel):
    constraints: PortfolioConstraints
    override_views: Optional[List[StructuredInvestmentView]] = None

@router.post("/optimize")
async def optimize_portfolio(request: OptimizationRequest) -> Dict[str, Any]:
    coordinator = MultiAgentPortfolioCoordinator()
    
    # In production, assets, prices, and covariances are pulled from the market data service
    # Representative Nifty 50 core asset universe
    assets = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS"]
    market_caps = {
        "RELIANCE.NS": 1900000.0, "TCS.NS": 1400000.0, "HDFCBANK.NS": 1200000.0,
        "INFY.NS": 650000.0, "ICICIBANK.NS": 750000.0, "ITC.NS": 550000.0, "LT.NS": 480000.0
    }
    sector_map = {
        "RELIANCE.NS": "Energy", "TCS.NS": "Technology", "HDFCBANK.NS": "Financials",
        "INFY.NS": "Technology", "ICICIBANK.NS": "Financials", "ITC.NS": "Consumer Goods", "LT.NS": "Industrials"
    }
    
    # Synthetic clean positive-definite covariance matrix
    np.random.seed(42)
    A = np.random.randn(len(assets), len(assets)) * 0.05
    Sigma = A @ A.T + np.eye(len(assets)) * 0.02

    views = request.override_views or []

    try:
        result = coordinator.execute_pipeline(
            assets=assets,
            market_caps=market_caps,
            covariance_matrix=Sigma,
            sector_map=sector_map,
            constraints=request.constraints,
            research_views=views
        )
        if result["status"] == "REJECTED_INFEASIBLE_CONSTRAINTS":
            raise HTTPException(status_code=422, detail=result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))