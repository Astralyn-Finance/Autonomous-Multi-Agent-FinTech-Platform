import sys
from pathlib import Path

# Dynamically resolve project root:
# current file -> backend/test/test_registry.py
# .parent      -> backend/test
# .parent.parent.parent -> FinTech (project root containing 'backend')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# test_registry.py
from backend.optimization.optimizer_selector import OptimizerRecommender, UserPortfolioRequirements
from backend.optimization.feasibility_checker import FeasibilityChecker

# 1. Test MIQP routing for exact cardinality
req_miqp = UserPortfolioRequirements(exact_k_stocks=3, max_weight=0.50)
solver, reason = OptimizerRecommender.select_optimizer(req_miqp)
print(f"Test 1 (Exact K=3): {solver} -> {reason}")
assert solver == "MIQP"

# 2. Test MVO_QP routing for standard bounds
req_mvo = UserPortfolioRequirements(min_weight=0.02, max_weight=0.25)
solver, reason = OptimizerRecommender.select_optimizer(req_mvo)
print(f"Test 2 (Continuous): {solver} -> {reason}")
assert solver == "MVO_QP"

# 3. Test feasibility validator on impossible constraints
try:
    bad_req = UserPortfolioRequirements(exact_k_stocks=5, max_weight=0.10) # 5 * 0.10 = 0.50 < 1.0
    FeasibilityChecker.validate(bad_req, universe_tickers=["RELIANCE.NSE", "TCS.NSE", "INFY.NSE"])
    print("Feasibility test failed to catch invalid budget!")
except ValueError as e:
    print(f"Test 3 (Feasibility Catch): Successfully caught invalid constraints -> {e}")