# test_agent.py
from backend.agents.sentiment_agent import GeminiSentimentAgent

agent = GeminiSentimentAgent()
result = agent.analyze(
    ticker="RELIANCE.NS",
    company_name="Reliance Industries",
    current_market_cap=2000000.0,
    retrieved_market_cap=1900000.0
)

print(f"Ticker: {result.ticker}")
print(f"Score: {result.score} (-1.0 to 1.0)")
print(f"Confidence: {result.confidence} (0.0 to 1.0)")
print(f"Rationale: {result.rationale}")
print(f"Constraint Note: {result.constraint_note}")
print(f"Max Weight Cap: {result.max_weight}")