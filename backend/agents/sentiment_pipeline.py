"""Free-news retrieval + Gemini sentiment adapter for portfolio orchestration."""
from __future__ import annotations
from typing import Iterable, Optional

from data.source_router import get_fundamentals
from sentiment_agent import GeminiSentimentAgent

def _extract_market_caps(tickers: Iterable[str]) -> dict[str, float]:
    """Retrieve market capitalization from source_router fundamentals policy.

    source_router.get_fundamentals() returns {ticker: MarketDataPoint}.
    A ticker with no retrievable market cap is absent.
    """
    ticker_list = list(tickers)
    fundamentals = get_fundamentals(ticker_list)
    market_caps: dict[str, float] = {}

    for t in ticker_list:
        data_point = fundamentals.get(t)
        if data_point is not None:
            cap = getattr(data_point, "market_cap", None)
            if cap is None and isinstance(data_point, dict):
                cap = data_point.get("market_cap")
            if cap is not None and float(cap) > 0:
                market_caps[t] = float(cap)

    return market_caps

def build_sentiment_views(tickers: Iterable[str], baseline_returns,
                         company_names: dict | None = None, limit: int = 10,
                         agent: GeminiSentimentAgent | None = None,
                         current_market_caps: Optional[dict[str, float]] = None,
                         retrieved_market_caps: Optional[dict[str, float]] = None,
                         max_tilt: float = 0.05, 
                        auto_confirm: bool = False) -> tuple[dict, dict, list, dict]:
    """Return (views, confidences, diagnostics) ready for orchestrator.run().

    Uses market caps retrieved directly via source_router.get_fundamentals()
    and routes them alongside LLM-scraped news to generate views, confidence,
    and constraints.
    """
    agent = agent or GeminiSentimentAgent()
    ticker_list = list(tickers)

    # 1. Fetch current market caps via source_router
    current_market_caps = _extract_market_caps(ticker_list)
    prior_market_caps = retrieved_market_caps or {}

    raw_results = {}
    views, confidences, diagnostics = {}, {}, {}
    constraints = []

    # 2. Analyze sentiment & market cap shifts via Gemini
    for ticker in ticker_list:
        name = (company_names or {}).get(ticker, ticker)
        cur_cap = current_market_caps.get(ticker)
        prev_cap = prior_market_caps.get(ticker)

        try:
            result = agent.generate_views(
                ticker=ticker,
                company_name=name,
                current_market_cap=cur_cap,
                retrieved_market_cap=prev_cap,
            )
        except Exception as exc:
            diagnostics[ticker] = {"status": "error", "error": str(exc), "articles": 0}
            continue

        baseline = float(baseline_returns.loc[ticker] if hasattr(baseline_returns, "loc")
                         else baseline_returns[ticker])
        view, confidence = agent.finalize_confidences(result, baseline, max_tilt=max_tilt)
        views[ticker] = view
        raw_results[ticker] = result

        if result.constraint_note:
            constraints.append({
                "ticker": ticker,
                "note": result.constraint_note,
                "max_weight": result.max_weight,
            })

        # 3. Interactive Confidence Calibration
    if views:
        print("\n" + "=" * 75)
        print("          LLM GENERATED VIEWS & CONFIDENCE CALIBRATION")
        print("=" * 75)
        for ticker, res in raw_results.items():
            cap_str = f"{current_market_caps.get(ticker, 0.0):,.0f}" if ticker in current_market_caps else "N/A"
            print(f"Ticker: {ticker:<12} | Mkt Cap: {cap_str:>14} | View: {views[ticker]*100:>6.2f}% | Conf: {res.confidence:>4.2f}")
            print(f"  Rationale: {res.rationale}")
            if res.constraint_note:
                print(f"  Constraint Note: {res.constraint_note} (Max Weight: {res.max_weight})")
        print("=" * 75)

        accept_all = True if auto_confirm else (
            input("\nAccept all recommended confidence scores? (y/n) [y]: ").strip().lower() in ["", "y", "yes"]
        )

        for ticker, res in raw_results.items():
            if accept_all:
                confidences[ticker] = res.confidence
            else:
                user_val = input(f"Confidence for {ticker} [default {res.confidence:.2f}]: ").strip()
                try:
                    confidences[ticker] = float(user_val) if user_val else res.confidence
                except ValueError:
                    confidences[ticker] = res.confidence

            diagnostics[ticker] = {
                "status": "ok",
                "score": res.score,
                "confidence": confidences[ticker],
                "articles": res.articles_used,
                "rationale": res.rationale,
                "constraint_note": res.constraint_note,
                "market_cap": current_market_caps.get(ticker),
            }

    return views, confidences, constraints, diagnostics
