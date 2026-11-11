"""Free-first financial news retrieval using public RSS feeds.

This module intentionally consumes RSS/Atom feeds rather than scraping
publisher pages. It returns source metadata and links so downstream agents can
trace every claim to retrieved material.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import html
import re
import time
from typing import Iterable, Optional
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


@dataclass(frozen=True)
class NewsArticle:
    article_id: str
    ticker: str
    title: str
    summary: str
    url: str
    source: str
    published_at: Optional[str] = None

    def to_prompt_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at,
            "url": self.url,
        }


class NewsRetriever:
    """Retrieve news from free RSS endpoints with bounded timeouts.

    Google News RSS is the broad fallback; Yahoo Finance RSS is also queried
    for Yahoo-style tickers. Feed failures are recorded and do not fabricate
    articles or prevent another configured feed from being tried.
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"

    def __init__(self, timeout: float = 10.0, session=None, user_agent: str = "PortfolioResearchBot/1.0"):
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    @staticmethod
    def _clean_text(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value or "")
        return re.sub(r"\s+", " ", html.unescape(value)).strip()

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None

    def _fetch_feed(self, url: str, ticker: str, max_articles: int,
                    lookback_days: int) -> list[NewsArticle]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        articles: list[NewsArticle] = []

        # RSS 2.0 and Atom feeds use slightly different item structures.
        nodes = root.findall(".//item")
        if not nodes:
            nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for node in nodes:
            title = self._clean_text(node.findtext("title") or node.findtext("{http://www.w3.org/2005/Atom}title") or "")
            summary_raw = node.findtext("description") or node.findtext("{http://www.w3.org/2005/Atom}summary") or ""
            summary = self._clean_text(summary_raw)
            link = node.findtext("link") or ""
            if not link:
                atom_link = node.find("{http://www.w3.org/2005/Atom}link")
                if atom_link is not None:
                    link = atom_link.attrib.get("href", "")
            pub_raw = node.findtext("pubDate") or node.findtext("{http://www.w3.org/2005/Atom}published") or node.findtext("{http://www.w3.org/2005/Atom}updated")
            published = self._parse_date(pub_raw)
            if not title or not link:
                continue
            if published and published < cutoff:
                continue
            source_node = node.find("source")
            source = self._clean_text(source_node.text or "") if source_node is not None else "RSS feed"
            article_id = link
            articles.append(NewsArticle(
                article_id=article_id,
                ticker=ticker,
                title=title,
                summary=summary[:1600],
                url=link,
                source=source,
                published_at=published.isoformat() if published else None,
            ))
            if len(articles) >= max_articles:
                break
        return articles

    def retrieve(self, ticker: str, company_name: Optional[str] = None,
                 max_articles: int = 8, lookback_days: int = 14) -> list[NewsArticle]:
        if not ticker or not ticker.strip():
            raise ValueError("ticker is required")
        if max_articles < 1 or lookback_days < 1:
            raise ValueError("max_articles and lookback_days must be positive")

        ticker = ticker.strip().upper()
        search_terms = " ".join(x for x in (company_name, ticker, "stock market") if x)
        feeds = [self.GOOGLE_NEWS_RSS.format(query=quote_plus(search_terms))]
        # Yahoo Finance commonly recognizes the .NS suffix used by NSE tickers.
        if ticker.endswith(".NS"):
            feeds.append(self.YAHOO_RSS.format(ticker=quote_plus(ticker)))

        collected: list[NewsArticle] = []
        seen: set[str] = set()
        errors: list[str] = []
        per_feed = max(max_articles, 1)
        for feed_url in feeds:
            try:
                for article in self._fetch_feed(feed_url, ticker, per_feed, lookback_days):
                    key = article.url or article.title.casefold()
                    if key not in seen:
                        seen.add(key)
                        collected.append(article)
            except (requests.RequestException, ET.ParseError, ValueError) as exc:
                errors.append(f"{feed_url}: {exc}")
            if len(collected) >= max_articles:
                break

        # Keep diagnostics available to callers without turning a feed outage
        # into fabricated data or an unhandled failure.
        self.last_errors = errors
        return collected[:max_articles]

    def retrieve_many(self, tickers: Iterable[str], company_names: Optional[dict] = None,
                      max_articles_per_ticker: int = 8,
                      lookback_days: int = 14) -> dict[str, list[NewsArticle]]:
        names = company_names or {}
        return {
            ticker: self.retrieve(ticker, names.get(ticker), max_articles_per_ticker, lookback_days)
            for ticker in tickers
        }
