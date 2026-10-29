""" Market data adapter for the strategy engine. Uses Yahoo Finance via yfinance as a research/demo source. It is not guaranteed to match Pocket Option's quote feed, and should not be described as exchange- verified real-time pricing. For live trading decisions, compare timestamps and quotes against the broker feed and use completed candles. """

from __future__ import annotations
import logging
import re
from typing import Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    FOREX_SYMBOLS = {
        "EUR/USD": "EURUSD=X",
        "EURUSD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USDJPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "AUDUSD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "USDCAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "USDCHF": "CHF=X",
        "NZD/USD": "NZDUSD=X",
        "NZDUSD": "NZDUSD=X",
        "EUR/GBP": "EURGBP=X",
        "EURGBP": "EURGBP=X",
        "EUR/JPY": "EURJPY=X",
        "EURJPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "GBPJPY": "GBPJPY=X",
    }

    INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"}
    PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y"}

    @classmethod
    def normalise_symbol(cls, symbol: str) -> str:
        raw = str(symbol or "").strip().upper().replace(" ", "")
        if raw in cls.FOREX_SYMBOLS:
            return cls.FOREX_SYMBOLS[raw]
        if re.fullmatch(r"[A-Z]{6}", raw):
            return cls.FOREX_SYMBOLS.get(raw, raw + "=X")
        return raw or "EURUSD=X"

    @classmethod
    def get_market_data(
        cls,
        symbol="EUR/USD",
        period="5d",
        interval="1m",
        drop_latest_candle=True,
    ) -> pd.DataFrame:
        """Fetch OHLCV history. By default drops latest potentially incomplete candle."""
        if interval not in cls.INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}. Choose from {sorted(cls.INTERVALS)}")
        if period not in cls.PERIODS:
            raise ValueError(f"Unsupported period {period!r}. Choose from {sorted(cls.PERIODS)}")

        ticker_symbol = cls.normalise_symbol(symbol)
        try:
            hist = yf.Ticker(ticker_symbol).history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
            if hist is None or hist.empty:
                logger.warning("No market data returned for %s", ticker_symbol)
                return pd.DataFrame()

            hist = hist.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            wanted = [c for c in ("open", "high", "low", "close", "volume") if c in hist.columns]
            hist = hist[wanted].copy()
            hist = hist.dropna(subset=["close"])
            if drop_latest_candle and len(hist) > 1:
                hist = hist.iloc[:-1].copy()
            hist.attrs["source"] = "Yahoo Finance via yfinance (not broker-feed verified)"
            hist.attrs["symbol"] = ticker_symbol
            hist.attrs["interval"] = interval
            return hist
        except Exception as exc:
            logger.exception("Market data fetch failed for %s: %s", ticker_symbol, exc)
            return pd.DataFrame()

    @classmethod
    def get_multi_platform_confirmation(cls, symbol="EUR/USD", interval="1h"):
        """ Backwards-compatible direction check. The returned percentage is the absolute change between the last two available closes, not a probability. """
        hist = cls.get_market_data(
            symbol=symbol,
            period="5d",
            interval=interval,
            drop_latest_candle=True,
        )
        if hist.empty or len(hist) < 2:
            return "NEUTRAL", 0.0

        previous = float(hist["close"].iloc[-2])
        latest = float(hist["close"].iloc[-1])
        if previous == 0:
            return "NEUTRAL", 0.0
        diff = latest - previous
        trend = "BULLISH" if diff > 0 else "BEARISH" if diff < 0 else "NEUTRAL"
        change_pct = abs(diff) / abs(previous) * 100.0
        return trend, round(change_pct, 5)

    @classmethod
    def get_latest_price(cls, symbol="EUR/USD", interval="1m") -> Optional[float]:
        hist = cls.get_market_data(
            symbol=symbol,
            period="1d",
            interval=interval,
            drop_latest_candle=False,
        )
        if hist.empty:
            return None
        value = float(hist["close"].iloc[-1])
        return value if pd.notna(value) else None
