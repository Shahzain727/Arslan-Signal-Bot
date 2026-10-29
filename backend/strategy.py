from __future__ import annotations

import numpy as np
import pandas as pd


class QuantitativeTradingStrategy:
    EXPIRY_CANDIDATES = (1, 2, 3, 4, 5, 10, 15, 30, 60)

    @staticmethod
    def _clean_prices(price_history):
        prices = np.asarray(price_history, dtype=float).reshape(-1)
        return prices[np.isfinite(prices)]

    @staticmethod
    def calculate_sma(price_history, period):
        prices = QuantitativeTradingStrategy._clean_prices(price_history)
        if period <= 0 or len(prices) < period:
            return None
        return float(np.mean(prices[-period:]))

    @staticmethod
    def calculate_hurst_exponent(price_series, max_lag=20):
        """Rough R/S-like scaling proxy; noisy on short financial series."""
        prices = QuantitativeTradingStrategy._clean_prices(price_series)
        if len(prices) < max_lag * 2:
            return 0.5

        lags, tau = [], []
        for lag in range(2, min(max_lag, len(prices) // 2)):
            differences = prices[lag:] - prices[:-lag]
            sd = float(np.std(differences))
            if np.isfinite(sd) and sd > 0:
                lags.append(lag)
                tau.append(sd)

        if len(lags) < 3:
            return 0.5
        try:
            slope = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
            if not np.isfinite(slope):
                return 0.5
            return float(np.clip(slope, 0.0, 1.0))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return 0.5

    @staticmethod
    def calculate_kama(price_series, period=10, fast=2, slow=30):
        """Return the latest point of a recursively calculated KAMA series."""
        prices = QuantitativeTradingStrategy._clean_prices(price_series)
        if len(prices) == 0:
            return None
        if len(prices) < period + 1:
            return float(prices[-1])

        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        kama = np.full(len(prices), np.nan, dtype=float)
        kama[period] = prices[period]

        for i in range(period + 1, len(prices)):
            change = abs(prices[i] - prices[i - period])
            volatility = float(np.sum(np.abs(np.diff(prices[i - period:i + 1]))))
            er = change / volatility if volatility > 0 else 0.0
            smoothing = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + smoothing * (prices[i] - kama[i - 1])

        valid = kama[np.isfinite(kama)]
        return float(valid[-1]) if len(valid) else float(prices[-1])

    @staticmethod
    def _normalise_frame(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        frame = df.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        # yfinance sometimes returns a MultiIndex column layout.
        if isinstance(df.columns, pd.MultiIndex):
            frame = df.copy()
            frame.columns = [
                str(c[0] if isinstance(c, tuple) else c).strip().lower()
                for c in df.columns
            ]
        aliases = {"adj close": "adj_close"}
        frame = frame.rename(columns=aliases)
        if "close" not in frame.columns:
            return None
        for col in ("open", "high", "low", "close", "volume"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["close"])
        if len(frame) < 2:
            return None
        return frame

    @staticmethod
    def _atr(frame, period=14):
        if not all(c in frame.columns for c in ("high", "low", "close")):
            return None
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        value = tr.rolling(period).mean().iloc[-1]
        return float(value) if pd.notna(value) and value > 0 else None

    @classmethod
    def _range_rejection(cls, frame, lookback=20):
        """Return (direction, explanation) for a confirmed edge rejection."""
        if not all(c in frame.columns for c in ("open", "high", "low", "close")):
            return None, None
        if len(frame) < lookback + 2:
            return None, None

        # Exclude the current candle from calculating the prior range.
        prior = frame.iloc[-(lookback + 1):-1]
        candle = frame.iloc[-1]
        support = float(prior["low"].min())
        resistance = float(prior["high"].max())
        atr = cls._atr(frame.iloc[:-1], 14)
        if atr is None:
            atr = float((prior["high"] - prior["low"]).median())
        if not np.isfinite(atr) or atr <= 0 or resistance <= support:
            return None, None

        o, h, l, c = map(float, (candle["open"], candle["high"], candle["low"], candle["close"]))
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        tolerance = max(atr * 0.25, abs(c) * 1e-6)

        # A rejection must close back inside the pre-existing range.
        if l <= support + tolerance and c > support and lower_wick >= max(body, tolerance * 0.2):
            return "CALL (BUY)", f"Range support rejection (support {support:.5f})"
        if h >= resistance - tolerance and c < resistance and upper_wick >= max(body, tolerance * 0.2):
            return "PUT (SELL)", f"Range resistance rejection (resistance {resistance:.5f})"
        return None, None

    @classmethod
    def _false_breakout(cls, frame, lookback=20):
        if not all(c in frame.columns for c in ("open", "high", "low", "close")) or len(frame) < lookback + 2:
            return None, None
        prior = frame.iloc[-(lookback + 1):-1]
        candle = frame.iloc[-1]
        support = float(prior["low"].min())
        resistance = float(prior["high"].max())
        o, h, l, c = map(float, (candle["open"], candle["high"], candle["low"], candle["close"]))
        atr = cls._atr(frame.iloc[:-1], 14) or float((prior["high"] - prior["low"]).median())
        tolerance = max(atr * 0.1, abs(c) * 1e-6)

        if l < support - tolerance and c > support and c >= o:
            return "CALL (BUY)", f"Bear-trap / false breakdown rejected near {support:.5f}"
        if h > resistance + tolerance and c < resistance and c <= o:
            return "PUT (SELL)", f"Bull-trap / false breakout rejected near {resistance:.5f}"
        return None, None

    @classmethod
    def _trend_signal(cls, prices, hurst, kama):
        if len(prices) < 35 or kama is None:
            return None, None
        fast_now = cls.calculate_sma(prices, 5)
        slow_now = cls.calculate_sma(prices, 30)
        fast_prev = cls.calculate_sma(prices[:-1], 5)
        slow_prev = cls.calculate_sma(prices[:-1], 30)
        current = float(prices[-1])
        bullish_cross = fast_prev is not None and slow_prev is not None and fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev is not None and slow_prev is not None and fast_prev >= slow_prev and fast_now < slow_now

        # Require directional alignment; Hurst alone is not confirmation.
        if hurst > 0.55 and fast_now > slow_now and current > kama:
            return "CALL (BUY)", "Trend continuation: SMA(5)>SMA(30), price>KAMA"
        if hurst > 0.55 and fast_now < slow_now and current < kama:
            return "PUT (SELL)", "Trend continuation: SMA(5)<SMA(30), price<KAMA"
        if bullish_cross and current > kama:
            return "CALL (BUY)", "Bullish SMA crossover confirmed above KAMA"
        if bearish_cross and current < kama:
            return "PUT (SELL)", "Bearish SMA crossover confirmed below KAMA"
        return None, None

    @classmethod
    def _select_expiry(cls, frame, direction):
        """ Heuristic expiry selector. It scores available horizons using directional persistence and volatility. This is not a trained probability model. """
        close = frame["close"].astype(float).to_numpy()
        if len(close) < 65:
            return None, "Need at least 65 candles to compare expiry horizons"

        atr = cls._atr(frame, 14)
        if atr is None:
            return None, "ATR unavailable; expiry not selected"

        sign = 1.0 if direction == "CALL (BUY)" else -1.0
        candidates = [h for h in cls.EXPIRY_CANDIDATES if h < len(close) - 5]
        scores = []
        for horizon in candidates:
            # Compare recent directional movement with recent noise.
            move = close[-1] - close[-(horizon + 1)]
            recent = np.diff(close[-(horizon + 1):])
            noise = float(np.std(recent)) * np.sqrt(max(horizon, 1))
            directional_alignment = sign * move / max(noise, atr * 0.15, 1e-12)
            # Penalise horizons whose latest movement opposes the proposed signal.
            if sign * move <= 0:
                directional_alignment -= 1.0
            # Avoid choosing very long expiries from a short, weak move.
            score = directional_alignment / np.sqrt(horizon)
            scores.append((score, horizon))

        if not scores:
            return None, "No eligible expiry horizon"
        scores.sort(reverse=True)
        best_score, best_horizon = scores[0]
        if not np.isfinite(best_score) or best_score <= 0:
            return None, "No expiry horizon aligns with the signal"
        return int(best_horizon), "Heuristic horizon ranking; must be validated by backtesting"

    @classmethod
    def evaluate_signal( cls, price_history=None, volume_history=None, ask_bid_delta=None, pair_name=None, df=None, **kwargs, ):
        """ Backwards-compatible return: (signal, reason). signal is CALL (BUY), PUT (SELL), or HOLD. If df contains OHLC, use it for rectangle/rejection checks. """
        frame = cls._normalise_frame(df) if df is not None else None

        if price_history is None and frame is not None:
            price_history = frame["close"].to_numpy()
        if price_history is None:
            return "HOLD", "No price history provided."

        prices = cls._clean_prices(price_history)
        if len(prices) < 65:
            return "HOLD", "Insufficient history: need at least 65 candles."

        if frame is not None:
            # Align price series with the cleaned OHLC frame.
            prices = frame["close"].to_numpy(dtype=float)
            if len(prices) < 65:
                return "HOLD", "Insufficient OHLC history: need at least 65 candles."

        current = float(prices[-1])
        hurst = cls.calculate_hurst_exponent(prices)
        kama = cls.calculate_kama(prices)
        atr = cls._atr(frame) if frame is not None else None

        # Regime / noise filter. Hurst is treated as a weak filter, not proof.
        if frame is not None and atr is not None:
            recent_ranges = (frame["high"] - frame["low"]).tail(10).astype(float)
            if len(recent_ranges) >= 8 and float(recent_ranges.median()) < atr * 0.25:
                return "HOLD", f"Low-volatility chop filter (Hurst {hurst:.2f}); no trade."

        candidates = []
        if frame is not None:
            for detector in (cls._false_breakout, cls._range_rejection):
                sig, why = detector(frame)
                if sig:
                    candidates.append((sig, why, 3))

        trend_sig, trend_reason = cls._trend_signal(prices, hurst, kama)
        if trend_sig:
            candidates.append((trend_sig, trend_reason, 2))

        if not candidates:
            return "HOLD", f"No validated setup; Hurst {hurst:.2f}. Wait for confirmation."

        # Prefer price-action rejection over trend signal, but reject conflicts.
        directions = {item[0] for item in candidates}
        if len(directions) > 1:
            return "HOLD", f"Conflicting strategy signals; Hurst {hurst:.2f}. No trade."

        candidates.sort(key=lambda x: x[2], reverse=True)
        signal, setup_reason, _ = candidates[0]
        expiry, expiry_note = cls._select_expiry(frame, signal) if frame is not None else (None, "OHLC candles required for expiry selection")
        if expiry is None:
            return "HOLD", f"{setup_reason}. Expiry not validated: {expiry_note}."

        pair_text = f"{pair_name}: " if pair_name else ""
        reason = (
            f"{pair_text}{setup_reason}; Hurst={hurst:.2f}; KAMA={kama:.5f}; "
            f"Suggested expiry={expiry} minute(s). {expiry_note}. "
            "Research/demo only; no accuracy guarantee."
        )
        return signal, reason
