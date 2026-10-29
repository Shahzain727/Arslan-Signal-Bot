""" Telegram signal bot for research/demo use. Set TELEGRAM_BOT_TOKEN in Render Environment Variables. Yahoo Finance data is not guaranteed to match Pocket Option's feed. This bot generates research signals only and does not place trades. """
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone

from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from market_dayallta import MarketAnalyzer
from strategy import QuantitativeTradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app_flask = Flask(__name__)


@app_flask.route("/")
def home():
    return "Pocket Option Forex Signal Bot is running. Research/demo signals only."


def run_flask():
    port = int(os.environ.get("PORT", "10000"))
    app_flask.run(host="0.0.0.0", port=port, use_reloader=False)


# Use Yahoo Finance-compatible pair names through MarketAnalyzer.
FOREX_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
    "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD", "EUR/CAD", "GBP/CHF", "AUD/CAD",
    "CAD/JPY", "CHF/JPY", "GBP/CAD", "GBP/AUD",
]


def get_recommended_expiry(reason: str):
    """Extract the strategy's suggested expiry without changing its old API."""
    match = re.search(r"Suggested expiry=(\d+)\s+minute", reason or "", flags=re.I)
    return int(match.group(1)) if match else None


def fetch_pair_candles(pair_name: str):
    """ Get 1-minute OHLC candles. The strategy needs OHLC, not just close prices, for price-action patterns and its heuristic expiry selector. """
    try:
        frame = MarketAnalyzer.get_market_data(
            symbol=pair_name,
            period="1d",
            interval="1m",
            drop_latest_candle=True,  # avoid acting on an incomplete candle
        )
        if frame is None or frame.empty or len(frame) < 65:
            logger.warning("Insufficient 1m OHLC candles for %s", pair_name)
            return None
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(frame.columns)):
            logger.warning("OHLC columns missing for %s", pair_name)
            return None
        return frame
    except Exception:
        logger.exception("Failed to fetch candles for %s", pair_name)
        return None


async def handle_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    status_message = await update.message.reply_text(
        "⚡ Scanning available 1-minute candles and checking setups..."
    )

    try:
        report_lines = [
            "📊 POCKET OPTION FOREX SIGNALS",
            f"🕒 Scan time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            "ℹ️ Data: Yahoo Finance; broker-feed match is not guaranteed.",
            "",
        ]
        signals_found = 0
        pairs_with_data = 0

        for pair_name in FOREX_PAIRS:
            frame = fetch_pair_candles(pair_name)
            if frame is None:
                continue
            pairs_with_data += 1

            signal, reason = QuantitativeTradingStrategy.evaluate_signal(
                df=frame,
                price_history=frame["close"].to_numpy(),
                pair_name=pair_name,
            )

            live_entry_price = float(frame["close"].iloc[-1])
            expiry = get_recommended_expiry(reason)

            if signal not in ("CALL (BUY)", "PUT (SELL)") or expiry is None:
                logger.info("%s skipped: %s | %s", pair_name, signal, reason)
                continue

            # Yahoo quotes can be delayed or differ from the broker quote.
            report_lines.extend([
                f"💱 {pair_name}",
                f"💵 Latest completed candle close: {live_entry_price:.5f}",
                f"🎯 Signal: {signal}",
                f"⏳ Suggested expiry: {expiry} minute(s)",
                f"📝 Reason: {reason}",
                "",
            ])
            signals_found += 1

        if signals_found == 0:
            if pairs_with_data == 0:
                message = (
                    "⚠️ No usable market data returned. Yahoo Finance may be "
                    "unavailable, rate-limited, or missing recent 1-minute candles."
                )
            else:
                message = (
                    "⚠️ No validated setup with an expiry recommendation found. "
                    "This is a valid NO TRADE result; try again after a completed candle."
                )
            await status_message.edit_text(message)
            return

        report_lines.append(
            "⚠️ Research/demo output only. Suggested expiry is heuristic, not a "
            "validated probability or guarantee. Confirm price and expiry on your broker."
        )
        response_text = "\n".join(report_lines)

        # Telegram message length is limited; send in chunks if needed.
        await status_message.edit_text(response_text[:3900])
        if len(response_text) > 3900:
            await update.message.reply_text(response_text[3900:7800])

    except Exception as exc:
        logger.exception("Error in /signal handler")
        await status_message.edit_text(
            f"⚠️ Signal scan failed: {type(exc).__name__}. Check Render logs."
        )


async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is not None:
        await update.message.reply_text(
            "👋 Welcome!\n"
            "Send /signal to scan forex pairs.\n\n"
            "Signals are for research/demo use only. No accuracy or profit is guaranteed."
        )


def main():
    # Render web service health endpoint.
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN. Add it under Render > Environment."
        )

    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", handle_start_command))
    application.add_handler(CommandHandler("signal", handle_signal_command))

    logger.info("Starting Telegram signal bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
