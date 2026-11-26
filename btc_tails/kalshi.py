"""Download public Kalshi data for the KXBTC15M series ("BTC price up in next 15 mins?").

No API key is used; these are public market-data endpoints:
  live:       GET /trade-api/v2/markets/{ticker}
              GET /trade-api/v2/series/KXBTC15M/markets/{ticker}/candlesticks
  archived:   GET /trade-api/v2/historical/markets/{ticker}
              GET /trade-api/v2/historical/markets/{ticker}/candlesticks
Markets that settled before GET /historical/cutoff -> market_settled_ts are only on the archived endpoints,
so each request tries live first and falls back to historical.

Each market is cached as data/raw/kalshi/{ticker}.json with its metadata and the 1-minute candlesticks
covering [open, open + CANDLE_MINUTES]. Re-running only fetches missing tickers.

    python -m btc_tails.kalshi 2026-10-01 2026-11-01
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import RAW_KALSHI

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
CANDLE_MINUTES = 3
ET = ZoneInfo("America/New_York")


def ticker_for_window(t_open):
    """Ticker for the market whose 15-minute window opens at UTC epoch `t_open`.

    Kalshi names the event by its CLOSE time in US Eastern time, e.g. a window 16:00-16:15 UTC on
    2026-07-15 (EDT) -> KXBTC15M-26JUL151215-15.
    """
    close_et = datetime.fromtimestamp(t_open + 900, timezone.utc).astimezone(ET)
    return f"{SERIES}-{close_et.strftime('%y%b%d%H%M').upper()}-{close_et.strftime('%M')}"


def _get(path, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "student-research-script"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(1.5 * (i + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"giving up on {path}")


def fetch_market(ticker):
    for prefix, candle_path in (("/markets/", f"/series/{SERIES}/markets/{ticker}/candlesticks"),
                                ("/historical/markets/", f"/historical/markets/{ticker}/candlesticks")):
        meta = _get(prefix + ticker)
        if meta is None:
            continue
        m = meta["market"]
        t0 = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        candles = _get(f"{candle_path}?start_ts={t0}&end_ts={t0 + 60 * CANDLE_MINUTES}&period_interval=1")
        return {"source": prefix.strip("/"), "market": m,
                "candlesticks": (candles or {}).get("candlesticks", []),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return {"source": "missing", "market": None, "candlesticks": []}


def download_series_meta(out=RAW_KALSHI):
    """Cache series terms, the series fee-change history, and the archive cutoff (all public)."""
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series": _get(f"/series/{SERIES}"),
        "fee_changes": _get(f"/series/fee_changes?series_ticker={SERIES}&show_historical=true"),
        "historical_cutoff": _get("/historical/cutoff"),
    }
    (out / "_series_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def download(start, end, out=RAW_KALSHI, pause=0.12):
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "_series_meta.json").exists():
        download_series_meta(out)
    t = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    t_end = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    n_new = 0
    while t < t_end:
        tk = ticker_for_window(t)
        path = out / f"{tk}.json"
        if not path.exists():
            rec = fetch_market(tk)
            rec["window_open_utc"] = t
            path.write_text(json.dumps(rec))
            n_new += 1
            if n_new % 200 == 0:
                print(f"{n_new} fetched (at {tk})", flush=True)
            time.sleep(pause)
        t += 900
    print(f"done: {n_new} new files in {out}")


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
