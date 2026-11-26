Local cache (ignored by Git). Created by the download scripts:

- `raw/binance/` – monthly BTCUSDT 1m zip archives + `.CHECKSUM` files from data.binance.vision
- `raw/kalshi/`  – one JSON file per KXBTC15M market (metadata + 1-minute candlesticks)
- `btc.sqlite`   – SQLite research database built by `python -m btc_tails.build_db`

Override the location with the environment variable `BTC_TAILS_DATA`.
