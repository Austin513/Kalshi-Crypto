"""Download Binance spot BTCUSDT 1-minute klines (monthly bulk archives).

Source: https://data.binance.vision  (docs: https://github.com/binance/binance-public-data)
URL pattern: data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip (+ .CHECKSUM)

Files are cached in data/raw/binance and verified against Binance's SHA-256 checksum.
Re-running skips files that already exist and verify.

    python -m btc_tails.download_binance
"""
import hashlib
import sys
import urllib.request

from . import MONTHS, PAIR, RAW_BINANCE

BASE = "https://data.binance.vision/data/spot/monthly/klines/{pair}/1m/{pair}-1m-{month}.zip"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def download(months=MONTHS, pair=PAIR, out=RAW_BINANCE):
    out.mkdir(parents=True, exist_ok=True)
    for month in months:
        url = BASE.format(pair=pair, month=month)
        zpath = out / url.rsplit("/", 1)[1]
        cpath = zpath.with_name(zpath.name + ".CHECKSUM")
        if not cpath.exists():
            _fetch(url + ".CHECKSUM", cpath)
        expected = cpath.read_text().split()[0]
        if zpath.exists() and _sha256(zpath) == expected:
            print(f"cached  {zpath.name}")
            continue
        print(f"fetch   {url}")
        _fetch(url, zpath)
        got = _sha256(zpath)
        if got != expected:
            zpath.unlink()
            sys.exit(f"checksum mismatch for {zpath.name}: {got} != {expected}")
        print(f"ok      {zpath.name} sha256 verified")


if __name__ == "__main__":
    download()
