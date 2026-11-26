# CPU-only image that downloads the data, builds the database, runs the checks, and executes the notebook.
FROM python:3.14-slim

WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 BTC_TAILS_DATA=/work/data MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY btc_tails ./btc_tails
COPY tests ./tests
COPY notebooks ./notebooks

# data/ is a volume so downloads are cached between runs:  docker run -v "$PWD/data:/work/data" ...
VOLUME ["/work/data"]

CMD ["sh", "-c", "python -m pytest -q tests \
 && python -m btc_tails.download_binance \
 && python -m btc_tails.build_db \
 && python -m btc_tails.kalshi 2026-10-01 2026-11-01 \
 && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/btc_tails.ipynb"]
