#!/usr/bin/env python
# python -m finrl.preview_mini

import sys
import numpy as np
import pandas as pd

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INTRA_TRAIN_START, INTRA_TRAIN_END,
    INTRA_VAL_START, INTRA_VAL_END,
    INTRA_TEST_START, INTRA_TEST_END,
)

sys.path.append("/home/krucz/Desktop/magistrat/Magistrat")
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30", "boll_ub", "boll_lb"]

periods = [
    ("Intraday train", INTRA_TRAIN_START, INTRA_TRAIN_END, "1m"),
    ("Intraday val",   INTRA_VAL_START,   INTRA_VAL_END,   "1m"),
    ("Intraday test",  INTRA_TEST_START,  INTRA_TEST_END,  "1m"),
]

processor = YahooFinanceProcessor()

def mahal_turbulence_intraday(df: pd.DataFrame, window: int = 389,
                              min_assets: int = 3, min_obs: int = 20) -> pd.Series:
    price = df.pivot(index="timestamp", columns="tic", values="close").sort_index()
    ret = price.pct_change()
    if ret.empty:
        return pd.Series(dtype=float, index=ret.index)
    tser = pd.Series(np.nan, index=ret.index, dtype=float)
    dates = pd.to_datetime(ret.index).date
    start = 0
    n = len(ret)
    while start < n:
        d = dates[start]
        end = start
        while end < n and dates[end] == d:
            end += 1
        block = ret.iloc[start:end]
        if len(block) <= window:
            start = end
            continue
        for j in range(window, len(block)):
            hist = block.iloc[j - window:j]
            cur = block.iloc[j]
            valid = cur.dropna().index
            if len(valid) < min_assets:
                continue
            hist = hist[valid].dropna(how="all")
            if hist.shape[0] < min_obs:
                continue
            cov = hist.cov(min_periods=max(5, min_obs // 2))
            mu = hist.mean()
            diag = np.nan_to_num(np.diag(cov.values))
            keep = cov.columns[diag > 0]
            if len(keep) < min_assets:
                continue
            cov = cov.loc[keep, keep]
            mu = mu[keep]
            curv = cur[keep]
            try:
                inv = np.linalg.pinv(cov.values)
                diff = (curv.values - mu.values).reshape(1, -1)
                tser.iloc[start + j] = (diff @ inv @ diff.T).item()
            except Exception:
                continue
        start = end
    return tser

def preview_ticker(df, label, ticker="AAPL", n_rows=1600):
    cols = ["open","high","low","close","volume"]
    n = len(df)
    any_nan = int(df.isna().any(axis=1).sum())
    base_nan = df[cols].isna().sum().reindex(cols).fillna(0).astype(int)
    ts_min = pd.to_datetime(df["timestamp"]).min() if "timestamp" in df else None
    ts_max = pd.to_datetime(df["timestamp"]).max() if "timestamp" in df else None
    print(f"\n=== {label} ===")
    print(f"Range: {ts_min} → {ts_max} | Rows: {n} | Rows with any NaN: {any_nan}")
    print("NaN by base cols:", base_nan.to_dict())
    try:
        by_tic = df.groupby("tic").size().sort_values(ascending=False)
        print("Rows per tic (top):", by_tic.head(10).to_dict())
    except Exception:
        pass
    df_sorted = df.sort_values(["timestamp","tic"]).reset_index(drop=True)
    df_ticker = df_sorted[df_sorted["tic"] == ticker].reset_index(drop=True)
    head_df = df_ticker.head(n_rows)
    with pd.option_context("display.max_rows", n_rows, "display.max_columns", None, "display.width", 200):
        print(f"\nFIRST {n_rows} ROWS for {ticker}:")
        print(head_df.to_string(index=False))

    t = mahal_turbulence_intraday(df_sorted, window=389, min_assets=3, min_obs=20)
    s = t.dropna()
    if s.empty:
        print("\nTurbulence: N=0")
    else:
        out = {
            "N": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }
        print("\nTurbulence summary:", out)
        per_day = s.groupby(pd.to_datetime(s.index).normalize()).size()
        with pd.option_context("display.max_rows", None):
            print("\nPer-day turbulence counts (non-NaN):")
            print(per_day.to_string())

if __name__ == "__main__":
    for label, start_date, end_date, interval in periods:
        try:
            df = processor.download_data(ticker_list=DOW_30_TICKER, start_date=start_date, end_date=end_date, time_interval=interval)
            df = processor.clean_data(df)
            df = processor.add_technical_indicator(df, tech_indicators)
            preview_ticker(df, f"{label} [{interval}] {start_date}..{end_date}", ticker="AAPL", n_rows=800)
        except Exception as e:
            print(f"\n=== {label} [{interval}] {start_date}..{end_date} ===")
            print("ERROR:", e)
