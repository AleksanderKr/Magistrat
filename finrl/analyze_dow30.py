#!/usr/bin/env python
"""
Examples
--------
python -m finrl.analyze_dow30 --period train --mode all
python -m finrl.analyze_dow30 --period test  --mode log_ret --draw true
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, levene
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt

from finrl.config import (
    TRAIN_START_DATE, TRAIN_END_DATE,
    VALIDATION_START_DATE, VALIDATION_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE,
    CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE,
    CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE,
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor

# ─────────────────────────────── settings ────────────────────────────────
DATA_SOURCE  = "yahoofinance"
TIME_INTERVAL = "1d"
TECHNICALS    = ["rsi_30", "boll_ub", "boll_lb"]

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ───────────────────── helpers: loading & slicing ────────────────────────
def load_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    dp = DataProcessor(DATA_SOURCE)
    d  = dp.download_data(tickers, start, end, TIME_INTERVAL)
    d  = dp.clean_data(d)
    d  = dp.add_technical_indicator(d, TECHNICALS)
    return d.reset_index(drop=True)

PERIOD_WINDOWS = {
    "train": dict(stable=(TRAIN_START_DATE, TRAIN_END_DATE),
                  crisis=(CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE)),
    "val":   dict(stable=(VALIDATION_START_DATE, VALIDATION_END_DATE),
                  crisis=(CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE)),
    "test":  dict(stable=(TEST_START_DATE, TEST_END_DATE),
                  crisis=(CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE)),
}

def get_period_data(period: str) -> Tuple[pd.DataFrame, Tuple[str, str, str, str]]:
    s_start, s_end = PERIOD_WINDOWS[period]["stable"]
    c_start, c_end = PERIOD_WINDOWS[period]["crisis"]
    overall_start  = min(s_start, c_start)
    overall_end    = max(s_end,   c_end)
    return load_data(DOW_30_TICKER, overall_start, overall_end), (s_start, s_end, c_start, c_end)

def split_sets(df: pd.DataFrame, bounds) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s_start, s_end, c_start, c_end = bounds
    stable  = df[(df["timestamp"] >= s_start) & (df["timestamp"] <= s_end)]
    crisis  = df[(df["timestamp"] >= c_start) & (df["timestamp"] <= c_end)]
    return stable, crisis

# ──────────────────────────── plotting util ──────────────────────────────
def kde_plot(series_s, series_c, title, filename):
    s_clean = series_s.dropna()
    c_clean = series_c.dropna()

    if s_clean.nunique() < 2 or c_clean.nunique() < 2:
        print(f"[warn] Skipping KDE for {filename}: one of the series is constant.")
        return

    xs = np.linspace(min(s_clean.min(), c_clean.min()),
                     max(s_clean.max(), c_clean.max()), 400)
    kde_s = gaussian_kde(s_clean)(xs)
    kde_c = gaussian_kde(c_clean)(xs)

    plt.figure(figsize=(6,3))
    plt.plot(xs, kde_s, label="stable",  lw=1.6)
    plt.plot(xs, kde_c, label="crisis",  lw=1.6, ls="--")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.show()


# ───────────────────────── analyses (each draw-aware) ────────────────────
def analyze_log_returns(df, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    stats = lambda s: s.agg(["mean", "std", "skew", "kurt"])
    out = pd.concat({"stable": stats(stable["log_ret"]), "crisis": stats(crisis["log_ret"])}, axis=1).T.round(5)
    print("\n=== Log-return summary ==="); print(out.to_string())
    stat, p = levene(stable["log_ret"].dropna(), crisis["log_ret"].dropna())
    print(f"\nLevene variance test: stat={stat:.3f}  p={p:.4e}")
    if draw:
        kde_plot(stable["log_ret"], crisis["log_ret"],
                 "Log-return KDE (stable vs crisis)",
                 "kde_logret.png")

def analyze_volatility(df, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    df["vol_30"]  = df.groupby("tic")["log_ret"].transform(lambda x: x.rolling(30).std())
    stable, crisis = split_sets(df, bounds)
    res = pd.DataFrame({"mean_vol":[stable["vol_30"].mean(), crisis["vol_30"].mean()],
                        "median_vol":[stable["vol_30"].median(), crisis["vol_30"].median()]},
                       index=["stable","crisis"]).round(5)
    print("\n=== Rolling 30-day volatility ==="); print(res.to_string())
    if draw:
        kde_plot(stable["vol_30"].dropna(), crisis["vol_30"].dropna(),
                 "30-day σ KDE", "kde_vol30.png")

def analyze_rsi_signals(df, bounds, draw=False):
    df = df.sort_values(["tic","timestamp"])
    df["rsi_30"] = Sdf.retype(df.copy())["rsi_30"]
    stable, crisis = split_sets(df, bounds)
    f = lambda s: pd.Series({"days>70": (s["rsi_30"]>70).sum(),
                             "days<30": (s["rsi_30"]<30).sum(),
                             "mean_rsi": s["rsi_30"].mean()})
    out = pd.concat({"stable":f(stable),"crisis":f(crisis)}, axis=1).T.round(2)
    print("\n=== RSI signal counts ==="); print(out.to_string())
    if draw:
        kde_plot(stable["rsi_30"], crisis["rsi_30"],
                 "RSI₃₀ KDE", "kde_rsi30.png")

def analyze_bollinger_behavior(df, bounds, draw=False):
    df = df.sort_values(["tic","timestamp"])
    ss = Sdf.retype(df.copy())
    df["boll_ub"], df["boll_lb"] = ss["boll_ub"], ss["boll_lb"]
    df["up"] = df["close"] > df["boll_ub"]; df["low"] = df["close"] < df["boll_lb"]
    stable, crisis = split_sets(df, bounds)
    g = lambda s: pd.Series({"breaks_upper": s["up"].sum(), "breaks_lower": s["low"].sum()})
    out = pd.concat({"stable":g(stable),"crisis":g(crisis)}, axis=1).T
    print("\n=== Bollinger breakouts ==="); print(out.to_string())
    # (wykres pomijam – binarne wyjście)

def analyze_volume(df, bounds, draw=False):
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    corr_s = stable["volume"].corr(stable["log_ret"]); corr_c = crisis["volume"].corr(crisis["log_ret"])
    out = pd.DataFrame({"mean_vol":[stable["volume"].mean(), crisis["volume"].mean()],
                        "median_vol":[stable["volume"].median(), crisis["volume"].median()],
                        "corr_vol_ret":[corr_s, corr_c]}, index=["stable","crisis"]).round(3)
    print("\n=== Volume stats ==="); print(out.to_string())
    if draw:
        kde_plot(np.log1p(stable["volume"]), np.log1p(crisis["volume"]),
                 "log(Volume) KDE", "kde_volume.png")

def analyze_correlation(df, bounds, draw=False):
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    def avg(s):
        p = s.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        return np.nan if p.empty else p.corr().values[np.triu_indices_from(p.corr(),1)].mean()
    out = pd.DataFrame({"avg_corr":[avg(stable), avg(crisis)]}, index=["stable","crisis"]).round(5)
    print("\n=== Avg pair-wise return correlation ==="); print(out.to_string())
    # (wykres jednowartościowy – pomijam)

def analyze_turbulence(df, bounds, draw=False):
    price = df.pivot(index="timestamp", columns="tic", values="close").sort_index()
    ret   = price.pct_change().dropna()
    w = 252; turb = pd.Series(0.0, index=ret.index, dtype=np.float32)
    for i in range(w, len(ret)):
        hist, cur = ret.iloc[i-w:i], ret.iloc[i]
        valid = cur.dropna().index; hist, cur = hist[valid], cur[valid]
        cov = hist.cov(); diff = cur - hist.mean()
        turb.iloc[i] = float(diff.values @ np.linalg.pinv(cov.values) @ diff.values.T)
    stable, crisis = split_sets(turb.to_frame("t").reset_index(), bounds)
    out = pd.DataFrame({"mean":[stable["t"].mean(), crisis["t"].mean()],
                        "median":[stable["t"].median(), crisis["t"].median()],
                        "p95":[stable["t"].quantile(.95), crisis["t"].quantile(.95)],
                        "max":[stable["t"].max(), crisis["t"].max()]},
                       index=["stable","crisis"]).round(2)
    print("\n=== Turbulence index statistics ==="); print(out.to_string())
    if draw:
        kde_plot(stable["t"], crisis["t"],
                 "Turbulence KDE", "kde_turb.png")

# -- mapping --
ANALYSIS_FUNCS = {
    "log_ret": analyze_log_returns,
    "volatility": analyze_volatility,
    "rsi": analyze_rsi_signals,
    "boll": analyze_bollinger_behavior,
    "volume": analyze_volume,
    "corr": analyze_correlation,
    "turb": analyze_turbulence,
}

# ─────────────────────────────── CLI ─────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Statistical analysis for DJIA.")
    p.add_argument("--mode",   choices=list(ANALYSIS_FUNCS)+["all"], default="all")
    p.add_argument("--period", choices=["train","val","test"],        default="train")
    p.add_argument("--draw",   type=str, choices=["true","false"],    default="false",
                   help="If 'true', draw KDE plots to figures/ and show them.")
    return p.parse_args()

# ─────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    df, bounds = get_period_data(args.period)
    draw_flag = (args.draw.lower() == "true")
    if args.mode == "all":
        for fn in ANALYSIS_FUNCS.values():
            fn(df.copy(), bounds, draw_flag)
    else:
        ANALYSIS_FUNCS[args.mode](df.copy(), bounds, draw_flag)
