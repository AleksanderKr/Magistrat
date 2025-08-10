#!/usr/bin/env python
"""
Examples
--------
python -m finrl.analyze_dow30 --period train --mode all
python -m finrl.analyze_dow30 --period train  --mode all --draw true
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, levene
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
STABLE_COLOR = "#42cef5"
CRISIS_COLOR = "#f55151"

def kde_plot(series_s: pd.Series, series_c: pd.Series, title: str, filename: str, xlabel: str, percent: bool=False):
    s_clean = pd.to_numeric(series_s, errors="coerce").dropna()
    c_clean = pd.to_numeric(series_c, errors="coerce").dropna()
    if s_clean.nunique() < 2 or c_clean.nunique() < 2:
        print(f"[warn] Skipping KDE for {filename}: one of the series is constant.")
        return

    xmin = float(min(s_clean.min(), c_clean.min()))
    xmax = float(max(s_clean.max(), c_clean.max()))
    if not np.isfinite([xmin, xmax]).all() or xmax <= xmin:
        print(f"[warn] Skipping KDE for {filename}: invalid domain.")
        return

    xs = np.linspace(xmin, xmax, 400)
    kde_s = gaussian_kde(s_clean.values.astype(float))(xs)
    kde_c = gaussian_kde(c_clean.values.astype(float))(xs)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, kde_s, label="stable", lw=2.0, color=STABLE_COLOR)
    ax.plot(xs, kde_c, label="crisis", lw=2.0, ls="--", color=CRISIS_COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)          # np. "Daily log return", "σ over 30 days", "RSI (0–100)"
    ax.set_ylabel("Density")
    if percent:                    # używaj True tylko dla log-returns i σ
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
    ax.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

def bar_plot(values, labels, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(5, 3))
    width = 0.25
    gap = 0.35
    x = np.arange(len(values)) * (width + gap)
    colors = [STABLE_COLOR, CRISIS_COLOR][:len(values)]
    bars = ax.bar(x, values, width=width, color=colors)
    ymax = np.nanmax(values) if len(values) else 1.0
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:.0f}%"))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(handles=bars, labels=labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)


def bar_plot_grouped(groups, bars, values, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.25
    bar_gap = 0.05
    group_gap = 0.4
    n_groups = len(groups)
    n_bars = len(bars)
    cluster_width = n_bars * width + (n_bars - 1) * bar_gap
    step = cluster_width + group_gap
    x0 = np.arange(n_groups) * step
    palette = [STABLE_COLOR, CRISIS_COLOR] + ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    ymax = np.nanmax(np.concatenate([np.asarray(v, dtype=float) for v in values])) if values else 1.0
    for i in range(n_bars):
        xi = x0 + i * (width + bar_gap)
        ax.bar(xi, values[i], width=width, label=bars[i], color=palette[i % len(palette)])
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xticks(x0 + cluster_width / 2 - width / 2, groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:.0f}%"))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

# ───────────────────────── analyses (each draw-aware) ────────────────────
def analyze_log_returns(df: pd.DataFrame, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    stats = lambda s: s.dropna().agg(["mean", "std", "skew", "kurt"])
    out = pd.concat({"stable": stats(stable["log_ret"]), "crisis": stats(crisis["log_ret"])}, axis=1).T.round(5)
    print("\n=== Log-return summary ===")
    print(out.to_string())
    s_lr = stable["log_ret"].dropna()
    c_lr = crisis["log_ret"].dropna()
    if len(s_lr) > 2 and len(c_lr) > 2:
        stat, p = levene(s_lr, c_lr)
        print(f"\nLevene variance test: stat={stat:.3f}  p={p:.4e}")
    else:
        print("\nLevene variance test: skipped (insufficient data)")
    if draw:
        kde_plot(s_lr, c_lr, "Log-return (Test Set)", "test_kde_logret.png", "Daily log return", True)

def analyze_volatility(df: pd.DataFrame, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    df["vol_30"]  = df.groupby("tic")["log_ret"].transform(lambda x: x.rolling(30).std())
    stable, crisis = split_sets(df, bounds)
    res = pd.DataFrame({"mean_vol": [stable["vol_30"].mean(), crisis["vol_30"].mean()],
                        "median_vol": [stable["vol_30"].median(), crisis["vol_30"].median()]},
                       index=["stable", "crisis"]).round(5)
    print("\n=== Rolling 30-day volatility ===")
    print(res.to_string())
    if draw:
        kde_plot(stable["vol_30"].dropna(), crisis["vol_30"].dropna(), "30-day Volatility (Test Set)", "test_kde_vol30.png", "σ over 30 days", True)


def analyze_rsi_signals(df: pd.DataFrame, bounds, draw=False):
    df = df.sort_values(["tic", "timestamp"]).copy()
    df["rsi_30"] = Sdf.retype(df.copy())["rsi_30"]
    stable, crisis = split_sets(df, bounds)
    f = lambda s: pd.Series({"days>70": (s["rsi_30"] > 70).sum(), "days<30": (s["rsi_30"] < 30).sum(), "mean_rsi": s["rsi_30"].mean()})
    out = pd.concat({"stable": f(stable), "crisis": f(crisis)}, axis=1).T.round(2)
    print("\n=== RSI signal counts ===")
    print(out.to_string())
    if draw:
        kde_plot(stable["rsi_30"], crisis["rsi_30"], "RSI30 (Test Set)", "test_kde_rsi30.png", "RSI (0–100)", False)

def analyze_bollinger_behavior(df, bounds, draw=False):
    df = df.sort_values(["tic","timestamp"])
    ss = Sdf.retype(df.copy())
    df["boll_ub"], df["boll_lb"] = ss["boll_ub"], ss["boll_lb"]
    df["up"] = df["close"] > df["boll_ub"]; df["low"] = df["close"] < df["boll_lb"]
    stable, crisis = split_sets(df, bounds)
    g = lambda s: pd.Series({"breaks_upper": s["up"].sum(), "breaks_lower": s["low"].sum(),
                             "n_obs": np.minimum(s["boll_ub"].notna(), s["boll_lb"].notna()).sum()})
    tmp = pd.concat({"stable":g(stable),"crisis":g(crisis)}, axis=1).T
    tmp["rate_upper"] = tmp["breaks_upper"] / tmp["n_obs"]
    tmp["rate_lower"] = tmp["breaks_lower"] / tmp["n_obs"]
    out = tmp[["breaks_upper","breaks_lower","rate_upper","rate_lower"]]
    print("\n=== Bollinger breakouts ==="); print(out.to_string(index=True))
    if draw:
        bar_plot([out.loc["stable","rate_upper"], out.loc["crisis","rate_upper"]],
                 ["stable","crisis"], "Share of upper-band breakouts (Test Set)", "test_boll_upper_rate.png", "Share of days", True)
        bar_plot([out.loc["stable","rate_lower"], out.loc["crisis","rate_lower"]],
                 ["stable","crisis"], "Share of lower-band breakouts (Test Set)", "test_boll_lower_rate.png", "Share of days", True)

def analyze_correlation(df, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    def avg(s):
        p = s.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        if p.empty: return np.nan
        c = p.corr()
        return c.values[np.triu_indices_from(c,1)].mean()
    out = pd.DataFrame({"avg_corr":[avg(stable), avg(crisis)]}, index=["stable","crisis"]).round(5)
    print("\n=== Avg pair-wise return correlation ==="); print(out.to_string())
    if draw and out["avg_corr"].notna().all():
        bar_plot([out.loc["stable","avg_corr"], out.loc["crisis","avg_corr"]],
                 ["stable","crisis"], "Average pair-wise correlation (Test Set)", "test_avg_corr_bar.png", "Correlation", False)


def analyze_volume(df: pd.DataFrame, bounds, draw=False):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    corr_s = stable["volume"].corr(stable["log_ret"])
    corr_c = crisis["volume"].corr(crisis["log_ret"])
    out = pd.DataFrame({"mean_vol": [stable["volume"].mean(), crisis["volume"].mean()],
                        "median_vol": [stable["volume"].median(), crisis["volume"].median()],
                        "corr_vol_ret": [corr_s, corr_c]},
                       index=["stable", "crisis"]).round(3)
    print("\n=== Volume stats ===")
    print(out.to_string())
    if draw:
        kde_plot(np.log1p(stable["volume"]), np.log1p(crisis["volume"]), "log(Volume) (Test Set)", "test_kde_volume.png", "log(1+volume)", False)

def analyze_turbulence(df: pd.DataFrame, bounds, draw=False):
    price = df.pivot(index="timestamp", columns="tic", values="close").sort_index()
    ret   = price.pct_change().dropna()

    w = 252
    turb = pd.Series(np.nan, index=ret.index, dtype=float)
    for i in range(w, len(ret)):
        hist = ret.iloc[i - w : i]
        cur  = ret.iloc[i]
        valid = cur.dropna().index
        hist, cur = hist[valid], cur[valid]
        if hist.shape[0] < 2 or len(valid) < 2:
            continue
        cov  = hist.cov()
        diff = cur - hist.mean()
        try:
            inv = np.linalg.pinv(cov.values)
            turb.iloc[i] = float(diff.values @ inv @ diff.values.T)
        except Exception:
            continue

    turb_df = (
        turb.to_frame("t")
            .reset_index()
            .rename(columns={"index": "timestamp"})
    )

    stable, crisis = split_sets(turb_df, bounds)

    stable_t = stable["t"].dropna()
    crisis_t = crisis["t"].dropna()

    out = pd.DataFrame(
        {
            "mean":   [stable_t.mean(),         crisis_t.mean()],
            "median": [stable_t.median(),       crisis_t.median()],
            "p95":    [stable_t.quantile(0.95), crisis_t.quantile(0.95)],
            "max":    [stable_t.max(),          crisis_t.max()],
        },
        index=["stable", "crisis"],
    ).round(2)

    print("\n=== Turbulence index statistics ===")
    print(out.to_string())

    if draw and len(stable_t) > 1 and len(crisis_t) > 1:
        kde_plot(stable_t, crisis_t, "Turbulence (Test Set))", "test_kde_turb.png", "Turbulence index", False)

def compute_bollinger_rates(df, bounds):
    df = df.sort_values(["tic","timestamp"]).copy()
    ss = Sdf.retype(df.copy())
    df["boll_ub"] = ss["boll_ub"]
    df["boll_lb"] = ss["boll_lb"]
    df["up"]  = df["close"] > df["boll_ub"]
    df["low"] = df["close"] < df["boll_lb"]
    stable, crisis = split_sets(df, bounds)
    f = lambda s: pd.Series({
        "rate_upper": (s["up"]  & df.loc[s.index, ["boll_ub","boll_lb"]].notna().all(axis=1)).sum() / df.loc[s.index, ["boll_ub","boll_lb"]].notna().all(axis=1).sum(),
        "rate_lower": (s["low"] & df.loc[s.index, ["boll_ub","boll_lb"]].notna().all(axis=1)).sum() / df.loc[s.index, ["boll_ub","boll_lb"]].notna().all(axis=1).sum(),
    })
    out = pd.concat({"stable": f(stable), "crisis": f(crisis)}, axis=1).T
    return out.loc["stable","rate_upper"], out.loc["crisis","rate_upper"], out.loc["stable","rate_lower"], out.loc["crisis","rate_lower"]

def plot_bollinger_rates_all():
    periods = ["train","val","test"]
    stable_upper, crisis_upper, stable_lower, crisis_lower = [], [], [], []
    for p in periods:
        df, bounds = get_period_data(p)
        us, uc, ls, lc = compute_bollinger_rates(df, bounds)
        stable_upper.append(us); crisis_upper.append(uc)
        stable_lower.append(ls); crisis_lower.append(lc)
    bars = ["stable","crisis"]
    vals_upper = [stable_upper, crisis_upper]
    vals_lower = [stable_lower, crisis_lower]
    bar_plot_grouped(groups=periods, bars=bars, values=vals_upper,
                     title="Upper-band breakout rate", filename="boll_upper_rate_all.png",
                     ylabel="Share of days", percent=True)
    bar_plot_grouped(groups=periods, bars=bars, values=vals_lower,
                     title="Lower-band breakout rate", filename="boll_lower_rate_all.png",
                     ylabel="Share of days", percent=True)

def compute_avg_corr(df, bounds):
    df = df.copy()
    df["log_ret"] = df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))
    stable, crisis = split_sets(df, bounds)
    def avg(s):
        p = s.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        if p.empty: return np.nan
        c = p.corr()
        return c.values[np.triu_indices_from(c,1)].mean()
    return avg(stable), avg(crisis)

def plot_avg_corr_all():
    periods = ["train","val","test"]
    stable_vals, crisis_vals = [], []
    for p in periods:
        df, bounds = get_period_data(p)
        s, c = compute_avg_corr(df, bounds)
        stable_vals.append(s); crisis_vals.append(c)
    bars = ["stable","crisis"]
    bar_plot_grouped(groups=periods, bars=bars, values=[stable_vals, crisis_vals],
                     title="Average pair-wise correlation", filename="avg_corr_all.png",
                     ylabel="Correlation", percent=False)


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
    draw_flag = True if args.mode == "all" else (args.draw.lower() == "true")

    if args.mode == "all":
        for fn in ANALYSIS_FUNCS.values():
            fn(df.copy(), bounds, draw_flag)
        if draw_flag:
            try:
                plot_bollinger_rates_all()
            except Exception as e:
                print(f"[warn] plot_bollinger_rates_all failed: {e}")
            try:
                plot_avg_corr_all()
            except Exception as e:
                print(f"[warn] plot_avg_corr_all failed: {e}")
    else:
        ANALYSIS_FUNCS[args.mode](df.copy(), bounds, draw_flag)
