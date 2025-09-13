#!/usr/bin/env python
"""
Examples
--------
python -m finrl.analyze_dow30_v2 --period train --mode all
python -m finrl.analyze_dow30_v2 --period val --mode boll --draw true
python -m finrl.analyze_dow30_v2 --period train --mode all --draw true --include-intraday-on-daily true
"""

#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, fligner
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
    BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE,
    BEAR2008_VALIDATION_START_DATE, BEAR2008_VALIDATION_END_DATE,
    BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE,
    INTRA_TRAIN_START, INTRA_TRAIN_END,
    INTRA_VAL_START, INTRA_VAL_END,
    INTRA_TEST_START, INTRA_TEST_END,
    TURB_INTRADAY_BUFFER_DAYS, TURB_DAILY_BUFFER_DAYS
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TECHNICALS = ["rsi_30", "boll_ub", "boll_lb"]

REG_KEYS = ["bull", "bullv", "bearish", "intra"]
REG_LABELS = dict(bull="Bullish Stable", bullv="Bullish Volatile", bearish="Bearish", intra="Intraday")
REG_COLORS = dict(bull="#6baed6", bullv="#fc9272", bearish="#74c476", intra="#9467bd")

BULL_STABLE_PERIODS = dict(
    train=(TRAIN_START_DATE, TRAIN_END_DATE),
    val=(VALIDATION_START_DATE, VALIDATION_END_DATE),
    test=(TEST_START_DATE, TEST_END_DATE),
)
BULL_VOLATILE_PERIODS = dict(
    train=(CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE),
    val=(CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE),
    test=(CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE),
)
BEARISH_PERIODS = dict(
    train=(BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE),
    val=(BEAR2008_VALIDATION_START_DATE, BEAR2008_VALIDATION_END_DATE),
    test=(BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE),
)
INTRA_PERIODS = dict(
    train=(INTRA_TRAIN_START, INTRA_TRAIN_END),
    val=(INTRA_VAL_START, INTRA_VAL_END),
    test=(INTRA_TEST_START, INTRA_TEST_END),
)

INTRA3_COLORS = dict(train="#2ca02c", val="#9467bd", test="#8c564b")

def _period_name(p: str) -> str:
    return dict(train="Train", val="Validation", test="Test")[p]

def titles(period: str) -> Dict[str, str]:
    P = _period_name(period)
    return {
        "logret_daily": f"Log-return density (daily regimes, {P})",
        "logret_intra": f"Log-return density (Intraday, {P})",
        "vol_daily":    f"Rolling volatility σ, 30-day window (Daily, {P})",
        "vol_intra":    f"Rolling volatility σ, 390-minute window (Intraday, {P})",
        "rsi_daily":    f"RSI(30) density (Daily regimes, {P})",
        "rsi_intra":    f"RSI(30) density (Intraday, {P})",
        "volu_daily":   f"log(1+volume) density (Daily regimes, {P})",
        "volu_intra":   f"log(1+volume) density (Intraday, {P})",
        "turb_daily":   f"Turbulence index density (Daily regimes, {P})",
        "turb_intra":   f"Turbulence index density (Intraday, {P})",
        "bar_ma":       f"Share of price>EMA200 ({P})",
        "bar_macd":     f"Share of MACD>signal ({P})",
        "bar_adx":      f"Share of ADX>25 ({P})",
        "bar_slope":    f"Share of positive log-price slope ({P})",
        "y_density":    "Density",
        "y_share":      "Share of observations",
        "y_corr":       "Correlation",
    }

def _percentile_limits(*arrays: pd.Series, p_lo: float = 1.0, p_hi: float = 99.0) -> Optional[Tuple[float, float]]:
    vals = []
    for a in arrays:
        x = pd.to_numeric(a, errors="coerce").dropna().values
        if x.size:
            vals.append(x)
    if not vals:
        return None
    cat = np.concatenate(vals)
    lo = float(np.nanpercentile(cat, p_lo))
    hi = float(np.nanpercentile(cat, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi

def _shared_xs(min_x: float, max_x: float, n: int = 400) -> np.ndarray:
    return np.linspace(min_x, max_x, n)

def _safe_kde(y: pd.Series, xs: np.ndarray) -> Optional[np.ndarray]:
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if y.nunique() < 2:
        return None
    try:
        return gaussian_kde(y.values)(xs)
    except Exception:
        return None

def _fmt_pct(x, pos):
    return f"{x*100:.0f}%"

def load_data(tickers: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    dp = DataProcessor("yahoofinance")
    d = dp.download_data(tickers, start, end, interval)
    d = dp.clean_data(d)
    d = dp.add_technical_indicator(d, TECHNICALS)
    return d.reset_index(drop=True)

def get_period_frames(period: str) -> Dict[str, pd.DataFrame]:
    bs, be = BULL_STABLE_PERIODS[period]
    bv_s, bv_e = BULL_VOLATILE_PERIODS[period]
    br_s, br_e = BEARISH_PERIODS[period]
    is_, ie = INTRA_PERIODS[period]
    bull = load_data(DOW_30_TICKER, bs, be, "1d")
    bullv = load_data(DOW_30_TICKER, bv_s, bv_e, "1d")
    bearish = load_data(DOW_30_TICKER, br_s, br_e, "1d")
    intra = load_data(DOW_30_TICKER, is_, ie, "1m")
    return {"bull": bull, "bullv": bullv, "bearish": bearish, "intra": intra}

def get_period_frames_with_buffer(period: str) -> Dict[str, pd.DataFrame]:
    bs, be = BULL_STABLE_PERIODS[period]
    bv_s, bv_e = BULL_VOLATILE_PERIODS[period]
    br_s, br_e = BEARISH_PERIODS[period]
    is_, ie = INTRA_PERIODS[period]
    bs_buf = (pd.Timestamp(bs) - pd.Timedelta(days=TURB_DAILY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    bv_buf = (pd.Timestamp(bv_s) - pd.Timedelta(days=TURB_DAILY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    br_buf = (pd.Timestamp(br_s) - pd.Timedelta(days=TURB_DAILY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    is_buf = (pd.Timestamp(is_) - pd.Timedelta(days=TURB_INTRADAY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    bull = load_data(DOW_30_TICKER, bs_buf, be, "1d")
    bullv = load_data(DOW_30_TICKER, bv_buf, bv_e, "1d")
    bearish = load_data(DOW_30_TICKER, br_buf, br_e, "1d")
    intra = load_data(DOW_30_TICKER, is_buf, ie, "1m")
    return {"bull": bull, "bullv": bullv, "bearish": bearish, "intra": intra}

def _get_intra_frames_all(buffer_for_turb: bool = False) -> Dict[str, pd.DataFrame]:
    if buffer_for_turb:
        tr = get_period_frames_with_buffer("train")["intra"]
        va = get_period_frames_with_buffer("val")["intra"]
        te = get_period_frames_with_buffer("test")["intra"]
    else:
        tr = get_period_frames("train")["intra"]
        va = get_period_frames("val")["intra"]
        te = get_period_frames("test")["intra"]
    return {"train": tr.copy(), "val": va.copy(), "test": te.copy()}

def _log_ret(df: pd.DataFrame) -> pd.Series:
    return df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))

def kde_plot_many(series_list: List[pd.Series], labels: List[str], title: str, filename: str, xlabel: str, percent: bool=False):
    xs_minmax = _percentile_limits(*series_list, p_lo=1.0, p_hi=99.0)
    if not xs_minmax:
        return
    xs = _shared_xs(xs_minmax[0], xs_minmax[1], 400)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for s, lab in zip(series_list, labels):
        s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
        if s.nunique() < 2:
            continue
        k = _safe_kde(s, xs)
        if k is None:
            continue
        ax.plot(xs, k, lw=2.0, label=f"{lab} (N={len(s)})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

def kde_plot_three(a: pd.Series, b: pd.Series, c: pd.Series, title: str, filename: str, xlabel: str, percent: bool=False, labels: Tuple[str, str, str]=("Train", "Validation", "Test")):
    sA = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    sB = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    sC = pd.to_numeric(c, errors="coerce").dropna().astype(float)
    if min(sA.nunique(), sB.nunique(), sC.nunique()) < 2:
        return
    lims = _percentile_limits(sA, sB, sC, p_lo=1.0, p_hi=99.0)
    if not lims:
        return
    xmin, xmax = lims
    xs = _shared_xs(xmin, xmax, 400)
    kA = _safe_kde(sA, xs); kB = _safe_kde(sB, xs); kC = _safe_kde(sC, xs)
    if kA is None or kB is None or kC is None:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(xs, kA, lw=2.0, label=f"{labels[0]} (N={len(sA)})", alpha=0.95)
    ax.plot(xs, kB, lw=2.0, ls="--", label=f"{labels[1]} (N={len(sB)})", alpha=0.95)
    ax.plot(xs, kC, lw=2.0, ls="-.", label=f"{labels[2]} (N={len(sC)})", alpha=0.95)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / filename, dpi=200); plt.close(fig)

def kde_plot_one(a: pd.Series, title: str, filename: str, xlabel: str, percent: bool=False):
    s = pd.to_numeric(a, errors="coerce").dropna()
    if s.nunique() < 2:
        return
    lims = _percentile_limits(s, p_lo=1.0, p_hi=99.0)
    if not lims:
        return
    xs = _shared_xs(lims[0], lims[1], 400)
    k = _safe_kde(s, xs)
    if k is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(xs, k, lw=2.0, alpha=0.95, label=f"{REG_LABELS['intra']} (N={len(s)})")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / filename, dpi=200); plt.close(fig)

def bar_plot(values, labels, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.24
    gap = 0.30
    x = np.arange(len(values)) * (width + gap)
    ax.bar(x, values, width=width)
    ymax = np.nanmax(values) if len(values) else 1.0
    if not np.isfinite(ymax):
        ymax = 1.0
    if percent:
        ax.set_ylim(0, ymax + 0.05)
    else:
        ax.set_ylim(0, ymax * 1.5)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

def bar_plot_grouped(groups, bars, values, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    width = 0.22
    bar_gap = 0.06
    group_gap = 0.40
    n_groups = len(groups)
    n_bars = len(bars)
    cluster_width = n_bars * width + (n_bars - 1) * bar_gap
    step = cluster_width + group_gap
    x0 = np.arange(n_groups) * step
    flat = []
    for v in values:
        v = np.asarray(v, dtype=float)
        flat.append(v)
    flat = np.concatenate(flat) if len(flat) else np.array([0.0])
    ymax = float(np.nanmax(flat)) if flat.size else 1.0
    if not np.isfinite(ymax):
        ymax = 1.0
    for i in range(n_bars):
        xi = x0 + i * (width + bar_gap)
        ax.bar(xi, values[i], width=width, label=bars[i])
    if percent:
        ax.set_ylim(0, ymax + 0.03)
    else:
        ax.set_ylim(0, ymax * 1.5)
    ax.set_xticks(x0 + cluster_width / 2 - width / 2, groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

def analyze_log_returns(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False, include_intraday_on_daily=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in dfs:
        dfs[k]["log_ret"] = _log_ret(dfs[k])
    stats = lambda s: s.dropna().agg(["mean", "std", "skew", "kurt"])
    out = pd.concat(
        {
            REG_LABELS["bull"]:   stats(dfs["bull"]["log_ret"]),
            REG_LABELS["bullv"]:  stats(dfs["bullv"]["log_ret"]),
            REG_LABELS["bearish"]:stats(dfs["bearish"]["log_ret"]),
            REG_LABELS["intra"]:  stats(dfs["intra"]["log_ret"]),
        }, axis=1
    ).T
    out["N"] = [
        dfs["bull"]["log_ret"].dropna().shape[0],
        dfs["bullv"]["log_ret"].dropna().shape[0],
        dfs["bearish"]["log_ret"].dropna().shape[0],
        dfs["intra"]["log_ret"].dropna().shape[0],
    ]
    out = out[["mean","std","skew","kurt","N"]].round(6)
    print("\n=== Log-return summary ===")
    print(out.to_string())

    s1, s2, s3, s4 = (
        dfs["bull"]["log_ret"].dropna(),
        dfs["bullv"]["log_ret"].dropna(),
        dfs["bearish"]["log_ret"].dropna(),
        dfs["intra"]["log_ret"].dropna(),
    )
    if min(len(s1), len(s2), len(s3), len(s4)) > 2:
        stat, p = fligner(s1, s2, s3, s4, center="median")
        print(f"\nFligner–Killeen test (variances): stat={stat:.3f}  p={p:.4e}")
    else:
        print("\nFligner–Killeen test: skipped (insufficient data)")

    if draw:
        T = titles(period)
        kde_plot_many([s1, s2, s3] + ([s4] if include_intraday_on_daily else []),
                      [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else []),
                      T["logret_daily"], f"{period}_kde_logret_daily.png", "Log return", True)
        F = _get_intra_frames_all(False)
        A = _log_ret(F["train"]).dropna(); B = _log_ret(F["val"]).dropna(); C = _log_ret(F["test"]).dropna()
        kde_plot_three(A, B, C, "Log-return density (Intraday, Train/Val/Test)", "intra_kde_logret_all.png", "Log return", True)

def analyze_volatility(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False, include_intraday_on_daily=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in ["bull", "bullv", "bearish"]:
        dfs[k]["lr"] = _log_ret(dfs[k])
        dfs[k]["vol"] = dfs[k].groupby("tic")["lr"].transform(lambda x: x.rolling(30, min_periods=10).std())
        dfs[k]["vol_ann"] = dfs[k]["vol"] * np.sqrt(252)
    dfs["intra"]["lr"] = _log_ret(dfs["intra"])
    dfs["intra"]["vol"] = dfs["intra"].groupby("tic")["lr"].transform(lambda x: x.rolling(390, min_periods=120).std())
    dfs["intra"]["vol_day"] = dfs["intra"]["vol"] * np.sqrt(390)
    dfs["intra"]["vol_ann"] = dfs["intra"]["vol"] * np.sqrt(390 * 252)

    res = pd.DataFrame(
        {
            "mean_vol": [dfs["bull"]["vol"].mean(), dfs["bullv"]["vol"].mean(), dfs["bearish"]["vol"].mean(), dfs["intra"]["vol"].mean()],
            "median_vol": [dfs["bull"]["vol"].median(), dfs["bullv"]["vol"].median(), dfs["bearish"]["vol"].median(), dfs["intra"]["vol"].median()],
            "mean_vol_ann": [dfs["bull"]["vol_ann"].mean(), dfs["bullv"]["vol_ann"].mean(), dfs["bearish"]["vol_ann"].mean(), dfs["intra"]["vol_ann"].mean()],
            "N": [
                dfs["bull"]["vol"].dropna().shape[0],
                dfs["bullv"]["vol"].dropna().shape[0],
                dfs["bearish"]["vol"].dropna().shape[0],
                dfs["intra"]["vol"].dropna().shape[0],
            ],
        },
        index=[REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"], REG_LABELS["intra"]],
    ).round(6)
    print("\n=== Rolling volatility (30d daily; 390m intraday) ===")
    print(res.to_string())

    if draw:
        T = titles(period)
        kde_plot_many(
            [dfs["bull"]["vol"].dropna(), dfs["bullv"]["vol"].dropna(), dfs["bearish"]["vol"].dropna()] + ([dfs["intra"]["vol"].dropna()] if include_intraday_on_daily else []),
            [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else []),
            T["vol_daily"], f"{period}_kde_vol_daily.png", "σ (30-day, daily)", True
        )
        F = _get_intra_frames_all(False)
        def v(df):
            r = _log_ret(df)
            return df.assign(lr=r).groupby("tic")["lr"].transform(lambda x: x.rolling(390, min_periods=120).std())
        A = v(F["train"]).dropna(); B = v(F["val"]).dropna(); C = v(F["test"]).dropna()
        kde_plot_three(A, B, C, "Rolling volatility σ, 390-minute window (Intraday, Train/Val/Test)", "intra_kde_vol_all.png", "σ (390-minute, intraday)", True)

def analyze_rsi_signals(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False, ci=False, include_intraday_on_daily=False):
    dfs = {k: v.sort_values(["tic", "timestamp"]).copy() for k, v in frames.items()}
    for k in dfs:
        dfs[k]["rsi_30"] = Sdf.retype(dfs[k].copy())["rsi_30"]

    def pack(s):
        valid = s["rsi_30"].notna()
        n = int(valid.sum())
        gt = int((s.loc[valid, "rsi_30"] > 70).sum())
        lt = int((s.loc[valid, "rsi_30"] < 30).sum())
        p_gt = gt / n if n else np.nan
        p_lt = lt / n if n else np.nan
        if ci and n:
            z = 1.95996398454
            h_gt = z * np.sqrt(p_gt * (1 - p_gt) / n) if np.isfinite(p_gt) else np.nan
            h_lt = z * np.sqrt(p_lt * (1 - p_lt) / n) if np.isfinite(p_lt) else np.nan
            lo_gt, hi_gt = max(0.0, p_gt - h_gt), min(1.0, p_gt + h_gt)
            lo_lt, hi_lt = max(0.0, p_lt - h_lt), min(1.0, p_lt + h_lt)
        else:
            lo_gt = hi_gt = lo_lt = hi_lt = np.nan
        return pd.Series(
            {
                "days>70": gt,
                "days<30": lt,
                "mean_rsi": s["rsi_30"].mean(),
                "N": n,
                "p>70": p_gt,
                "p<30": p_lt,
                "p>70_CIlo": lo_gt,
                "p>70_CIhi": hi_gt,
                "p<30_CIlo": lo_lt,
                "p<30_CIhi": hi_lt,
            }
        )

    out = pd.concat(
        {
            REG_LABELS["bull"]: pack(dfs["bull"]),
            REG_LABELS["bullv"]: pack(dfs["bullv"]),
            REG_LABELS["bearish"]: pack(dfs["bearish"]),
            REG_LABELS["intra"]: pack(dfs["intra"]),
        }, axis=1
    ).T
    out.loc[:, ["mean_rsi", "p>70", "p<30", "p>70_CIlo", "p>70_CIhi", "p<30_CIlo", "p<30_CIhi"]] = out.loc[
        :, ["mean_rsi", "p>70", "p<30", "p>70_CIlo", "p>70_CIhi", "p<30_CIlo", "p<30_CIhi"]
    ].astype(float).round(4)
    print("\n=== RSI(30) signal counts and shares ===")
    print(out.to_string())

    if draw:
        T = titles(period)
        kde_plot_many(
            [dfs["bull"]["rsi_30"].dropna(), dfs["bullv"]["rsi_30"].dropna(), dfs["bearish"]["rsi_30"].dropna()] + ([dfs["intra"]["rsi_30"].dropna()] if include_intraday_on_daily else []),
            [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else []),
            T["rsi_daily"], f"{period}_kde_rsi_daily.png", "RSI (0–100)", False
        )
        F = _get_intra_frames_all(False)
        def r(df): return Sdf.retype(df.sort_values(["tic","timestamp"]).copy())["rsi_30"]
        A = r(F["train"]).dropna(); B = r(F["val"]).dropna(); C = r(F["test"]).dropna()
        kde_plot_three(A, B, C, "RSI(30) density (Intraday, Train/Val/Test)", "intra_kde_rsi30_all.png", "RSI (0–100)", False)

def analyze_bollinger_behavior(frames: Dict[str, pd.DataFrame], period: str, draw=False, ci=False, include_intraday_on_daily=False):
    dfs = {k: v.sort_values(["tic", "timestamp"]).copy() for k, v in frames.items()}
    for k in dfs:
        ss = Sdf.retype(dfs[k].copy())
        dfs[k]["boll_ub"] = ss["boll_ub"]
        dfs[k]["boll_lb"] = ss["boll_lb"]
        dfs[k]["up"] = dfs[k]["close"] > dfs[k]["boll_ub"]
        dfs[k]["low"] = dfs[k]["close"] < dfs[k]["boll_lb"]

    def g(s):
        valid = np.minimum(s["boll_ub"].notna(), s["boll_lb"].notna())
        n = int(valid.sum())
        bu = int((s["up"] & valid).sum())
        bl = int((s["low"] & valid).sum())
        pu = bu / n if n else np.nan
        pl = bl / n if n else np.nan
        if ci and n:
            z = 1.95996398454
            hu = z * np.sqrt(pu * (1 - pu) / n) if np.isfinite(pu) else np.nan
            hl = z * np.sqrt(pl * (1 - pl) / n) if np.isfinite(pl) else np.nan
            lo_u, hi_u = max(0.0, pu - hu), min(1.0, pu + hu)
            lo_l, hi_l = max(0.0, pl - hl), min(1.0, pl + hl)
        else:
            lo_u = hi_u = lo_l = hi_l = np.nan
        return pd.Series(
            {
                "breaks_upper": bu,
                "breaks_lower": bl,
                "rate_upper": pu,
                "rate_lower": pl,
                "n_obs": n,
                "rate_upper_CIlo": lo_u,
                "rate_upper_CIhi": hi_u,
                "rate_lower_CIlo": lo_l,
                "rate_lower_CIhi": hi_l,
            }
        )

    out = pd.concat({REG_LABELS["bull"]: g(dfs["bull"]), REG_LABELS["bullv"]: g(dfs["bullv"]), REG_LABELS["bearish"]: g(dfs["bearish"]), REG_LABELS["intra"]: g(dfs["intra"])}, axis=1).T
    out = out.round(4)
    print("\n=== Bollinger breakouts ===")
    print(out.to_string(index=True))
    return out

def analyze_correlation(frames: Dict[str, pd.DataFrame], period: str, draw=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in dfs:
        dfs[k]["log_ret"] = _log_ret(dfs[k])

    def avg(s):
        p = s.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        if p.empty:
            return np.nan
        c = p.corr()
        return c.values[np.triu_indices_from(c, 1)].mean()

    vals = {
        REG_LABELS["bull"]: avg(dfs["bull"]),
        REG_LABELS["bullv"]: avg(dfs["bullv"]),
        REG_LABELS["bearish"]: avg(dfs["bearish"]),
        REG_LABELS["intra"]: avg(dfs["intra"]),
    }
    out = pd.DataFrame.from_dict(vals, orient="index", columns=["avg_corr"]).round(5)
    print("\n=== Avg pair-wise return correlation ===")
    print(out.to_string())
    return out

def analyze_volume(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False, include_intraday_on_daily=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in dfs:
        dfs[k]["log_ret"] = _log_ret(dfs[k])

    def pack(s):
        return pd.Series(
            {
                "mean_vol": s["volume"].mean(),
                "median_vol": s["volume"].median(),
                "corr_vol_ret": s["volume"].corr(s["log_ret"]),
                "N": s["volume"].dropna().shape[0],
            }
        )

    out = pd.concat({
        REG_LABELS["bull"]: pack(dfs["bull"]),
        REG_LABELS["bullv"]: pack(dfs["bullv"]),
        REG_LABELS["bearish"]: pack(dfs["bearish"]),
        REG_LABELS["intra"]: pack(dfs["intra"])
    }, axis=1).T.round(4)
    print("\n=== Volume stats ===")
    print(out.to_string())

    if draw:
        T = titles(period)
        kde_plot_many(
            [np.log1p(dfs["bull"]["volume"]), np.log1p(dfs["bullv"]["volume"]), np.log1p(dfs["bearish"]["volume"])] + ([np.log1p(dfs["intra"]["volume"])] if include_intraday_on_daily else []),
            [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else []),
            T["volu_daily"], f"{period}_kde_volume_daily.png", "log(1+volume)", False
        )
        F = _get_intra_frames_all(False)
        A = np.log1p(F["train"]["volume"]).dropna(); B = np.log1p(F["val"]["volume"]).dropna(); C = np.log1p(F["test"]["volume"]).dropna()
        kde_plot_three(A, B, C, "log(1+volume) density (Intraday, Train/Val/Test)", "intra_kde_volume_all.png", "log(1+volume)", False)

def mahal_turbulence_from_df(df: pd.DataFrame, window: int,
                             min_assets: int = 2, min_obs: int = 10) -> pd.Series:
    price = df.pivot(index="timestamp", columns="tic", values="close").sort_index()
    ret = price.pct_change()
    if ret.empty:
        return pd.Series(dtype=float, index=ret.index)
    tser = pd.Series(np.nan, index=ret.index, dtype=float)
    for i in range(window, len(ret)):
        hist = ret.iloc[i - window:i]
        cur = ret.iloc[i]
        valid = cur.dropna().index
        if len(valid) < min_assets:
            continue
        hist = hist[valid].dropna(how="all")
        if hist.shape[0] < min_obs:
            continue
        cov = hist.cov()
        mu = hist.mean()
        diag = np.diag(cov.values)
        keep = cov.columns[diag > 0]
        if len(keep) < min_assets:
            continue
        cov = cov.loc[keep, keep]
        mu = mu[keep]
        curv = cur[keep]
        try:
            inv = np.linalg.pinv(cov.values)
            diff = (curv.values - mu.values).reshape(1, -1)
            tser.iloc[i] = (diff @ inv @ diff.T).item()
        except Exception:
            continue
    return tser

def mahal_turbulence_intraday(df: pd.DataFrame, window: int = 120,
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

def analyze_turbulence(frames: Dict[str, pd.DataFrame], period: str, draw=False,
                       balanced=False, include_intraday_on_daily=False):
    t_bull  = mahal_turbulence_from_df(frames["bull"],   252)
    t_bullv = mahal_turbulence_from_df(frames["bullv"],  252)
    t_bear  = mahal_turbulence_from_df(frames["bearish"],252)
    t_intra = mahal_turbulence_intraday(frames["intra"], window=120, min_assets=3, min_obs=20)

    bs, be   = BULL_STABLE_PERIODS[period]
    bv_s, bv_e = BULL_VOLATILE_PERIODS[period]
    br_s, br_e = BEARISH_PERIODS[period]
    is_, ie  = INTRA_PERIODS[period]

    t_bull  = t_bull.loc[(t_bull.index  >= pd.Timestamp(bs))  & (t_bull.index  <= pd.Timestamp(be))]
    t_bullv = t_bullv.loc[(t_bullv.index>= pd.Timestamp(bv_s))& (t_bullv.index<= pd.Timestamp(bv_e))]
    t_bear  = t_bear.loc[(t_bear.index >= pd.Timestamp(br_s)) & (t_bear.index <= pd.Timestamp(br_e))]
    t_intra = t_intra.loc[(t_intra.index>= pd.Timestamp(is_)) & (t_intra.index<= pd.Timestamp(ie))]

    def stats(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return pd.Series({"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan, "N": 0})
        return pd.Series({"mean": s.mean(), "median": s.median(), "p95": s.quantile(0.95), "max": s.max(), "N": len(s)})

    out = pd.concat({
        REG_LABELS["bull"]:   stats(t_bull),
        REG_LABELS["bullv"]:  stats(t_bullv),
        REG_LABELS["bearish"]:stats(t_bear),
        REG_LABELS["intra"]:  stats(t_intra)
    }, axis=1).T.round(3)

    print("\n=== Turbulence index statistics ===")
    print(out.to_string())

    if draw and min(len(t_bull.dropna()), len(t_bullv.dropna()), len(t_bear.dropna())) > 1:
        T = titles(period)
        kde_plot_many(
            [t_bull.dropna(), t_bullv.dropna(), t_bear.dropna()] + ([t_intra.dropna()] if include_intraday_on_daily else []),
            [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else []),
            T["turb_daily"], f"{period}_kde_turb_daily.png", "Turbulence index", False
        )
        F = _get_intra_frames_all(True)
        A = mahal_turbulence_intraday(F["train"], window=120, min_assets=3, min_obs=20)
        B = mahal_turbulence_intraday(F["val"],   window=120, min_assets=3, min_obs=20)
        C = mahal_turbulence_intraday(F["test"],  window=120, min_assets=3, min_obs=20)

        def _mask_intra_to_period(s: pd.Series, period_key: str) -> pd.Series:
            is2, ie2 = INTRA_PERIODS[period_key]
            idx_lo, idx_hi = pd.Timestamp(is2), pd.Timestamp(ie2)
            return s.loc[(s.index >= idx_lo) & (s.index <= idx_hi)]

        A = _mask_intra_to_period(A, "train").dropna()
        B = _mask_intra_to_period(B, "val").dropna()
        C = _mask_intra_to_period(C, "test").dropna()

        if min(len(A), len(B), len(C)) > 1:
            kde_plot_three(A, B, C, "Turbulence index density (Intraday, Train/Val/Test)", "intra_kde_turb_all.png", "Turbulence index", False)


def compute_bollinger_rates(frames: Dict[str, pd.DataFrame]):
    outs = {}
    for key in REG_KEYS:
        df = frames[key].sort_values(["tic", "timestamp"]).copy()
        ss = Sdf.retype(df.copy())
        df["boll_ub"] = ss["boll_ub"]
        df["boll_lb"] = ss["boll_lb"]
        df["up"] = df["close"] > df["boll_ub"]
        df["low"] = df["close"] < df["boll_lb"]
        valid = np.minimum(df["boll_ub"].notna(), df["boll_lb"].notna())
        n = int(valid.sum())
        ru = (df["up"] & valid).sum() / n if n else np.nan
        rl = (df["low"] & valid).sum() / n if n else np.nan
        outs[key] = (ru, rl)
    return outs

def plot_bollinger_rates_all(include_intraday_on_daily=False):
    periods = ["train", "val", "test"]
    bull_u, bullv_u, bear_u, intra_u = [], [], [], []
    bull_l, bullv_l, bear_l, intra_l = [], [], [], []
    for p in periods:
        frames = get_period_frames(p)
        r = compute_bollinger_rates(frames)
        bull_u.append(r["bull"][0]); bullv_u.append(r["bullv"][0]); bear_u.append(r["bearish"][0])
        bull_l.append(r["bull"][1]); bullv_l.append(r["bullv"][1]); bear_l.append(r["bearish"][1])
        intra_u.append(r["intra"][0]); intra_l.append(r["intra"][1])
    bars_daily = [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else [])
    vals_upper = [bull_u, bullv_u, bear_u] + ([intra_u] if include_intraday_on_daily else [])
    vals_lower = [bull_l, bullv_l, bear_l] + ([intra_l] if include_intraday_on_daily else [])
    bar_plot_grouped(groups=periods, bars=bars_daily, values=vals_upper, title="Upper-band breakout rate", filename="boll_upper_rate_all.png", ylabel="Share of observations", percent=True)
    bar_plot_grouped(groups=periods, bars=bars_daily, values=vals_lower, title="Lower-band breakout rate", filename="boll_lower_rate_all.png", ylabel="Share of observations", percent=True)

def compute_avg_corr(frames: Dict[str, pd.DataFrame]):
    def avg(df):
        x = df.copy()
        x["log_ret"] = _log_ret(x)
        p = x.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        if p.empty:
            return np.nan
        c = p.corr()
        return c.values[np.triu_indices_from(c, 1)].mean()
    return dict(bull=avg(frames["bull"]), bullv=avg(frames["bullv"]), bearish=avg(frames["bearish"]), intra=avg(frames["intra"]))

def plot_avg_corr_all(include_intraday_on_daily=False):
    periods = ["train", "val", "test"]
    bull, bullv, bear, intra = [], [], [], []
    for p in periods:
        frames = get_period_frames(p)
        vals = compute_avg_corr(frames)
        bull.append(vals["bull"]); bullv.append(vals["bullv"]); bear.append(vals["bearish"]); intra.append(vals["intra"])
    bars = [REG_LABELS["bull"], REG_LABELS["bullv"], REG_LABELS["bearish"]] + ([REG_LABELS["intra"]] if include_intraday_on_daily else [])
    series = [bull, bullv, bear] + ([intra] if include_intraday_on_daily else [])
    bar_plot_grouped(groups=periods, bars=bars, values=series, title="Average pair-wise correlation", filename="avg_corr_all.png", ylabel="Correlation", percent=False)

ANALYSIS_FUNCS = {
    "log_ret": analyze_log_returns,
    "volatility": analyze_volatility,
    "rsi": analyze_rsi_signals,
    "boll": analyze_bollinger_behavior,
    "volume": analyze_volume,
    "corr": analyze_correlation,
    "turb": analyze_turbulence,
}

def parse_args():
    p = argparse.ArgumentParser(description="Statistical analysis for DJIA across Bullish Stable/Volatile/Bearish/Intraday.")
    p.add_argument("--mode", choices=list(ANALYSIS_FUNCS)+["all"], default="all")
    p.add_argument("--period", choices=["train","val","test"], default="train")
    p.add_argument("--draw", type=str, choices=["true","false"], default="false")
    p.add_argument("--balanced-kde", type=str, choices=["true","false"], default="false")
    p.add_argument("--ci", type=str, choices=["true","false"], default="true")
    p.add_argument("--include-intraday-on-daily", type=str, choices=["true","false"], default="false")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    frames = get_period_frames(args.period)
    frames_buf = get_period_frames_with_buffer(args.period)
    draw_flag = True if args.mode == "all" else (args.draw.lower() == "true")
    balanced = args.balanced_kde.lower() == "true"
    ci_flag = args.ci.lower() == "true"
    incl_intra = args.include_intraday_on_daily.lower() == "true"

    if args.mode == "all":
        ANALYSIS_FUNCS["log_ret"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, incl_intra)
        ANALYSIS_FUNCS["volatility"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, incl_intra)
        ANALYSIS_FUNCS["rsi"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, ci_flag, incl_intra)
        ANALYSIS_FUNCS["boll"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, ci_flag, incl_intra)
        ANALYSIS_FUNCS["volume"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, incl_intra)
        ANALYSIS_FUNCS["corr"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        ANALYSIS_FUNCS["turb"](frames_buf, args.period, draw_flag, balanced, incl_intra)
    else:
        fn = ANALYSIS_FUNCS[args.mode]
        if args.mode in {"log_ret", "volatility", "volume"}:
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, incl_intra)
        elif args.mode == "rsi":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, ci_flag, incl_intra)
        elif args.mode == "boll":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, ci_flag, incl_intra)
        elif args.mode == "corr":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        elif args.mode == "turb":
            fn(frames_buf, args.period, draw_flag, balanced, incl_intra)

    if draw_flag:
        if args.mode in {"all", "boll"}:
            try:
                plot_bollinger_rates_all(incl_intra)
            except Exception as e:
                print(f"[warn] plot_bollinger_rates_all failed: {e}")
        if args.mode in {"all", "corr"}:
            try:
                plot_avg_corr_all(incl_intra)
            except Exception as e:
                print(f"[warn] plot_avg_corr_all failed: {e}")

