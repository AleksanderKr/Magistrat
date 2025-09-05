#!/usr/bin/env python
"""
Examples
--------
python -m finrl.analyze_dow30_v2 --period train --mode all
python -m finrl.analyze_dow30_v2 --period train --mode all --draw true
"""

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

REG_KEYS = ["bull", "bear", "intra"]
REG_LABELS = dict(bull="`Stable`", bear="Volatile", intra="Intraday")
REG_COLORS = dict(bull="#1f77b4", bear="#d62728", intra="#2ca02c")

BULL_PERIODS = dict(
    train=(TRAIN_START_DATE, TRAIN_END_DATE),
    val=(VALIDATION_START_DATE, VALIDATION_END_DATE),
    test=(TEST_START_DATE, TEST_END_DATE),
)
BEAR_PERIODS = dict(
    train=(CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE),
    val=(CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE),
    test=(CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE),
)
INTRA_PERIODS = dict(
    train=(INTRA_TRAIN_START, INTRA_TRAIN_END),
    val=(INTRA_VAL_START, INTRA_VAL_END),
    test=(INTRA_TEST_START, INTRA_TEST_END),
)
INTRA3_COLORS = dict(train="#2ca02c", val="#9467bd", test="#8c564b")

# ----------------------------- helpers --------------------------------- #
def _period_name(p: str) -> str:
    return dict(train="Train", val="Validation", test="Test")[p]

def legend_labels(period: str) -> Tuple[str, str]:
    bs, be = BULL_PERIODS[period]
    cs, ce = BEAR_PERIODS[period]
    return (f"Stable",
            f"Volatile")

def legend_label_intra(period: str) -> str:
    is_, ie = INTRA_PERIODS[period]
    return f"Intraday"

def titles(period: str) -> Dict[str, str]:
    P = _period_name(period)
    return {
        "logret_bb":    f"Log-return density (Stable vs Volatile, {P} set)",
        "logret_intra": f"Log-return density (Intraday, {P} set)",
        "vol_bb":       f"Rolling volatility σ, 30-day window (Daily, {P} set)",
        "vol_intra":    f"Rolling volatility σ, 390-minute window (Intraday, {P} set)",
        "rsi_bb":       f"RSI(30) density (Stable vs Volatile, {P} set)",
        "rsi_intra":    f"RSI(30) density (Intraday, {P} set)",
        "volu_bb":      f"log(1+volume) density (Stable vs Volatile, {P} set)",
        "volu_intra":   f"log(1+volume) density (Intraday, {P} set)",
        "turb_bb":      f"Turbulence index density (Stable vs Volatile, {P} set)",
        "turb_intra":   f"Turbulence index density (Intraday, {P} set)",
        "bar_boll_u":   "Upper-band breakout rate by period",
        "bar_boll_l":   "Lower-band breakout rate by period",
        "bar_corr":     "Average pair-wise correlation by period",
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
    bs, be = BULL_PERIODS[period]
    cs, ce = BEAR_PERIODS[period]
    is_, ie = INTRA_PERIODS[period]
    bull = load_data(DOW_30_TICKER, bs, be, "1d")
    bear = load_data(DOW_30_TICKER, cs, ce, "1d")
    intra = load_data(DOW_30_TICKER, is_, ie, "1m")
    return {"bull": bull, "bear": bear, "intra": intra}

def get_period_frames_with_buffer(period: str) -> Dict[str, pd.DataFrame]:
    bs, be = BULL_PERIODS[period]
    cs, ce = BEAR_PERIODS[period]
    is_, ie = INTRA_PERIODS[period]
    bs_buf = (pd.Timestamp(bs) - pd.Timedelta(days=TURB_DAILY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    cs_buf = (pd.Timestamp(cs) - pd.Timedelta(days=TURB_DAILY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    is_buf = (pd.Timestamp(is_) - pd.Timedelta(days=TURB_INTRADAY_BUFFER_DAYS)).strftime("%Y-%m-%d")
    bull = load_data(DOW_30_TICKER, bs_buf, be, "1d")
    bear = load_data(DOW_30_TICKER, cs_buf, ce, "1d")
    intra = load_data(DOW_30_TICKER, is_buf, ie, "1m")
    return {"bull": bull, "bear": bear, "intra": intra}

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

def _mask_intra_to_period(s: pd.Series, period_key: str) -> pd.Series:
    is_, ie = INTRA_PERIODS[period_key]
    idx_lo, idx_hi = pd.Timestamp(is_), pd.Timestamp(ie)
    return s.loc[(s.index >= idx_lo) & (s.index <= idx_hi)]

def build_intra_logret_all() -> Tuple[pd.Series, pd.Series, pd.Series]:
    F = _get_intra_frames_all(False)
    A = _log_ret(F["train"]).dropna()
    B = _log_ret(F["val"]).dropna()
    C = _log_ret(F["test"]).dropna()
    return A, B, C

def build_intra_vol_all() -> Tuple[pd.Series, pd.Series, pd.Series]:
    F = _get_intra_frames_all(False)
    def v(df):
        r = _log_ret(df)
        return df.assign(lr=r).groupby("tic")["lr"].transform(lambda x: x.rolling(390, min_periods=120).std())
    A = v(F["train"]).dropna(); B = v(F["val"]).dropna(); C = v(F["test"]).dropna()
    return A, B, C

def build_intra_rsi_all() -> Tuple[pd.Series, pd.Series, pd.Series]:
    F = _get_intra_frames_all(False)
    def r(df): return Sdf.retype(df.sort_values(["tic","timestamp"]).copy())["rsi_30"]
    A = r(F["train"]).dropna(); B = r(F["val"]).dropna(); C = r(F["test"]).dropna()
    return A, B, C

def build_intra_volume_all() -> Tuple[pd.Series, pd.Series, pd.Series]:
    F = _get_intra_frames_all(False)
    A = np.log1p(F["train"]["volume"]).dropna()
    B = np.log1p(F["val"]["volume"]).dropna()
    C = np.log1p(F["test"]["volume"]).dropna()
    return A, B, C

def build_intra_turb_all() -> Tuple[pd.Series, pd.Series, pd.Series]:
    F = _get_intra_frames_all(True)
    A = mahal_turbulence_from_df(F["train"], 390)
    B = mahal_turbulence_from_df(F["val"], 390)
    C = mahal_turbulence_from_df(F["test"], 390)
    A = _mask_intra_to_period(A, "train").dropna()
    B = _mask_intra_to_period(B, "val").dropna()
    C = _mask_intra_to_period(C, "test").dropna()
    return A, B, C
# ----------------------------- helpers --------------------------------- #

def kde_plot_three(a: pd.Series, b: pd.Series, c: pd.Series,
                   title: str, filename: str, xlabel: str, percent: bool=False,
                   labels: Tuple[str, str, str]=("Train", "Validation", "Test")):
    sA = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    sB = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    sC = pd.to_numeric(c, errors="coerce").dropna().astype(float)
    if min(sA.nunique(), sB.nunique(), sC.nunique()) < 2:
        print(f"[warn] Skipping KDE for {filename}: a series is constant.")
        return
    lims = _percentile_limits(sA, sB, sC, p_lo=1.0, p_hi=99.0)
    if not lims:
        print(f"[warn] Skipping KDE for {filename}: invalid domain.")
        return
    xmin, xmax = lims
    xs = _shared_xs(xmin, xmax, 400)
    kA = _safe_kde(sA, xs); kB = _safe_kde(sB, xs); kC = _safe_kde(sC, xs)
    if kA is None or kB is None or kC is None:
        print(f"[warn] Skipping KDE for {filename}: KDE failed.")
        return

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(xs, kA, lw=2.0, label=f"{labels[0]} (N={len(sA)})", color=INTRA3_COLORS["train"], alpha=0.95)
    ax.plot(xs, kB, lw=2.0, ls="--", label=f"{labels[1]} (N={len(sB)})", color=INTRA3_COLORS["val"], alpha=0.95)
    ax.plot(xs, kC, lw=2.0, ls="-.", label=f"{labels[2]} (N={len(sC)})", color=INTRA3_COLORS["test"], alpha=0.95)
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

def kde_plot_two(a: pd.Series, b: pd.Series, title: str, filename: str, xlabel: str,
                 percent: bool=False, balanced: bool=False, labels: Tuple[str,str]=None):
    s1 = pd.to_numeric(a, errors="coerce").dropna()
    s2 = pd.to_numeric(b, errors="coerce").dropna()
    if balanced:
        n = min(len(s1), len(s2))
        if n >= 10:
            s1 = s1.sample(n, random_state=7)
            s2 = s2.sample(n, random_state=7)
    if min(s1.nunique(), s2.nunique()) < 2:
        print(f"[warn] Skipping KDE for {filename}: a series is constant.")
        return
    lims = _percentile_limits(s1, s2, p_lo=1.0, p_hi=99.0)
    if not lims:
        print(f"[warn] Skipping KDE for {filename}: invalid domain.")
        return
    xmin, xmax = lims
    xs = _shared_xs(xmin, xmax, 400)
    k1 = _safe_kde(s1, xs)
    k2 = _safe_kde(s2, xs)
    if k1 is None or k2 is None:
        print(f"[warn] Skipping KDE for {filename}: KDE failed.")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    lab1 = labels[0] if labels else REG_LABELS["bull"]
    lab2 = labels[1] if labels else REG_LABELS["bear"]
    ax.plot(xs, k1, label=f"{lab1} (N={len(s1)})", lw=2.0, color=REG_COLORS["bull"], alpha=0.95)
    ax.plot(xs, k2, label=f"{lab2} (N={len(s2)})", lw=2.0, ls="--", color=REG_COLORS["bear"], alpha=0.95)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="both", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / filename, dpi=200); plt.close(fig)

def kde_plot_one(a: pd.Series, title: str, filename: str, xlabel: str, percent: bool=False):
    s = pd.to_numeric(a, errors="coerce").dropna()
    if s.nunique() < 2:
        print(f"[warn] Skipping KDE for {filename}: a series is constant.")
        return
    lims = _percentile_limits(s, p_lo=1.0, p_hi=99.0)
    if not lims:
        print(f"[warn] Skipping KDE for {filename}: invalid domain.")
        return
    xmin, xmax = lims
    xs = _shared_xs(xmin, xmax, 400)
    k = _safe_kde(s, xs)
    if k is None:
        print(f"[warn] Skipping KDE for {filename}: KDE failed.")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(xs, k, lw=2.0, color=REG_COLORS["intra"], alpha=0.95, label=f"{REG_LABELS['intra']} (N={len(s)})")
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

def bar_plot(values, labels, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(5, 3))
    width = 0.25
    gap = 0.35
    x = np.arange(len(values)) * (width + gap)
    colors = [REG_COLORS["bull"], REG_COLORS["bear"], REG_COLORS["intra"]][:len(values)]
    bars = ax.bar(x, values, width=width, color=colors)
    ymax = np.nanmax(values) if len(values) else 1.0
    if not np.isfinite(ymax):
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(handles=bars, labels=labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

def bar_plot_grouped(groups, bars, values, title, filename, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    width = 0.24
    bar_gap = 0.06
    group_gap = 0.44
    n_groups = len(groups)
    n_bars = len(bars)
    cluster_width = n_bars * width + (n_bars - 1) * bar_gap
    step = cluster_width + group_gap
    x0 = np.arange(n_groups) * step
    palette = [REG_COLORS["bull"], REG_COLORS["bear"], REG_COLORS["intra"]]
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
        ax.bar(xi, values[i], width=width, label=bars[i], color=palette[i % len(palette)])
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xticks(x0 + cluster_width / 2 - width / 2, groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, which="both", axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)

# ----------------------------- analyses -------------------------------- #

def _log_ret(df: pd.DataFrame) -> pd.Series:
    return df.groupby("tic")["close"].transform(lambda x: np.log(x / x.shift(1)))

def analyze_log_returns(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in dfs:
        dfs[k]["log_ret"] = _log_ret(dfs[k])

    stats = lambda s: s.dropna().agg(["mean", "std", "skew", "kurt"])
    out = pd.concat(
        {
            "Stable": stats(dfs["bull"]["log_ret"]),
            "Volatile": stats(dfs["bear"]["log_ret"]),
            "Intraday": stats(dfs["intra"]["log_ret"]),
        }, axis=1
    ).T
    out["N"] = [
        dfs["bull"]["log_ret"].dropna().shape[0],
        dfs["bear"]["log_ret"].dropna().shape[0],
        dfs["intra"]["log_ret"].dropna().shape[0],
    ]
    out = out[["mean","std","skew","kurt","N"]].round(6)
    print("\n=== Log-return summary ===")
    print(out.to_string())

    s1, s2, s3 = (
        dfs["bull"]["log_ret"].dropna(),
        dfs["bear"]["log_ret"].dropna(),
        dfs["intra"]["log_ret"].dropna(),
    )
    if min(len(s1), len(s2), len(s3)) > 2:
        stat, p = fligner(s1, s2, s3, center="median")
        print(f"\nFligner–Killeen test (variances): stat={stat:.3f}  p={p:.4e}")
    else:
        print("\nFligner–Killeen test: skipped (insufficient data)")

    if draw:
        T = titles(period)
        A, B, C = build_intra_logret_all()
        kde_plot_three(
            A, B, C,
            "Log-return density (Intraday, Train/Val/Test)",
            "intra_kde_logret_all.png",
            "Log return",
            True,
            labels=("Train", "Validation", "Test"),
        )
        kde_plot_two(
            s1, s2,
            T["logret_bb"],
            f"{period}_kde_logret.png",
            "Log return",
            True, balanced,
            labels=legend_labels(period),
        )
        kde_plot_one(
            s3,
            T["logret_intra"],
            f"{period}_kde_logret_intra.png",
            "Log return",
            True,
        )

def analyze_volatility(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False):
    dfs = {k: v.copy() for k, v in frames.items()}
    for k in ["bull", "bear"]:
        dfs[k]["lr"] = _log_ret(dfs[k])
        dfs[k]["vol"] = dfs[k].groupby("tic")["lr"].transform(lambda x: x.rolling(30, min_periods=10).std())
        dfs[k]["vol_ann"] = dfs[k]["vol"] * np.sqrt(252)
    dfs["intra"]["lr"] = _log_ret(dfs["intra"])
    dfs["intra"]["vol"] = dfs["intra"].groupby("tic")["lr"].transform(lambda x: x.rolling(390, min_periods=120).std())
    dfs["intra"]["vol_day"] = dfs["intra"]["vol"] * np.sqrt(390)
    dfs["intra"]["vol_ann"] = dfs["intra"]["vol"] * np.sqrt(390 * 252)

    res = pd.DataFrame(
        {
            "mean_vol": [dfs["bull"]["vol"].mean(), dfs["bear"]["vol"].mean(), dfs["intra"]["vol"].mean()],
            "median_vol": [dfs["bull"]["vol"].median(), dfs["bear"]["vol"].median(), dfs["intra"]["vol"].median()],
            "mean_vol_ann": [dfs["bull"]["vol_ann"].mean(), dfs["bear"]["vol_ann"].mean(), dfs["intra"]["vol_ann"].mean()],
            "N": [
                dfs["bull"]["vol"].dropna().shape[0],
                dfs["bear"]["vol"].dropna().shape[0],
                dfs["intra"]["vol"].dropna().shape[0],
            ],
        },
        index=["Stable", "Volatile", "Intraday"],
    ).round(6)
    print("\n=== Rolling volatility (30d daily; 390m intraday) ===")
    print(res.to_string())

    if draw:
        T = titles(period)
        A, B, C = build_intra_vol_all()
        kde_plot_three(
            A, B, C,
            "Rolling volatility σ, 390-minute window (Intraday, Train/Val/Test)",
            "intra_kde_vol_all.png",
            "σ (390-minute, intraday)",
            True,
            labels=("Train", "Validation", "Test"),
        )
        kde_plot_two(
            dfs["bull"]["vol"].dropna(),
            dfs["bear"]["vol"].dropna(),
            T["vol_bb"],
            f"{period}_kde_vol.png",
            "σ (30-day, daily)",
            True, balanced,
            labels=legend_labels(period),
        )
        kde_plot_one(
            dfs["intra"]["vol"].dropna(),
            T["vol_intra"],
            f"{period}_kde_vol_intra.png",
            "σ (390-minute, intraday)",
            True,
        )

def analyze_rsi_signals(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False, ci=False):
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
        {"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])}, axis=1
    ).T
    out.loc[:, ["mean_rsi", "p>70", "p<30", "p>70_CIlo", "p>70_CIhi", "p<30_CIlo", "p<30_CIhi"]] = out.loc[
        :, ["mean_rsi", "p>70", "p<30", "p>70_CIlo", "p>70_CIhi", "p<30_CIlo", "p<30_CIhi"]
    ].astype(float).round(4)
    print("\n=== RSI(30) signal counts and shares ===")
    print(out.to_string())

    if draw:
        T = titles(period)
        A, B, C = build_intra_rsi_all()
        kde_plot_three(
            A, B, C,
            "RSI(30) density (Intraday, Train/Val/Test)",
            "intra_kde_rsi30_all.png",
            "RSI (0–100)",
            False,
            labels=("Train", "Validation", "Test"),
        )
        kde_plot_two(
            dfs["bull"]["rsi_30"].dropna(),
            dfs["bear"]["rsi_30"].dropna(),
            T["rsi_bb"],
            f"{period}_kde_rsi30.png",
            "RSI (0–100)",
            False, balanced,
            labels=legend_labels(period),
        )
        kde_plot_one(
            dfs["intra"]["rsi_30"].dropna(),
            T["rsi_intra"],
            f"{period}_kde_rsi30_intra.png",
            "RSI (0–100)",
            False,
        )

def analyze_bollinger_behavior(frames: Dict[str, pd.DataFrame], period: str, draw=False, ci=False):
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

    out = pd.concat({"Stable": g(dfs["bull"]), "Volatile": g(dfs["bear"]), "Intraday": g(dfs["intra"])}, axis=1).T
    out = out.round(4)
    print("\n=== Bollinger breakouts ===")
    print(out.to_string(index=True))
    """
    if draw:
        groups = ["Stable", "Volatile", "Intraday"]
        vals_u = [out.loc[g, "rate_upper"] for g in groups]
        vals_l = [out.loc[g, "rate_lower"] for g in groups]
        bar_plot_grouped(
            groups=groups,
            bars=["Upper"],
            values=[vals_u],
            title=f"Upper-band breakout rate ({period})",
            filename=f"{period}_boll_upper_rate.png",
            ylabel="Share of observations",
            percent=True,
        )
        bar_plot_grouped(
            groups=groups,
            bars=["Lower"],
            values=[vals_l],
            title=f"Lower-band breakout rate ({period})",
            filename=f"{period}_boll_lower_rate.png",
            ylabel="Share of observations",
            percent=True,
        )
    """
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

    vals = {"Stable": avg(dfs["bull"]), "Volatile": avg(dfs["bear"]), "Intraday": avg(dfs["intra"])}
    out = pd.DataFrame.from_dict(vals, orient="index", columns=["avg_corr"]).round(5)

    print("\n=== Avg pair-wise return correlation ===")
    print(out.to_string())
    """
    if draw and out["avg_corr"].notna().all():
        groups = list(out.index)
        bar_plot_grouped(
            groups=groups,
            bars=["Avg corr"],
            values=[[out.loc[g, "avg_corr"] for g in groups]],
            title=f"Average pair-wise correlation ({period})",
            filename=f"{period}_avg_corr.png",
            ylabel="Correlation",
            percent=False,
        )
    """
    return out

def analyze_volume(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False):
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

    out = pd.concat({"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])}, axis=1).T.round(4)
    print("\n=== Volume stats ===")
    print(out.to_string())

    if draw:
        T = titles(period)
        A, B, C = build_intra_volume_all()
        kde_plot_three(
            A, B, C,
            "log(1+volume) density (Intraday, Train/Val/Test)",
            "intra_kde_volume_all.png",
            "log(1+volume)",
            False,
            labels=("Train", "Validation", "Test"),
        )
        kde_plot_two(
            np.log1p(dfs["bull"]["volume"]),
            np.log1p(dfs["bear"]["volume"]),
            T["volu_bb"],
            f"{period}_kde_volume.png",
            "log(1+volume)",
            False, balanced,
            labels=legend_labels(period),
        )
        kde_plot_one(
            np.log1p(dfs["intra"]["volume"]),
            T["volu_intra"],
            f"{period}_kde_volume_intra.png",
            "log(1+volume)",
            False,
        )

def mahal_turbulence_from_df(df: pd.DataFrame, window: int) -> pd.Series:
    price = df.pivot(index="timestamp", columns="tic", values="close").sort_index()
    ret = price.pct_change()
    if ret.empty:
        return pd.Series(dtype=float)
    tser = pd.Series(np.nan, index=ret.index, dtype=float)
    for i in range(window, len(ret)):
        hist = ret.iloc[i - window : i]
        cur = ret.iloc[i]
        valid = cur.dropna().index
        hist, cur = hist[valid], cur[valid]
        if hist.shape[0] < 2 or len(valid) < 2:
            continue
        cov = hist.cov()
        diff = cur - hist.mean()
        try:
            inv = np.linalg.pinv(cov.values)
            tser.iloc[i] = float(diff.values @ inv @ diff.values.T)
        except Exception:
            continue
    return tser

def analyze_turbulence(frames: Dict[str, pd.DataFrame], period: str, draw=False, balanced=False):
    t_bull = mahal_turbulence_from_df(frames["bull"], 252)
    t_bear = mahal_turbulence_from_df(frames["bear"], 252)
    t_intra = mahal_turbulence_from_df(frames["intra"], 390)

    bs, be = BULL_PERIODS[period]
    cs, ce = BEAR_PERIODS[period]
    is_, ie = INTRA_PERIODS[period]
    t_bull = t_bull.loc[(t_bull.index >= pd.Timestamp(bs)) & (t_bull.index <= pd.Timestamp(be))]
    t_bear = t_bear.loc[(t_bear.index >= pd.Timestamp(cs)) & (t_bear.index <= pd.Timestamp(ce))]
    t_intra = t_intra.loc[(t_intra.index >= pd.Timestamp(is_)) & (t_intra.index <= pd.Timestamp(ie))]

    def stats(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return pd.Series({"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan, "N": 0})
        return pd.Series({"mean": s.mean(), "median": s.median(), "p95": s.quantile(0.95), "max": s.max(), "N": len(s)})

    out = pd.concat({"Stable": stats(t_bull), "Volatile": stats(t_bear), "Intraday": stats(t_intra)}, axis=1).T.round(3)
    print("\n=== Turbulence index statistics ===")
    print(out.to_string())

    if draw and min(len(t_bull.dropna()), len(t_bear.dropna()), len(t_intra.dropna())) > 1:
        T = titles(period)
        A, B, C = build_intra_turb_all()
        kde_plot_three(
            A, B, C,
            "Turbulence index density (Intraday, Train/Val/Test)",
            "intra_kde_turb_all.png",
            "Turbulence index",
            False,
            labels=("Train", "Validation", "Test"),
        )
        kde_plot_two(
            t_bull.dropna(),
            t_bear.dropna(),
            T["turb_bb"],
            f"{period}_kde_turb.png",
            "Turbulence index",
            False, balanced,
            labels=legend_labels(period),
        )
        kde_plot_one(
            t_intra.dropna(),
            T["turb_intra"],
            f"{period}_kde_turb_intra.png",
            "Turbulence index",
            False,
        )
def analyze_ma_trend(frames: Dict[str, pd.DataFrame], period: str, draw=False):
    dfs = {k: df.sort_values(["tic", "timestamp"]).copy() for k, df in frames.items()}
    def pack(df):
        ss = Sdf.retype(df.copy())
        df["ema50"] = ss["close_50_ema"]
        df["ema200"] = ss["close_200_ema"]
        df["above200"] = df["close"] > df["ema200"]
        df["golden"] = (df["ema50"] > df["ema200"]) & (df["ema50"].shift(1) <= df["ema200"].shift(1))
        df["death"] = (df["ema50"] < df["ema200"]) & (df["ema50"].shift(1) >= df["ema200"].shift(1))
        valid = df[["ema50","ema200"]].notna().all(axis=1)
        n = int(valid.sum())
        p_above = float(df.loc[valid, "above200"].mean()) if n else np.nan
        n_gold = int(df.loc[valid, "golden"].sum())
        n_death = int(df.loc[valid, "death"].sum())
        return pd.Series({"p_above_200": p_above, "golden_cross": n_gold, "death_cross": n_death, "N": n})
    out = pd.concat(
        {"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])},
        axis=1
    ).T.round(4)
    print("\n=== EMA trend proxies (50/200) ===")
    print(out.to_string())
    if draw:
        T = titles(period)
        vals = [float(out.loc["Stable","p_above_200"]), float(out.loc["Volatile","p_above_200"]), float(out.loc["Intraday","p_above_200"])]
        labs = ["Stable","Volatile","Intraday"]
        bar_plot(vals, labs, T["bar_ma"], f"{period}_bar_ma.png", titles(period)["y_share"], percent=True)
    return out

def analyze_macd(frames: Dict[str, pd.DataFrame], period: str, draw=False):
    dfs = {k: df.sort_values(["tic", "timestamp"]).copy() for k, df in frames.items()}
    def pack(df):
        ss = Sdf.retype(df.copy())
        df["macd"] = ss["macd"]
        df["macds"] = ss["macds"]
        valid = df[["macd","macds"]].notna().all(axis=1)
        n = int(valid.sum())
        bull_share = float((df.loc[valid, "macd"] > df.loc[valid, "macds"]).mean()) if n else np.nan
        cross_up = int(((df["macd"] > df["macds"]) & (df["macd"].shift(1) <= df["macds"].shift(1)) & valid).sum())
        cross_dn = int(((df["macd"] < df["macds"]) & (df["macd"].shift(1) >= df["macds"].shift(1)) & valid).sum())
        return pd.Series({"p_macd>signal": bull_share, "cross_up": cross_up, "cross_dn": cross_dn, "N": n})
    out = pd.concat(
        {"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])},
        axis=1
    ).T.round(4)
    print("\n=== MACD signals ===")
    print(out.to_string())
    if draw:
        T = titles(period)
        vals = [float(out.loc["Stable","p_macd>signal"]), float(out.loc["Volatile","p_macd>signal"]), float(out.loc["Intraday","p_macd>signal"])]
        labs = ["Stable","Volatile","Intraday"]
        bar_plot(vals, labs, T["bar_macd"], f"{period}_bar_macd.png", titles(period)["y_share"], percent=True)
    return out

def analyze_adx(frames: Dict[str, pd.DataFrame], period: str, draw=False):
    dfs = {k: df.sort_values(["tic", "timestamp"]).copy() for k, df in frames.items()}
    def pack(df):
        ss = Sdf.retype(df.copy())
        df["adx"] = ss["adx"]
        df["+di"] = ss["pdi"]
        df["-di"] = ss["mdi"]
        valid = df["adx"].notna()
        n = int(valid.sum())
        strong = float((df.loc[valid, "adx"] > 25).mean()) if n else np.nan
        up_dir = float(((df.loc[valid, "+di"] > df.loc[valid, "-di"]) & (df.loc[valid, "adx"] > 25)).mean()) if n else np.nan
        dn_dir = float(((df.loc[valid, "+di"] < df.loc[valid, "-di"]) & (df.loc[valid, "adx"] > 25)).mean()) if n else np.nan
        return pd.Series({"p_adx>25": strong, "p_trend_up_strong": up_dir, "p_trend_dn_strong": dn_dir, "N": n})
    out = pd.concat(
        {"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])},
        axis=1
    ).T.round(4)
    print("\n=== ADX trend strength ===")
    print(out.to_string())
    if draw:
        T = titles(period)
        vals = [float(out.loc["Stable","p_adx>25"]), float(out.loc["Volatile","p_adx>25"]), float(out.loc["Intraday","p_adx>25"])]
        labs = ["Stable","Volatile","Intraday"]
        bar_plot(vals, labs, T["bar_adx"], f"{period}_bar_adx.png", titles(period)["y_share"], percent=True)
    return out

def analyze_trend_slope(frames: Dict[str, pd.DataFrame], period: str, window=60, draw=False):
    dfs = {k: df.sort_values(["tic", "timestamp"]).copy() for k, df in frames.items()}
    def slope_win(s):
        y = np.log(s.values)
        x = np.arange(len(y))
        if len(y) < 2:
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]
        return b
    def pack(df):
        r = df.groupby("tic")["close"].rolling(window, min_periods=window//2).apply(slope_win, raw=False).reset_index(name="slope")
        m = float(r["slope"].mean())
        p_pos = float((r["slope"] > 0).mean())
        return pd.Series({"mean_slope": m, "p_slope>0": p_pos})
    out = pd.concat(
        {"Stable": pack(dfs["bull"]), "Volatile": pack(dfs["bear"]), "Intraday": pack(dfs["intra"])},
        axis=1
    ).T.round(6)
    print(f"\n=== Log-price slope (window={window}) ===")
    print(out.to_string())
    if draw:
        T = titles(period)
        vals = [float(out.loc["Stable","p_slope>0"]), float(out.loc["Volatile","p_slope>0"]), float(out.loc["Intraday","p_slope>0"])]
        labs = ["Stable","Volatile","Intraday"]
        bar_plot(vals, labs, T["bar_slope"], f"{period}_bar_slope.png", titles(period)["y_share"], percent=True)
    return out

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

def plot_bollinger_rates_all():
    periods = ["train", "val", "test"]
    bull_u, bear_u, intra_u = [], [], []
    bull_l, bear_l, intra_l = [], [], []
    for p in periods:
        frames = get_period_frames(p)
        r = compute_bollinger_rates(frames)
        bull_u.append(r["bull"][0]); bear_u.append(r["bear"][0]); intra_u.append(r["intra"][0])
        bull_l.append(r["bull"][1]); bear_l.append(r["bear"][1]); intra_l.append(r["intra"][1])
    bars = ["Stable", "Volatile", "Intraday"]
    vals_upper = [bull_u, bear_u, intra_u]
    vals_lower = [bull_l, bear_l, intra_l]
    bar_plot_grouped(
        groups=periods, bars=bars, values=vals_upper,
        title="Upper-band breakout rate", filename="boll_upper_rate_all.png",
        ylabel="Share of observations", percent=True
    )
    bar_plot_grouped(
        groups=periods, bars=bars, values=vals_lower,
        title="Lower-band breakout rate", filename="boll_lower_rate_all.png",
        ylabel="Share of observations", percent=True
    )

def compute_avg_corr(frames: Dict[str, pd.DataFrame]):
    def avg(df):
        x = df.copy()
        x["log_ret"] = _log_ret(x)
        p = x.pivot(index="timestamp", columns="tic", values="log_ret").dropna()
        if p.empty:
            return np.nan
        c = p.corr()
        return c.values[np.triu_indices_from(c, 1)].mean()
    return dict(bull=avg(frames["bull"]), bear=avg(frames["bear"]), intra=avg(frames["intra"]))

def plot_avg_corr_all():
    periods = ["train", "val", "test"]
    bull, bear, intra = [], [], []
    for p in periods:
        frames = get_period_frames(p)
        vals = compute_avg_corr(frames)
        bull.append(vals["bull"]); bear.append(vals["bear"]); intra.append(vals["intra"])
    bars = ["Stable", "Volatile", "Intraday"]
    bar_plot_grouped(
        groups=periods, bars=bars, values=[bull, bear, intra],
        title="Average pair-wise correlation", filename="avg_corr_all.png",
        ylabel="Correlation", percent=False
    )

ANALYSIS_FUNCS = {
    "log_ret": analyze_log_returns,
    "volatility": analyze_volatility,
    "rsi": analyze_rsi_signals,
    "boll": analyze_bollinger_behavior,
    "volume": analyze_volume,
    "corr": analyze_correlation,
    "turb": analyze_turbulence,
    "ma": analyze_ma_trend,
    "macd": analyze_macd,
    "adx": analyze_adx,
    "slope": analyze_trend_slope,
}

def parse_args():
    p = argparse.ArgumentParser(description="Statistical analysis for DJIA across Stable/Volatile/Intraday.")
    p.add_argument("--mode", choices=list(ANALYSIS_FUNCS)+["all"], default="all")
    p.add_argument("--period", choices=["train","val","test"], default="train")
    p.add_argument("--draw", type=str, choices=["true","false"], default="false")
    p.add_argument("--balanced-kde", type=str, choices=["true","false"], default="false")
    p.add_argument("--ci", type=str, choices=["true","false"], default="true")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    frames = get_period_frames(args.period)
    frames_buf = get_period_frames_with_buffer(args.period)
    draw_flag = True if args.mode == "all" else (args.draw.lower() == "true")
    balanced = args.balanced_kde.lower() == "true"
    ci_flag = args.ci.lower() == "true"

    if args.mode == "all":
        ANALYSIS_FUNCS["log_ret"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced)
        ANALYSIS_FUNCS["volatility"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced)
        ANALYSIS_FUNCS["rsi"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, ci_flag)
        ANALYSIS_FUNCS["boll"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, ci_flag)
        ANALYSIS_FUNCS["volume"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced)
        ANALYSIS_FUNCS["corr"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        ANALYSIS_FUNCS["turb"](frames_buf, args.period, draw_flag, balanced)
        ANALYSIS_FUNCS["ma"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        ANALYSIS_FUNCS["macd"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        ANALYSIS_FUNCS["adx"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        ANALYSIS_FUNCS["slope"]({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)

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
        fn = ANALYSIS_FUNCS[args.mode]
        if args.mode in {"log_ret", "volatility", "volume"}:
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced)
        elif args.mode == "rsi":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, balanced, ci_flag)
        elif args.mode == "boll":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag, ci_flag)
        elif args.mode == "corr":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        elif args.mode == "turb":
            fn(frames_buf, args.period, draw_flag, balanced)
        elif args.mode in {"ma", "macd", "adx"}:
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)
        elif args.mode == "slope":
            fn({k: v.copy() for k, v in frames.items()}, args.period, draw_flag)

