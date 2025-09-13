"""
python -m finrl.test_fetch_and_preview
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from stockstats import StockDataFrame as Sdf
from matplotlib.ticker import FuncFormatter
from pathlib import Path

FIG_TREND_DIR = Path("finrl/figures/trend")
FIG_TREND_DIR.mkdir(parents=True, exist_ok=True)

from finrl.config_tickers import DOW_30_TICKER
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
    BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE,
    BEAR2008_VALIDATION_START_DATE, BEAR2008_VALIDATION_END_DATE,
    BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE,
)

SPLIT_COLORS = {
    "train": "#6baed6",
    "val": "#fc9272",
    "test": "#74c476",
}
sys.path.append("/home/krucz/Desktop/magistrat/Magistrat")
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30", "boll_ub", "boll_lb"]

periods = [
    ("Bullish Stable train", TRAIN_START_DATE, TRAIN_END_DATE, "1d"),
    ("Bullish Stable val",   VALIDATION_START_DATE, VALIDATION_END_DATE, "1d"),
    ("Bullish Stable test",  TEST_START_DATE, TEST_END_DATE, "1d"),
    ("Bullish Volatile train", CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE, "1d"),
    ("Bullish Volatile val",   CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE, "1d"),
    ("Bullish Volatile test",  CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE, "1d"),
    ("Bearish train", BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE, "1d"),
    ("Bearish val",   BEAR2008_VALIDATION_START_DATE, BEAR2008_VALIDATION_END_DATE, "1d"),
    ("Bearish test",  BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE, "1d"),
    ("Intraday train", INTRA_TRAIN_START, INTRA_TRAIN_END, "1m"),
    ("Intraday val",   INTRA_VAL_START, INTRA_VAL_END, "1m"),
    ("Intraday test",  INTRA_TEST_START, INTRA_TEST_END, "1m"),
]

processor = YahooFinanceProcessor()

def preview(df, label):
    cols = ["open","high","low","close","volume"]
    n = len(df)
    any_nan = int(df.isna().any(axis=1).sum())
    nan_cols = df.isna().sum().sort_values(ascending=False)
    base_nan = df[cols].isna().sum().reindex(cols).fillna(0).astype(int)
    ts_min = pd.to_datetime(df["timestamp"]).min() if "timestamp" in df else None
    ts_max = pd.to_datetime(df["timestamp"]).max() if "timestamp" in df else None
    print(f"\n=== {label} ===")
    print(f"Range: {ts_min} → {ts_max} | Rows: {n} | Rows with any NaN: {any_nan}")
    print("NaN by base cols:", base_nan.to_dict())
    print("Top NaN columns:", nan_cols.head(10).to_dict())
    try:
        by_tic = df.groupby("tic").size().sort_values(ascending=False)
        print("Rows per tic (top):", by_tic.head(10).to_dict())
    except Exception:
        pass
    print(df.head(5))

for label, start_date, end_date, interval in periods:
    try:
        df = processor.download_data(ticker_list=DOW_30_TICKER, start_date=start_date, end_date=end_date, time_interval=interval)
        df = processor.clean_data(df)
        df = processor.add_technical_indicator(df, tech_indicators)
        preview(df, f"{label} [{interval}] {start_date}..{end_date}")
    except Exception as e:
        print(f"\n=== {label} [{interval}] {start_date}..{end_date} ===")
        print("ERROR:", e)

def to_ts(s):
    return pd.to_datetime(s)

from matplotlib.ticker import FuncFormatter

def _fmt_pct(x, pos):
    return f"{x*100:.0f}%"

def _load_split_frames(proc, DOW_30_TICKER, split):
    if split == "train":
        bull_s, bull_e   = TRAIN_START_DATE, TRAIN_END_DATE
        bullv_s, bullv_e = CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE
        bear_s, bear_e   = BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE
        intra_s, intra_e = INTRA_TRAIN_START, INTRA_TRAIN_END
    elif split == "val":
        bull_s, bull_e   = VALIDATION_START_DATE, VALIDATION_END_DATE
        bullv_s, bullv_e = CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE
        bear_s, bear_e   = BEAR2008_VALIDATION_START_DATE, BEAR2008_VALIDATION_END_DATE
        intra_s, intra_e = INTRA_VAL_START, INTRA_VAL_END
    else:
        bull_s, bull_e   = TEST_START_DATE, TEST_END_DATE
        bullv_s, bullv_e = CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE
        bear_s, bear_e   = BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE
        intra_s, intra_e = INTRA_TEST_START, INTRA_TEST_END

    def _dl(s, e, itv):
        d = proc.download_data(ticker_list=DOW_30_TICKER, start_date=s, end_date=e, time_interval=itv)
        d = proc.clean_data(d)
        return d.reset_index(drop=True)

    return {
        "bull":    _dl(bull_s, bull_e, "1d"),
        "bullv":   _dl(bullv_s, bullv_e, "1d"),
        "bearish": _dl(bear_s, bear_e, "1d"),
        "intra":   _dl(intra_s, intra_e, "1m"),
    }

def _bar_inline(vals, labels, title, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(vals))
    ax.bar(x, vals)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.lower())
    path = FIG_TREND_DIR / f"{safe}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[saved] {path}")

def analyze_ma_trend_inline(frames, split):
    def pack(df):
        ss = Sdf.retype(df.copy())
        df["ema50"] = ss["close_50_ema"]
        df["ema200"] = ss["close_200_ema"]
        df["above200"] = df["close"] > df["ema200"]
        df["golden"] = (df["ema50"] > df["ema200"]) & (df["ema50"].shift(1) <= df["ema200"].shift(1))
        df["death"]  = (df["ema50"] < df["ema200"]) & (df["ema50"].shift(1) >= df["ema200"].shift(1))
        valid = df[["ema50","ema200"]].notna().all(axis=1)
        n = int(valid.sum())
        return pd.Series({
            "p_above_200": float(df.loc[valid, "above200"].mean()) if n else np.nan,
            "golden_cross": int(df.loc[valid, "golden"].sum()),
            "death_cross":  int(df.loc[valid, "death"].sum()),
            "N": n
        })
    out = pd.concat({
        "Bullish Stable":   pack(frames["bull"]),
        "Bullish Volatile": pack(frames["bullv"]),
        "Bearish":          pack(frames["bearish"]),
        "Intraday":         pack(frames["intra"])
    }, axis=1).T.round(4)
    print(f"\n=== EMA trend proxies (50/200), {split} ===")
    print(out.to_string())
    vals = [
        float(out.loc["Bullish Stable","p_above_200"]),
        float(out.loc["Bullish Volatile","p_above_200"]),
        float(out.loc["Bearish","p_above_200"]),
        float(out.loc["Intraday","p_above_200"]),
    ]
    _bar_inline(vals, ["Bullish Stable","Bullish Volatile","Bearish","Intraday"], f"Share of price>EMA200 ({split})", "Share", percent=True)
    return out

def analyze_macd_inline(frames, split):
    def pack(df):
        ss = Sdf.retype(df.copy())
        df["macd"]  = ss["macd"]
        df["macds"] = ss["macds"]
        valid = df[["macd","macds"]].notna().all(axis=1)
        n = int(valid.sum())
        bull_share = float((df.loc[valid, "macd"] > df.loc[valid, "macds"]).mean()) if n else np.nan
        cross_up = int(((df["macd"] > df["macds"]) & (df["macd"].shift(1) <= df["macds"].shift(1)) & valid).sum())
        cross_dn = int(((df["macd"] < df["macds"]) & (df["macd"].shift(1) >= df["macds"].shift(1)) & valid).sum())
        return pd.Series({"p_macd>signal": bull_share, "cross_up": cross_up, "cross_dn": cross_dn, "N": n})
    out = pd.concat({
        "Bullish Stable":   pack(frames["bull"]),
        "Bullish Volatile": pack(frames["bullv"]),
        "Bearish":          pack(frames["bearish"]),
        "Intraday":         pack(frames["intra"])
    }, axis=1).T.round(4)
    print(f"\n=== MACD signals, {split} ===")
    print(out.to_string())
    vals = [
        float(out.loc["Bullish Stable","p_macd>signal"]),
        float(out.loc["Bullish Volatile","p_macd>signal"]),
        float(out.loc["Bearish","p_macd>signal"]),
        float(out.loc["Intraday","p_macd>signal"]),
    ]
    _bar_inline(vals, ["Bullish Stable","Bullish Volatile","Bearish","Intraday"], f"Share of MACD>signal ({split})", "Share", percent=True)
    return out

def analyze_adx_inline(frames, split):
    ADX_STRONG = 25.0

    def pack(df):
        ss = Sdf.retype(df.copy())

        adx = ss["adx"]
        pdi = ss["pdi"]
        ndi = ss["ndi"]

        df = df.copy()
        df["adx"] = pd.to_numeric(adx, errors="coerce")
        df["+di"] = pd.to_numeric(pdi, errors="coerce")
        df["-di"] = pd.to_numeric(ndi, errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid = df[["adx", "+di", "-di"]].notna().all(axis=1)
        n = int(valid.sum())

        strong_mask = valid & (df["adx"] > ADX_STRONG)
        n_strong = int(strong_mask.sum())

        p_adx = float((df.loc[valid, "adx"] > ADX_STRONG).mean()) if n else np.nan
        p_trend_up_strong = float((df.loc[strong_mask, "+di"] > df.loc[strong_mask, "-di"]).mean()) if n_strong else np.nan
        p_trend_dn_strong = float((df.loc[strong_mask, "+di"] < df.loc[strong_mask, "-di"]).mean()) if n_strong else np.nan

        return pd.Series({
            "p_adx>25": p_adx,
            "p_trend_up_strong": p_trend_up_strong,
            "p_trend_dn_strong": p_trend_dn_strong,
            "N": n,
            "N_strong": n_strong,
        })

    out = pd.concat({
        "Bullish Stable":   pack(frames["bull"]),
        "Bullish Volatile": pack(frames["bullv"]),
        "Bearish":          pack(frames["bearish"]),
        "Intraday":         pack(frames["intra"])
    }, axis=1).T.round(4)

    print(f"\n=== ADX trend strength, {split} ===")
    print(out.to_string())

    vals = [
        float(out.loc["Bullish Stable","p_adx>25"]),
        float(out.loc["Bullish Volatile","p_adx>25"]),
        float(out.loc["Bearish","p_adx>25"]),
        float(out.loc["Intraday","p_adx>25"]),
    ]
    _bar_inline(vals,
                ["Bullish Stable","Bullish Volatile","Bearish","Intraday"],
                f"Share of ADX>25 ({split})",
                "Share",
                percent=True)
    return out

def analyze_trend_slope_inline(frames, split, window=60):
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
    out = pd.concat({
        "Bullish Stable":   pack(frames["bull"]),
        "Bullish Volatile": pack(frames["bullv"]),
        "Bearish":          pack(frames["bearish"]),
        "Intraday":         pack(frames["intra"])
    }, axis=1).T.round(6)
    print(f"\n=== Log-price slope (window={window}), {split} ===")
    print(out.to_string())
    vals = [
        float(out.loc["Bullish Stable","p_slope>0"]),
        float(out.loc["Bullish Volatile","p_slope>0"]),
        float(out.loc["Bearish","p_slope>0"]),
        float(out.loc["Intraday","p_slope>0"]),
    ]
    _bar_inline(vals, ["Bullish Stable","Bullish Volatile","Bearish","Intraday"], f"Share of positive slope ({split})", "Share", percent=True)
    return out

LABEL_MAP = {"train": "Train", "val": "Validation", "test": "Test"}

def _bar_grouped_splits(groups, bars, values, title, ylabel, percent=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    width = 0.22
    gap = 0.06
    step = len(bars) * width + (len(bars) - 1) * gap + 0.35
    x0 = np.arange(len(groups)) * step

    for i, bar in enumerate(bars):
        xi = x0 + i * (width + gap)
        color = SPLIT_COLORS.get(bar.lower())
        ax.bar(xi, values[i], width=width, label=LABEL_MAP.get(bar.lower(), bar.title()), color=color)

    ax.set_xticks(x0 + (len(bars) * width + (len(bars) - 1) * gap) / 2 - width / 2, groups)
    ax.set_title(title)                 # np. "Share of positive slope"
    ax.set_ylabel(ylabel)               # "Share"
    ax.margins(y=0.2)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    path = FIG_TREND_DIR / "slope_share_all.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_slope_share_all(processor):
    splits = ["train", "val", "test"]
    regimes = ["Bullish Stable", "Bullish Volatile", "Bearish", "Intraday"]

    per_regime = {r: [] for r in regimes}
    for split in splits:
        frames = _load_split_frames(processor, DOW_30_TICKER, split)
        out = analyze_trend_slope_inline(frames, split, window=60)
        for r in regimes:
            per_regime[r].append(float(out.loc[r, "p_slope>0"]))

    bars = ["train", "val", "test"]
    values = [[per_regime[r][j] for r in regimes] for j in range(len(bars))]

    _bar_grouped_splits(
        groups=regimes,
        bars=bars,
        values=values,
        title="Share of positive slope",
        ylabel="Share",
        percent=True,
    )
    return bars


# DJIA background figure
overall_start = min(
    to_ts(BEAR2008_TRAIN_START_DATE),
    to_ts(TRAIN_START_DATE),
    to_ts(CRISIS_TRAIN_START_DATE),
    to_ts(INTRA_TRAIN_START),
)
overall_end = max(
    to_ts(TEST_END_DATE),
    to_ts(CRISIS_TEST_END_DATE),
    to_ts(BEAR2008_TEST_END_DATE),
    to_ts(INTRA_TEST_END),
)

"""panel = processor.download_data(
    ticker_list=DOW_30_TICKER,
    start_date=overall_start.strftime("%Y-%m-%d"),
    end_date=overall_end.strftime("%Y-%m-%d"),
    time_interval="1d"
)
panel = processor.clean_data(panel).copy()
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values(["timestamp","tic"])

first_close = panel.dropna(subset=["close"]).groupby("tic")["close"].transform("first")
panel["norm_close"] = panel["close"] / first_close

bnh = panel.groupby("timestamp")["norm_close"].agg(["mean","count"]).reset_index()
bnh = bnh.rename(columns={"mean":"eqw_bnh","count":"n_constituents"})
bnh = bnh[bnh["n_constituents"] >= 20]

fig = plt.figure(figsize=(12, 4.5))
ax = plt.gca()
ax.plot(bnh["timestamp"], bnh["eqw_bnh"], linewidth=1.2, label="DJIA") # DOW 30 not DJIA
"""
djia = processor.download_data(
    ticker_list=["^DJI"],
    start_date=overall_start.strftime("%Y-%m-%d"),
    end_date=overall_end.strftime("%Y-%m-%d"),
    time_interval="1d"
)
djia = processor.clean_data(djia).copy()
djia["timestamp"] = pd.to_datetime(djia["timestamp"])
djia = djia.sort_values("timestamp")

fig = plt.figure(figsize=(12, 4.5))
ax = plt.gca()
ax.plot(djia["timestamp"], djia["close"], linewidth=1.2, label="DJIA")

big_blocks = [
    ("Bullish Stable",   to_ts(TRAIN_START_DATE),          to_ts(TEST_END_DATE),            0.07, "lightgreen"),
    ("Bullish Volatile", to_ts(CRISIS_TRAIN_START_DATE),   to_ts(CRISIS_TEST_END_DATE),     0.10, "lightcoral"),
    ("Bearish",          to_ts(BEAR2008_TRAIN_START_DATE), to_ts(BEAR2008_TEST_END_DATE),   0.08, "lightblue"),
    ("Intraday",         to_ts(INTRA_TRAIN_START),         to_ts(INTRA_TEST_END),           0.10, "lightcoral"),
]

for label, s, e, alpha, color in big_blocks:
    ax.axvspan(s, e, alpha=alpha, color=color)
    mid = s + (e - s) / 2
    ax.annotate(label, xy=(mid, 0.98), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=11)

sub_blocks = [
    ("Train", to_ts(TRAIN_START_DATE), to_ts(TRAIN_END_DATE)),
    ("Val",   to_ts(VALIDATION_START_DATE), to_ts(VALIDATION_END_DATE)),
    ("Test",  to_ts(TEST_START_DATE), to_ts(TEST_END_DATE)),
    ("Train", to_ts(CRISIS_TRAIN_START_DATE), to_ts(CRISIS_TRAIN_END_DATE)),
    ("Val",   to_ts(CRISIS_VALIDATION_START_DATE), to_ts(CRISIS_VALIDATION_END_DATE)),
    ("Test",  to_ts(CRISIS_TEST_START_DATE), to_ts(CRISIS_TEST_END_DATE)),
    ("Train", to_ts(BEAR2008_TRAIN_START_DATE), to_ts(BEAR2008_TRAIN_END_DATE)),
    ("Val",   to_ts(BEAR2008_VALIDATION_START_DATE), to_ts(BEAR2008_VALIDATION_END_DATE)),
    ("Test",  to_ts(BEAR2008_TEST_START_DATE), to_ts(BEAR2008_TEST_END_DATE)),
]

for label, s, e in sub_blocks:
    ax.axvspan(s, e, alpha=0.10, color="gray")
    mid = s + (e - s) / 2
    ax.annotate(label, xy=(mid, 0.92), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8)
    ax.axvline(e, linestyle="--", linewidth=0.8, color="k", alpha=0.5)

ax.set_title("Dataset splits presented on DJIA index")
ax.set_ylabel("Index value")
ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=9, frameon=False)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))
ax.grid(True, linewidth=0.4, alpha=0.5)
fig.tight_layout()
plt.savefig(FIG_TREND_DIR / "djia_splits.png", dpi=200)
plt.close(fig)
print(f"[saved] {FIG_TREND_DIR / 'djia_splits.png'}")


# Run the analytics for each split
for split in ["train","val","test"]:
    try:
        frames = _load_split_frames(processor, DOW_30_TICKER, split)
        analyze_ma_trend_inline(frames, split)
        analyze_macd_inline(frames, split)
        analyze_adx_inline(frames, split)
        analyze_trend_slope_inline(frames, split, window=60)
    except Exception as e:
        print(f"[trend] {split} failed:", e)

try:
    plot_slope_share_all(processor)
except Exception as e:
    print("[trend] slope_share_all failed:", e)
