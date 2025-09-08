import os, json, shutil, datetime, argparse
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np_test import DailyTradingTestEnv
from finrl.meta.env_stock_trading.env_stocktrading_np_train import DailyTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_train import IntradayTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_test  import IntradayTradingTestEnv
from finrl.train import train
from finrl.test  import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    INTRA_TRAIN_START, INTRA_TRAIN_END,
    INTRA_TEST_START, INTRA_TEST_END,
    CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE,
    CRISIS_TEST_START_DATE,  CRISIS_TEST_END_DATE,
    BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE,
    BEAR2008_TEST_START_DATE,  BEAR2008_TEST_END_DATE,
)

"""
start with
python -m finrl.batch_runs --runs 20 --model ppo  --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model ddpg --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model sac  --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model td3  --task trading --freq daily --dataset stable --note batch

end with
python -m finrl.batch_runs --task trading --freq daily --dataset stable --note batch --compare_agents ppo,ddpg,sac,td3 --aggregate_only
"""

ERL_FIXED_PARAMS = {
    "learning_rate":    3.950916564838231e-05,
    "batch_size":       1024,
    "gamma":            0.9646752384611725,
    "net_dimension":    512,
    "target_step":      8192,
    "horizon_len":      2048,
    "repeat_times":     2.0,
    "buffer_size":      int(1e6),
    "buffer_init_size": 2048,
    "eval_gap":         64,
    "eval_times":       16,
    "if_use_per":       False,
    "seed":             312,
}

BREAK_STEP = 3_00_000
#TOP_K      = 20

best_runs: list[tuple[float, float, str]] = []

LOG_COLUMNS = [
    "timestamp", "series", "trial_id", "model",
    "return", "sharpe", "cagr", "agent_vs_bnh",
    "ann_vol", "bnh_ann_vol", "max_dd", "bnh_max_dd",
    "params_json"
]

def init_log(series_tag: str, log_file: str) -> None:
    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=LOG_COLUMNS)
        df.to_excel(log_file, index=False)
    with open(log_file.replace(".xlsx", ".tag"), "a") as tagfile:
        tagfile.write(f"# NEW_SERIES {series_tag}\n")

def append_log(row: dict, log_file: str) -> None:
    df_new = pd.DataFrame([row])
    if os.path.isfile(log_file):
        with pd.ExcelWriter(log_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            sheet = writer.sheets["Sheet1"]
            start_row = sheet.max_row
            df_new.to_excel(writer, index=False, header=False, startrow=start_row)
    else:
        df_new.to_excel(log_file, index=False)

def _pick_env_and_dates(task: str, freq: str, dataset: str):
    if freq == "intraday":
        interval = "1m"
        if dataset != "intraday":
            raise ValueError("Use 'intraday'")
        s_train, e_train = INTRA_TRAIN_START, INTRA_TRAIN_END
        s_test,  e_test  = INTRA_TEST_START,  INTRA_TEST_END
    else:
        interval = "1d"
        if dataset == "stable":
            s_train, e_train = TRAIN_START_DATE, TRAIN_END_DATE
            s_test, e_test = TEST_START_DATE, TEST_END_DATE
        elif dataset == "crisis":
            s_train, e_train = CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE
            s_test, e_test = CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE
        elif dataset == "bear":
            s_train, e_train = BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE
            s_test, e_test = BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE
        else:
            raise ValueError("Use 'stable' or 'crisis' or 'bear'")
    if task == "trading":
        if freq == "intraday":
            env_train, env_test = IntradayTradingTrainEnv, IntradayTradingTestEnv
            ep_len, warmup = 390, 60
        else:
            env_train, env_test = DailyTradingTrainEnv, DailyTradingTestEnv
            ep_len, warmup = 252, 20
        env_extra_train = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=True)
        env_extra_test  = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=False)
    elif task == "allocation":
        from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
        env_train = env_test = StockPortfolioEnv
        if freq == "intraday":
            lookback, ep_len, warmup, rebal, rew_scale = 60, 390, 60, 5, (2**-9)
        else:
            lookback, ep_len, warmup, rebal, rew_scale = 252, 252, 20, 1, 1.0
        env_extra_train = dict(
            lookback=lookback,
            ep_len=ep_len,
            warmup_lookback=warmup,
            rebalance_every=rebal,
            transaction_cost_pct=1e-3,
            reward_scaling=rew_scale,
            if_train=True,
        )
        env_extra_test = {**env_extra_train, "if_train": False}
    else:
        raise ValueError("task ∈ {'trading','allocation'}")
    return env_train, env_test, interval, s_train, e_train, s_test, e_test, env_extra_train, env_extra_test

def run_once(run_idx: int, series_dir: str, model_name: str, series_tag: str,
             task: str, freq: str, dataset: str, log_file: str):
    erl_params = ERL_FIXED_PARAMS.copy()
    erl_params["seed"] += run_idx
    env_train, env_test, interval, s_train, e_train, s_test, e_test, env_extra_train, env_extra_test = \
        _pick_env_and_dates(task, freq, dataset)
    cwd = os.path.join(series_dir, f"trial_{run_idx:03d}")
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({
            "erl": erl_params,
            "task": task, "freq": freq, "dataset": dataset,
            "interval": interval,
            "dates": dict(train=(s_train, e_train), test=(s_test, e_test)),
            "series": series_tag
        }, f, indent=2)

    tickers = _filter_tickers_intersection(
        DOW_30_TICKER, "yahoofinance", interval, s_train, e_train, s_test, e_test
    )

    train(
        start_date=s_train, end_date=e_train,
        ticker_list=tickers, data_source="yahoofinance",
        time_interval=interval, technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=env_train, model_name=model_name,
        cwd=cwd, erl_params=erl_params, break_step=BREAK_STEP,
        env_extra=env_extra_train,
    )
    assets, sharpe, cagr, agent_vs_bnh, ann_vol, bnh_ann_vol, max_dd, bnh_max_dd = test(
        start_date=s_test, end_date=e_test,
        ticker_list=tickers, data_source="yahoofinance",
        time_interval=interval, technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=env_test, model_name=model_name,
        cwd=cwd, net_dimension=erl_params["net_dimension"],
        env_extra=env_extra_test,
    )
    ret = assets[-1] / assets[0] - 1
    append_log({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "series":    series_tag,
        "trial_id":  f"trial_{run_idx:03d}",
        "model":     model_name,
        "return":    round(ret, 4),
        "sharpe":    round(sharpe, 4),
        "cagr":      round(cagr, 4),
        "agent_vs_bnh": round(agent_vs_bnh, 4),
        "ann_vol":   round(ann_vol, 4),
        "bnh_ann_vol": round(bnh_ann_vol, 4),
        "max_dd":    round(max_dd, 4),
        "bnh_max_dd": round(bnh_max_dd, 4),
        "params_json": json.dumps({
            "erl": erl_params,
            "task": task, "freq": freq, "dataset": dataset
        })
    }, log_file)
    """best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)"""
    return ret, sharpe

def _is_intraday(freq: str) -> bool:
    return freq == "intraday"

def _stack_curves(series_dir: str):
    assets, bnhs = [], []
    trials = sorted([d for d in os.listdir(series_dir) if d.startswith("trial_")])
    for t in trials:
        p_assets = os.path.join(series_dir, t, "assets.npy")
        p_bnh    = os.path.join(series_dir, t, "bnh.npy")
        if os.path.isfile(p_assets) and os.path.isfile(p_bnh):
            a = np.load(p_assets)
            b = np.load(p_bnh)
            n = min(len(a), len(b))
            assets.append(a[:n])
            bnhs.append(b[:n])
    if not assets:
        return None, None
    L = min(map(len, assets))
    assets = np.stack([x[:L] for x in assets], axis=0)
    bnh    = np.median(np.stack([x[:L] for x in bnhs], axis=0), axis=0)
    return assets, bnh

def _aggregate_equity(curves: np.ndarray, intraday: bool, start_date: str):
    norm = curves / curves[:, [0]]
    mean = norm.mean(axis=0)
    median = np.median(norm, axis=0)
    p10, p90 = np.percentile(norm, [10, 90], axis=0)
    n = mean.shape[0]
    if intraday:
        x = np.arange(n)
    else:
        x = pd.bdate_range(start=start_date, periods=n)
    return x, mean, median, p10, p90

def _plot_equity_band(curves: np.ndarray, ref_curve: np.ndarray, path: str,
                      intraday: bool, start_date: str):
    x, mean, median, p10, p90 = _aggregate_equity(curves, intraday, start_date)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.fill_between(x, p10, p90, alpha=0.25, label="10–90%")
    ax.plot(x, mean, linewidth=1.8, label="Mean")
    ax.plot(x, median, linewidth=1.2, linestyle="--", label="Median")
    ref = ref_curve / ref_curve[0]
    if len(ref) != len(mean):
        ref = ref[:len(mean)]
    ax.plot(x, ref, linewidth=1.2, linestyle=":", label="Buy & Hold")
    ax.set_xlabel("Step" if intraday else "Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if not intraday:
        ax.set_xlim(x[0], x[-1])
        fig.autofmt_xdate()
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _summarise_metrics_from_curves(curves: np.ndarray, ref_curve: np.ndarray, intraday: bool):
    rows = []
    steps_per_year = 252 if not intraday else 252*390
    for a in curves:
        n = min(len(a), len(ref_curve))
        a = a[:n].astype(float)
        b = ref_curve[:n].astype(float)
        ret = a[-1]/a[0] - 1.0
        rets = np.diff(a)/a[:-1]
        vol = rets.std(ddof=1) * np.sqrt(steps_per_year) if rets.size else np.nan
        cagr = (a[-1]/a[0])**(steps_per_year/max(1, rets.size)) - 1 if rets.size else np.nan
        mdd = (a/np.maximum.accumulate(a) - 1).min()
        bret = b[-1]/b[0]
        alpha = a[-1]/(a[0]*bret) - 1
        rf = 0.0
        sharpe = ((rets.mean() - rf/steps_per_year) / rets.std(ddof=1) * np.sqrt(steps_per_year)) if rets.std(ddof=1) > 0 else np.nan
        rows.append((ret, cagr, vol, mdd, sharpe, alpha))
    df = pd.DataFrame(rows, columns=["EpisodeReturn", "CAGR", "AnnVol", "MaxDD", "Sharpe", "Alpha_vs_BnH"])
    return df

def _filter_tickers_intersection(tickers, data_source, interval, s_train, e_train, s_test, e_test):
    dp = DataProcessor(data_source)
    df_tr = dp.clean_data(dp.download_data(tickers, s_train, e_train, interval))
    df_te = dp.clean_data(dp.download_data(tickers, s_test,  e_test,  interval))
    have_tr = set(df_tr["tic"].unique())
    have_te = set(df_te["tic"].unique())
    keep = [t for t in tickers if t in have_tr and t in have_te]
    return keep

def _print_agg_table(df: pd.DataFrame):
    cols = ["EpisodeReturn","CAGR","AnnVol","MaxDD","Sharpe","Alpha_vs_BnH"]
    mean = df[cols].mean()
    std  = df[cols].std(ddof=0)
    widths = {c: max(len(c), 12) for c in cols}
    header = " | ".join([c.ljust(widths[c]) for c in cols])
    sep = "-+-".join(["-"*widths[c] for c in cols])
    print("\n=== Aggregate metrics (mean ± std) ===")
    print(header)
    print(sep)
    row = " | ".join([f"{mean[c]:.6f} ± {std[c]:.6f}".ljust(widths[c]) for c in cols])
    print(row)

def _series_dir_for(model: str, task: str, freq: str, dataset: str, note: str):
    tag = f"FIXED_{model}_{task}_{freq}_{dataset}_runs_{note}"
    return os.path.join("fixed_runs", tag)

def _agent_mean_curve(series_dir: str, intraday: bool, start_date: str):
    curves, bnh = _stack_curves(series_dir)
    if curves is None:
        return None, None, None
    norm = curves / curves[:, [0]]
    mean = norm.mean(axis=0)
    n = mean.shape[0]
    x = np.arange(n) if intraday else pd.bdate_range(start=start_date, periods=n)
    ref = bnh / bnh[0]
    if len(ref) != n:
        ref = ref[:n]
    return x, mean, ref

def compare_and_plot_agents(agents_csv: str, task: str, freq: str, dataset: str, note: str, start_date_daily: str, out_name: str):
    agents = [a.strip() for a in agents_csv.split(",") if a.strip()]
    intraday = (freq == "intraday")
    xs, means, labels = [], [], []
    ref_any = None
    for a in agents:
        d = _series_dir_for(a, task, freq, dataset, note)
        x, m, ref = _agent_mean_curve(d, intraday, start_date_daily)
        if x is None:
            continue
        xs.append(x); means.append(m); labels.append(a.upper())
        if ref_any is None:
            ref_any = ref
    if not means:
        print("No agents found for comparison.")
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for x, m, lbl in zip(xs, means, labels):
        ax.plot(x, m, linewidth=1.8, label=lbl)
    if ref_any is not None:
        ax.plot(xs[0], ref_any[:len(xs[0])], linewidth=1.2, linestyle="--", label="Buy & Hold")
    ax.set_xlabel("Step" if intraday else "Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if not intraday:
        ax.set_xlim(xs[0][0], xs[0][-1]); fig.autofmt_xdate()
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    out_dir = _series_dir_for(agents[0], task, freq, dataset, note)
    plt.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
    plt.close(fig)

def aggregate_and_plot(series_dir: str, freq: str, start_date_daily: str, out_prefix: str):
    curves, bnh = _stack_curves(series_dir)
    if curves is None:
        print("No trials found for aggregation.")
        return
    intraday = _is_intraday(freq)
    _plot_equity_band(curves, bnh, os.path.join(series_dir, f"{out_prefix}_EquityBand.jpg"),
                      intraday, start_date_daily)
    df = _summarise_metrics_from_curves(curves, bnh, intraday)
    df.to_csv(os.path.join(series_dir, f"{out_prefix}_metrics_all.csv"), index=False)
    desc = df.describe()
    desc.to_csv(os.path.join(series_dir, f"{out_prefix}_metrics_desc.csv"), index=True)
    _print_agg_table(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=10)
    parser.add_argument("--model", default="ppo")
    parser.add_argument("--note",  default="batch")
    parser.add_argument("--task", choices=["trading", "allocation"], default="trading")
    parser.add_argument("--freq", choices=["daily", "intraday"], default="daily")
    parser.add_argument("--dataset", choices=["stable", "crisis", "intraday", "bear"], default="stable")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--compare_agents", type=str, default="")
    args = parser.parse_args()

    LOG_FILE = f"fixed_{args.task}_{args.freq}_{args.dataset}.xlsx"
    series_tag = f"FIXED_{args.model}_{args.task}_{args.freq}_{args.dataset}_runs_{args.note}"
    series_dir = os.path.join("fixed_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)
    if not args.aggregate_only:
        init_log(series_tag, LOG_FILE)
        returns, sharpes = [], []
        for i in range(args.runs):
            r, s = run_once(i, series_dir, args.model, series_tag, args.task, args.freq, args.dataset, LOG_FILE)
            returns.append(r)
            sharpes.append(s)
            print(f"[Run {i:02d}] Return={r:.4f}  Sharpe={s:.4f}")
        print("\n=== Aggregate results ===")
        print(f"avg Return = {sum(returns)/len(returns):.4f}")
        print(f"std Return = {np.std(returns):.4f}")
        print(f"avg Sharpe = {sum(sharpes)/len(sharpes):.4f}")
        print(f"std Sharpe = {np.std(sharpes):.4f}")
        print(f"\n→ Saved to {LOG_FILE}")
    aggregate_and_plot(
        series_dir=series_dir,
        freq=args.freq,
        start_date_daily=TEST_START_DATE,
        out_prefix="series"
    )
    if args.compare_agents:
        compare_and_plot_agents(
            agents_csv=args.compare_agents,
            task=args.task,
            freq=args.freq,
            dataset=args.dataset,
            note=args.note,
            start_date_daily=TEST_START_DATE,
            out_name="Combined_Agents_Equity.jpg"
        )

"""
start with
python -m finrl.batch_runs --runs 20 --model ppo  --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model ddpg --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model sac  --task trading --freq daily --dataset stable --note batch
python -m finrl.batch_runs --runs 20 --model td3  --task trading --freq daily --dataset stable --note batch

end with
python -m finrl.batch_runs --task trading --freq daily --dataset stable --note batch --compare_agents ppo,ddpg,sac,td3 --aggregate_only
"""