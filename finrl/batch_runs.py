import os, json, shutil, datetime, argparse
import numpy as np
import pandas as pd
from openpyxl import load_workbook

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
)

ERL_FIXED_PARAMS = {
    "learning_rate":    8.772259590429399e-04,
    "batch_size":       512,
    "gamma":            0.977038766982085,
    "seed":             312,
    "net_dimension":    512,
    "net_dims":         [256, 256],
    "target_step":      8192,
    "horizon_len":      2048,
    "repeat_times":     3.0,
    "buffer_size":      int(1e6),
    "buffer_init_size": 1024,
    "if_use_per":       False,
    "eval_gap":         64,
    "eval_times":       16,
}

BREAK_STEP = 3_00_000
TOP_K      = 5

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
            s_train, e_train = TRAIN_START_DATE,   TRAIN_END_DATE
            s_test,  e_test  = TEST_START_DATE,    TEST_END_DATE
        elif dataset == "crisis":
            s_train, e_train = CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE
            s_test,  e_test  = CRISIS_TEST_START_DATE,  CRISIS_TEST_END_DATE
        else:
            raise ValueError("Use 'stable' or 'crisis'")

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
            "dates": dict(train=(s_train, e_train), test=(s_test, e_test))
        }, f, indent=2)

    train(
        start_date=s_train, end_date=e_train,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
        time_interval=interval, technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=env_train, model_name=model_name,
        cwd=cwd, erl_params=erl_params, break_step=BREAK_STEP,
        env_extra=env_extra_train,
    )

    assets, sharpe, cagr, agent_vs_bnh, ann_vol, bnh_ann_vol, max_dd, bnh_max_dd = test(
        start_date=s_test, end_date=e_test,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
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

    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    return ret, sharpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=10)
    parser.add_argument("--model", default="ppo")
    parser.add_argument("--note",  default="batch")
    parser.add_argument("--task", choices=["trading", "allocation"], default="trading")
    parser.add_argument("--freq", choices=["daily", "intraday"], default="daily")
    parser.add_argument("--dataset", choices=["stable", "crisis", "intraday"], default="stable")
    args = parser.parse_args()

    LOG_FILE = f"fixed_{args.task}_{args.freq}_{args.dataset}.xlsx"
    series_tag = f"FIXED_{args.model}_{args.task}_{args.freq}_{args.dataset}_{args.runs}runs_{args.note}"
    series_dir = os.path.join("fixed_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)
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
