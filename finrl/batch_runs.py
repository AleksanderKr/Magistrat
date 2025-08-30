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
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
from finrl.config import INTRA_TRAIN_START, INTRA_TRAIN_END, INTRA_TEST_START, INTRA_TEST_END

"""
python -m finrl.batch_runs --runs 30 --model ppo --note algTrade_dailyBull --envset daily
python -m finrl.batch_runs --runs 30 --model ppo --note algTrade_dailyBear --envset daily
python -m finrl.batch_runs --runs 30 --model ppo --note algTrade_intraday --envset intraday
"""

ERL_FIXED_PARAMS = {
    "learning_rate":    8.772259590429399e-04,
    "batch_size":       512,
    "gamma":            0.977038766982085,
    "seed":             312,                # will be incremented per run
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

LOG_FILE = "fixed_log.xlsx"
best_runs: list[tuple[float, float, str]] = []

LOG_COLUMNS = [
    "timestamp", "series", "trial_id", "model",
    "return", "sharpe", "cagr", "agent_vs_bnh", "params_json"
]


def init_log(series_tag: str) -> None:
    if not os.path.isfile(LOG_FILE):
        df = pd.DataFrame(columns=LOG_COLUMNS)
        df.to_excel(LOG_FILE, index=False)
    with open(LOG_FILE.replace(".xlsx", ".tag"), "a") as tagfile:
        tagfile.write(f"# NEW_SERIES {series_tag}\n")


def append_log(row: dict) -> None:
    df_new = pd.DataFrame([row])
    if os.path.isfile(LOG_FILE):
        with pd.ExcelWriter(LOG_FILE, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            book = writer.book
            sheet = writer.sheets["Sheet1"]
            start_row = sheet.max_row
            df_new.to_excel(writer, index=False, header=False, startrow=start_row)
    else:
        df_new.to_excel(LOG_FILE, index=False)

def _pick_env_and_dates(envset: str):
    if envset == "intraday":
        env_train = IntradayTradingTrainEnv
        env_test  = IntradayTradingTestEnv
        interval  = "1m"

        s_train = INTRA_TRAIN_START
        e_train = INTRA_TRAIN_END
        s_test  = INTRA_TEST_START
        e_test  = INTRA_TEST_END
    else:
        env_train = DailyTradingTrainEnv
        env_test  = DailyTradingTestEnv
        interval  = "1d"

        s_train = TRAIN_START_DATE
        e_train = TRAIN_END_DATE
        s_test  = TEST_START_DATE
        e_test  = TEST_END_DATE

    return env_train, env_test, interval, s_train, e_train, s_test, e_test

def run_once(run_idx: int, series_dir: str, model_name: str, series_tag: str, envset: str):
    erl_params = ERL_FIXED_PARAMS.copy()
    erl_params["seed"] += run_idx

    env_train, env_test, interval, s_train, e_train, s_test, e_test = _pick_env_and_dates(envset)
    cwd = os.path.join(series_dir, f"trial_{run_idx:03d}")
    os.makedirs(cwd, exist_ok=True)

    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params}, f, indent=2)

    train(
        start_date=s_train, end_date=e_train,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
        time_interval=interval, technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=env_train, model_name=model_name,
        cwd=cwd, erl_params=erl_params, break_step=BREAK_STEP
    )

    assets, sharpe, cagr, agent_vs_bnh = test(
        start_date=s_test, end_date=e_test,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
        time_interval=interval, technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=env_test, model_name=model_name,
        cwd=cwd, net_dimension=erl_params["net_dimension"]
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
        "params_json": json.dumps({"erl": erl_params})
    })

    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    return ret, sharpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=10)
    parser.add_argument("--model", default="sac")
    parser.add_argument("--note", default="trade", help="Series description")
    parser.add_argument("--envset", choices=["daily", "intraday"], default="daily",
                        help="daily = 1d env, intraday = 1m env")

    args = parser.parse_args()

    series_tag = f"FIXED_{args.model}_{args.envset}_{args.runs}runs_{args.note}"
    series_dir = os.path.join("fixed_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)
    init_log(series_tag)

    returns, sharpes = [], []
    for i in range(args.runs):
        r, s = run_once(i, series_dir, args.model, series_tag)
        returns.append(r)
        sharpes.append(s)
        print(f"[Run {i:02d}] Return={r:.4f}  Sharpe={s:.4f}")

    print("\n=== Aggregate results ===")
    print(f"avg Return = {sum(returns)/len(returns):.4f}")
    print(f"std Return = {np.std(returns):.4f}")
    print(f"avg Sharpe = {sum(sharpes)/len(sharpes):.4f}")
    print(f"std Sharpe = {np.std(sharpes):.4f}")
    print(f"\n→ Saved to {LOG_FILE}")
