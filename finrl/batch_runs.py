# batch_runs.py  ────────────────────────────────────────────────────────────
# Fire off a fixed-param train-and-test multiple times.
# Logs →  fixed_log.csv      Checkpoints →  fixed_runs/<series>/<trial_id>/
# python -m finrl.batch_runs --runs 20 --model sac
# --------------------------------------------------------------------------
import os, json, csv, shutil, datetime, argparse

import numpy as np

from finrl.train import train
from finrl.test  import test
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE

# ───────── Fixed ElegantRL hyper-parameters ────────────
ERL_FIXED_PARAMS = {
    "learning_rate":   4.21016461159303e-05,
    "batch_size":      4096,
    "gamma":           0.9989652812396075,
    "seed":            312,                # will be incremented per run
    "net_dimension":   1024,
    "net_dims":        [256, 256],
    "target_step":     8192,
    "horizon_len":     4096,
    "repeat_times":    4.0,
    "buffer_size":     int(1e6),
    "buffer_init_size":16384,
    "if_use_per":      False,
    "eval_gap":        256,
    "eval_times":      16,
}
ALPHA = 0.7523392858019977     # reward weight log-return
BETA  = 0.03639080558854646    # reward weight DSR
BREAK_STEP = 3_00_000          # ~3·10^5 env steps
TOP_K      = 5                 # keep best K dirs (by total return)

LOG_FILE = "fixed_log.csv"
best_runs: list[tuple[float, float, str]] = []   # (return, sharpe, path)

# ───────── CSV helpers ─────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "series", "trial_id", "model",
    "alpha", "beta",
    "return", "sharpe",
    "params_json"
]

def init_log(series_tag: str) -> None:
    new_file = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(CSV_HEADER)
        wr.writerow([datetime.datetime.now().isoformat(),
                     f"NEW_SERIES_{series_tag}", "", "", "", "", "", "", ""])

def append_log(row: dict) -> None:
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)

# ───────── One training-testing cycle ──────────────────────────────────────
def run_once(run_idx: int, series_dir: str, model_name: str, series_tag: str):
    # unique sub-folder & seed
    erl_params = ERL_FIXED_PARAMS.copy()
    erl_params["seed"] = ERL_FIXED_PARAMS["seed"] + run_idx
    cwd = os.path.join(series_dir, f"trial_{run_idx:03d}")
    os.makedirs(cwd, exist_ok=True)

    env_extra = dict(alpha=ALPHA, beta=BETA)

    # save hyper-params for reproducibility
    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params, **env_extra}, f, indent=2)

    # ——— train ———
    train(
        start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
        time_interval="1D", technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=StockTradingEnv, model_name=model_name,
        cwd=cwd, erl_params=erl_params, break_step=BREAK_STEP,
        env_extra=env_extra
    )

    # ——— test  ———
    assets, sharpe, _ = test(
        start_date=TEST_START_DATE, end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER, data_source="yahoofinance",
        time_interval="1D", technical_indicator_list=INDICATORS,
        drl_lib="elegantrl", env=StockTradingEnv, model_name=model_name,
        cwd=cwd, net_dimension=erl_params["net_dimension"],
        env_extra=env_extra
    )
    ret = assets[-1] / assets[0] - 1

    # ——— CSV log ———
    append_log({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "series":    series_tag,
        "trial_id":  f"trial_{run_idx:03d}",
        "model":     model_name,
        "alpha":     f"{ALPHA:.3f}",
        "beta":      f"{BETA:.3f}",
        "return":    f"{ret:.4f}",
        "sharpe":    f"{sharpe:.4f}",
        "params_json": json.dumps({"erl": erl_params, **env_extra})
    })

    # ——— keep only TOP-K best ———
    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    return ret, sharpe

# ───────── Main —──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=10,
                        help="How many independent train+test cycles to run")
    parser.add_argument("--model", default="sac",
                        help="ElegantRL model name (ddpg, ppo, sac, td3 …)")
    args = parser.parse_args()

    series_tag = f"FIXED_{args.model}_{args.runs}runs_{datetime.datetime.now():%y%m%d_%H%M%S}"
    series_dir = os.path.join("fixed_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)
    init_log(series_tag)

    returns, sharpes = [], []
    for i in range(args.runs):
        r, s = run_once(i, series_dir, args.model, series_tag)
        returns.append(r); sharpes.append(s)
        print(f"[Run {i:02d}] Return={r:.4f}  Sharpe={s:.4f}")

    print("\n=== Aggregate results ===")
    print(f"avg Return = {sum(returns)/len(returns):.4f}")
    print(f"std Return = {np.std(returns):.4f}")
    print(f"avg Sharpe = {sum(sharpes)/len(sharpes):.4f}")
    print(f"std Sharpe = {np.std(sharpes):.4f}")
