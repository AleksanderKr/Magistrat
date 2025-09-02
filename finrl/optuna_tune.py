"""
Optuna tuner for FinRL agents.

Modes
-----
* single  → maximise **return** only (one‑objective)
* pareto  → maximise **return** and **Sharpe** simultaneously (two‑objective)

Usage examples
--------------
python -m finrl.optuna_tune --trials 20 --mode single --model ppo --task trading --freq daily --dataset stable

All runs are logged to ``optuna_log.csv`` and the best checkpoints are kept
in ``optuna_runs/<trial_id>``.
"""
import argparse
import csv
import datetime
import json
import os
import shutil
from typing import List, Tuple

import optuna
from optuna.samplers import NSGAIISampler

from finrl.meta.env_stock_trading.env_stocktrading_np_test import DailyTradingTestEnv
from finrl.meta.env_stock_trading.env_stocktrading_np_train import DailyTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_train import IntradayTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_test import IntradayTradingTestEnv
from finrl.train import train
from finrl.test import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE, TRAIN_END_DATE,
    VALIDATION_START_DATE, VALIDATION_END_DATE,
    CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE,
    CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE,
    INTRA_TRAIN_START, INTRA_TRAIN_END,
    INTRA_VAL_START, INTRA_VAL_END,
)

# ───────────────────────────── logging ──────────────────────────────
TOP_K = 5                           # keep only top‑K checkpoints
best_runs: List[Tuple[float, float, str]] = []  # (return, sharpe, path)


def _init_log(series_tag: str) -> None:
    """Create CSV header if missing and add a separator row."""
    header = [
        "timestamp",
        "series",
        "trial_id",
        "mode",
        "model",
        "return",
        "sharpe",
        "agent_vs_bnh",
        "params_json",
    ]
    new_file = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(header)
        wr.writerow(
            [datetime.datetime.now().isoformat(), f"NEW_SERIES_{series_tag}"] + [""] * (len(header) - 2)
        )


def _append_log(row: dict) -> None:
    fieldnames = [
        "timestamp",
        "series",
        "trial_id",
        "mode",
        "model",
        "return",
        "sharpe",
        "agent_vs_bnh",
        "params_json",
    ]
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


# ─────────────── ElegantRL hyper‑parameter sampler ────────────────

def sample_erl_params(trial) -> dict:
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])

    net_dims_options = {
        "256_256": [256, 256],
        "512_256": [512, 256],
        "512_512": [512, 512],
        "1024_512": [1024, 512],
    }
    net_dims_key = trial.suggest_categorical("net_dims_key", list(net_dims_options.keys()))

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": batch_size,
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "net_dims": net_dims_options[net_dims_key],
        "target_step": 2048 * 4,
        "horizon_len": 2048,
        "repeat_times": trial.suggest_categorical("repeat_times", [2.0, 3.0, 4.0]),
        "buffer_size": int(1e6),
        "buffer_init_size": batch_size * 2,
        "eval_gap": 64,
        "eval_times": 16,
        # "net_dimension": trial.suggest_categorical("net_dimension", [256, 512, 1024]),
    }


# ───────────────────────── one Optuna trial ───────────────────────

def run_trial(trial, mode: str, model_name: str, series_tag: str, series_dir: str,
              task: str, freq: str, dataset: str):
    """
    task:       'trading' | 'allocation'
    freq:       'daily' | 'intraday'
    dataset:    'stable' | 'crisis' | 'intraday'
    """
    # ────────────────────────────────────────────
    if freq == "intraday":
        interval = "1m"
        if dataset != "intraday":
            raise ValueError("Use 'intraday'.")
        sdate, edate  = INTRA_TRAIN_START, INTRA_TRAIN_END
        svdate, evdate = INTRA_VAL_START, INTRA_VAL_END
    else:
        interval = "1d"
        if dataset == "stable":
            sdate, edate  = TRAIN_START_DATE, TRAIN_END_DATE
            svdate, evdate = VALIDATION_START_DATE, VALIDATION_END_DATE
        elif dataset == "crisis":
            sdate, edate  = CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE
            svdate, evdate = CRISIS_VALIDATION_START_DATE, CRISIS_VALIDATION_END_DATE
        else:
            raise ValueError("Use 'stable' or 'crisis'.")

    # ─────────────────────────────
    if task == "trading":
        if freq == "intraday":
            env_train = IntradayTradingTrainEnv
            env_test  = IntradayTradingTestEnv
            ep_len, warmup = 390, 60
        else:
            env_train = DailyTradingTrainEnv
            env_test  = DailyTradingTestEnv
            ep_len, warmup = 252, 20

        env_extra_train = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=True)
        env_extra_test  = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=False)

    elif task == "allocation":
        from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
        env_train = StockPortfolioEnv
        env_test  = StockPortfolioEnv
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
        raise ValueError("Task can be 'trading' or 'allocation'")

    # ─────────────────────────────────────
    erl_params = sample_erl_params(trial)

    # ───────────────────────────────────────────────────────────
    run_id = f"trial_{trial.number:03d}"
    cwd = os.path.join(series_dir, run_id)
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params,
                   "task": task, "freq": freq, "dataset": dataset,
                   "interval": interval,
                   "dates": dict(train=(sdate, edate), val=(svdate, evdate))}, f, indent=2)

    # ──────────────────────────────────────────────────────────────────
    train(
        start_date=sdate,
        end_date=edate,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval=interval,
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env_train,
        model_name=model_name,
        cwd=cwd,
        erl_params=erl_params,
        break_step=int(3e5),
        env_extra=env_extra_train,
    )

    # ────────────────────────────────────────────────────────────────
    assets, sharpe, cagr, agent_vs_bnh = test(
        start_date=svdate,
        end_date=evdate,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval=interval,
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env_test,
        model_name=model_name,
        cwd=cwd,
        #net_dimension=erl_params["net_dimension"],
        env_extra=env_extra_test,
    )
    ret = assets[-1] / assets[0] - 1

    # ─── log CSV ───────────────────────────────────────────────────────────────
    _append_log(
        {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "series": series_tag,
            "trial_id": run_id,
            "mode": mode,
            "model": model_name,
            "return": f"{ret:.4f}",
            "sharpe": f"{sharpe:.4f}",
            "agent_vs_bnh": f"{agent_vs_bnh:.4f}",
            "params_json": json.dumps({
                "erl": erl_params,
                "task": task, "freq": freq, "dataset": dataset
            }),
        }
    )

    # ─── keep top-K ───────────────────────────────────────────────────────────
    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    # ───────────────────────────────────────────────────────
    if mode == "single":
        return ret
    else:
        return ret, sharpe


# ─────────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["single", "pareto"],
        default="single",
        help="single = maximise return  |  pareto = optimise (return, Sharpe)",
    )
    parser.add_argument(
        "--model",
        default="ddpg",
        help="ElegantRL model name (ddpg, td3, ppo, sac)",
    )
    parser.add_argument("--task", choices=["trading", "allocation"], default="trading")
    parser.add_argument("--freq", choices=["daily", "intraday"], default="daily")
    parser.add_argument("--dataset", choices=["stable", "crisis", "intraday"], default="stable")

    args = parser.parse_args()
    LOG_FILE = f"optuna_{args.task}_{args.freq}_{args.dataset}.csv"

    # new study & folder
    series_tag = f"{args.task.upper()}_{args.mode.upper()}_{args.model}_{args.freq}_{args.dataset}_{args.trials}trials_{datetime.datetime.now():%y%m%d_%H%M%S}"
    series_dir = os.path.join("optuna_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)

    _init_log(series_tag)

    SEED = 312
    if args.mode == "single":
        study_name = f"finrl_single_{args.model}_{args.task}_{args.freq}_{args.dataset}"
        #study_name = f"finrl_Second_single_sac_SINGLE_sac_20trials_250729_082359"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            load_if_exists=True,
        )
    else:
        study_name = f"finrl_pareto_{args.model}_{args.task}_{args.freq}_{args.dataset}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            directions=["maximize", "maximize"],
            sampler=NSGAIISampler(seed=SEED),
            load_if_exists=True,
        )

    study.optimize(
        lambda t: run_trial(t, args.mode, args.model, series_tag, series_dir,
                            args.task, args.freq, args.dataset),
        n_trials=args.trials,
    )
    # summary
    if args.mode == "single":
        best = study.best_trial
        print(f"Best trial {best.number}: Return={best.value:.4f}\nParams={best.params}")
    else:
        print("Pareto front (return, Sharpe):")
        for tr in study.best_trials:
            print(f"  id={tr.number:3d}  Ret={tr.values[0]:.2%}  Sharpe={tr.values[1]:.3f}")
