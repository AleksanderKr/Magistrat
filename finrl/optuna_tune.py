"""
Optuna tuner for FinRL agents.

Modes
-----
* single  → maximise **return** only (one‑objective)
* pareto  → maximise **return** and **Sharpe** simultaneously (two‑objective)

Usage examples
--------------
python -m finrl.optuna_tune --trials 20 --mode single --model ddpg
python -m finrl.optuna_tune --trials 20 --mode pareto  --model ppo

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

from finrl.train import train
from finrl.test import test
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VALIDATION_START_DATE,
    VALIDATION_END_DATE,
)

# ───────────────────────────── logging ──────────────────────────────
LOG_FILE = "optuna_log_second_pareto_ppo.csv"
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
        "net_dimension": trial.suggest_categorical("net_dimension", [256, 512, 1024]),
        "net_dims": net_dims_options[net_dims_key],
        "target_step": 2048 * 4,
        "horizon_len": 2048,
        "repeat_times": trial.suggest_categorical("repeat_times", [2.0, 3.0, 4.0]),
        "buffer_size": int(1e6),
        "buffer_init_size": batch_size * 2,
        "eval_gap": 64,
        "eval_times": 16,
    }


# ───────────────────────── one Optuna trial ───────────────────────

def run_trial(trial, mode: str, model_name: str, series_tag: str, series_dir: str):
    """Run a single trial.

    Returns
    -------
    single → float   (return)
    pareto → tuple   (return, sharpe)
    """
    erl_params = sample_erl_params(trial)

    run_id = f"trial_{trial.number:03d}"
    cwd = os.path.join(series_dir, run_id)
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params}, f, indent=2)

    # ─── train ───
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=StockTradingEnv,
        model_name=model_name,
        cwd=cwd,
        erl_params=erl_params,
        break_step=int(3e5),
    )

    # ─── evaluate ───
    assets, sharpe, _, agent_vs_bnh = test(
        start_date=VALIDATION_START_DATE,
        end_date=VALIDATION_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=StockTradingEnv,
        model_name=model_name,
        cwd=cwd,
        net_dimension=erl_params["net_dimension"],
    )
    ret = assets[-1] / assets[0] - 1

    # ─── CSV log ───
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
            "params_json": json.dumps({"erl": erl_params}),
        }
    )

    # ─── keep top‑K checkpoints by return ───
    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    # ─── objective ───
    if mode == "single":
        return ret  # maximise return only
    else:  # pareto
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
    args = parser.parse_args()

    # new study & folder
    series_tag = f"{args.mode.upper()}_{args.model}_{args.trials}trials_" \
                 f"{datetime.datetime.now():%y%m%d_%H%M%S}"
    series_dir = os.path.join("optuna_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)

    _init_log(series_tag)

    if args.mode == "single":
        study_name = f"finrl_single_second_{args.model}"
        #study_name = f"finrl_Second_single_sac_SINGLE_sac_20trials_250729_082359"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            direction="maximize",
            load_if_exists=True,
        )
    else:
        study_name = f"finrl_pareto_second_{args.model}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            directions=["maximize", "maximize"],
            sampler=NSGAIISampler(),
            load_if_exists=True,
        )

    study.optimize(
        lambda t: run_trial(t, args.mode, args.model, series_tag, series_dir),
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
