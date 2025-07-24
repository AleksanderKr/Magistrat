# optuna_tune.py
#
# Usage examples:
#   python -m finrl.optuna_tune --trials 20 --mode single --model ddpg
#   python -m finrl.optuna_tune --trials 20 --mode pareto --model ppo
#   MODELS: ddpg, ppo, sac, td3
# Results:
#   • All runs are logged to     optuna_log.csv
#   • The best checkpoints live in  optuna_runs/<trial_id>
# ─────────────────────────────────────────────────────────────────────────────
import os, json, csv, datetime, argparse, shutil, optuna
from finrl.train import train
from finrl.test  import test
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE,  TEST_END_DATE,
)

LOG_FILE = "optuna_log_td3.csv"
TOP_K    = 5                                    # keep only top-K actors
best_runs: list[tuple[float, float, str]] = []  # (return, sharpe, path)


# ───────────────────────────── CSV helpers ────────────────────────────────
def _init_log(series_tag: str) -> None:
    """Create CSV header if missing and add a separator row announcing a
    new Optuna run-series."""
    new_file = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow([
                "timestamp", "series", "trial_id", "mode", "model",
                "alpha", "beta", "sharpe_w",
                "return", "sharpe", "reward",
                "params_json"
            ])
        wr.writerow([
            datetime.datetime.now().isoformat(), f"NEW_SERIES_{series_tag}",
            "", "", "", "", "", "", "", "", "", ""
        ])


def _append_log(row: dict) -> None:
    with open(LOG_FILE, "a", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "series", "trial_id", "mode", "model",
                "alpha", "beta", "sharpe_w",
                "return", "sharpe", "reward",
                "params_json"
            ],
        )
        wr.writerow(row)


# ────────────────── ElegantRL hyper-parameter sampler ────────────────────
def sample_erl_params(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "net_dimension": trial.suggest_categorical("net_dimension", [256, 512, 1024]),
        "net_dims":      [trial.suggest_categorical("hidden", [256, 512, 1024])]*2,
        # fixed but exposed here for clarity
        "target_step":   2048 * 4,
        "horizon_len":   2048,
        "repeat_times":  2.0,
        "buffer_size":   int(2e6),
        "buffer_init_size": 8192 * 4,
        "eval_gap":   64,
        "eval_times": 16,
    }


# ─────────────────────────── one Optuna trial ────────────────────────────
def run_trial(trial, mode: str, model_name: str, series_tag: str, series_dir: str):
    """
    Returns:
      • single → float  (return + w · Sharpe)
      • pareto → tuple  (return, Sharpe)
    """
    erl_params = sample_erl_params(trial)

    # reward-function weights passed to the environment
    alpha = trial.suggest_float("alpha", 0.7, 1.3)
    beta  = trial.suggest_float("beta",  0.0, 0.5)
    env_extra = dict(alpha=alpha, beta=beta)

    # additional Sharpe weight in the *objective* (single-objective mode)
    sharpe_w = (
        trial.suggest_float("sharpe_weight", 0.0, 1.0) if mode == "single" else None
    )

    run_id = f"trial_{trial.number:03d}"
    cwd    = os.path.join(series_dir, run_id)
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params, "alpha": alpha, "beta": beta}, f, indent=2)

    # ─────────── train ───────────
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
        break_step=3e5,
        env_extra=env_extra,
    )

    # ─────────── evaluate ────────
    assets, sharpe, _ = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=StockTradingEnv,
        model_name=model_name,
        cwd=cwd,
        net_dimension=erl_params["net_dimension"],
        env_extra=env_extra,
    )
    ret = assets[-1] / assets[0] - 1

    # ─────────── CSV log ──────────
    params_dict = {"erl": erl_params, "alpha": alpha, "beta": beta,
                   "sharpe_w": sharpe_w}
    _append_log(
        {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "series": series_tag,
            "trial_id": run_id,
            "mode": mode,
            "model": model_name,
            "alpha": f"{alpha:.3f}",
            "beta": f"{beta:.3f}",
            "sharpe_w": f"{sharpe_w:.2f}" if sharpe_w is not None else "",
            "return": f"{ret:.4f}",
            "sharpe": f"{sharpe:.4f}",
            "reward": f"{(ret + sharpe_w * sharpe):.4f}" if sharpe_w is not None else "",
            "params_json": json.dumps(params_dict)
        }
    )

    # ───────── housekeeping (keep top-K) ─────────
    best_runs.append((ret, sharpe, cwd))
    best_runs.sort(key=lambda t: t[0], reverse=True)  # sort by total return
    while len(best_runs) > TOP_K:
        _, _, path_to_del = best_runs.pop()
        shutil.rmtree(path_to_del, ignore_errors=True)

    # ───────── return for Optuna ─────────
    if mode == "single":
        return ret + sharpe_w * sharpe
    else:
        return ret, sharpe


# ───────────────────────────── script entry ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["single", "pareto"],
        default="single",
        help="single = maximise (return + w·Sharpe)  |  pareto = two-objective",
    )
    parser.add_argument(
        "--model",
        default="ddpg",
        help="ElegantRL model name (ddpg, td3, ppo, …)",
    )
    args = parser.parse_args()

    # ── create a dedicated directory for this run-series ─────────
    series_tag = f"{args.mode.upper()}_{args.model}_{args.trials}trials_" \
                 f"{datetime.datetime.now():%y%m%d_%H%M%S}"
    series_dir = os.path.join("optuna_runs", series_tag)
    os.makedirs(series_dir, exist_ok=True)

    _init_log(series_tag)  # CSV separator row

    if args.mode == "single":
        study_name = f"finrl_single_{args.model}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            direction="maximize",
            load_if_exists=True,
        )
    else:
        study_name = f"finrl_pareto_{args.model}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_finrl.db",
            directions=["maximize", "maximize"],
            load_if_exists=True,
        )

    study.optimize(
        lambda t: run_trial(t, args.mode, args.model, series_tag, series_dir),
        n_trials=args.trials,
    )

    # ────────── summary ───────────
    if args.mode == "single":
        print(
            f"Best trial {study.best_trial.number}: "
            f"Reward={study.best_value:.4f}\nParams={study.best_trial.params}"
        )
    else:
        print("Pareto front (return, Sharpe):")
        for tr in study.best_trials:
            print(f"  id={tr.number:3d}  Ret={tr.values[0]:.2%}  Sharpe={tr.values[1]:.3f}")
