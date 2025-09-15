#!/usr/bin/env python3
import os, json, argparse, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use(os.environ.get("MPLBACKEND", "Agg"))
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np_test import DailyTradingTestEnv
from finrl.meta.env_stock_trading.env_stocktrading_np_train import DailyTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_train import IntradayTradingTrainEnv
from finrl.meta.env_stock_trading.env_intraday_np_test import IntradayTradingTestEnv
from finrl.train import train
from finrl.test import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    ERL_FIXED_PARAMS,
    INDICATORS,
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    INTRA_TRAIN_START, INTRA_TRAIN_END,
    INTRA_TEST_START, INTRA_TEST_END,
    CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE,
    CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE,
    BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE,
    BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE,
)

"""
Przykład uruchomienia:

python -m finrl.single_run --model ppo --task trading --freq daily --dataset stable --outdir results/single

Dostępne opcje:
  --model      ppo | ddpg | sac | td3
  --task       trading | allocation
  --freq       daily | intraday
  --dataset    stable | volatile | bear | intraday
  --outdir     katalog wyników (default: results/single_run)
"""


def _pick_env_and_dates(task: str, freq: str, dataset: str):
    if freq == "intraday":
        interval = "1m"
        if dataset != "intraday":
            raise ValueError("Use 'intraday'")
        s_train, e_train = INTRA_TRAIN_START, INTRA_TRAIN_END
        s_test, e_test = INTRA_TEST_START, INTRA_TEST_END
    else:
        interval = "1d"
        if dataset == "stable":
            s_train, e_train = TRAIN_START_DATE, TRAIN_END_DATE
            s_test, e_test = TEST_START_DATE, TEST_END_DATE
        elif dataset == "volatile":
            s_train, e_train = CRISIS_TRAIN_START_DATE, CRISIS_TRAIN_END_DATE
            s_test, e_test = CRISIS_TEST_START_DATE, CRISIS_TEST_END_DATE
        elif dataset == "bear":
            s_train, e_train = BEAR2008_TRAIN_START_DATE, BEAR2008_TRAIN_END_DATE
            s_test, e_test = BEAR2008_TEST_START_DATE, BEAR2008_TEST_END_DATE
        else:
            raise ValueError("Use 'stable' or 'volatile' or 'bear'")
    if task == "trading":
        if freq == "intraday":
            env_train, env_test = IntradayTradingTrainEnv, IntradayTradingTestEnv
            ep_len, warmup = 390, 60
        else:
            env_train, env_test = DailyTradingTrainEnv, DailyTradingTestEnv
            ep_len, warmup = 252, 20
        env_extra_train = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=True)
        env_extra_test = dict(ep_len=ep_len, warmup_lookback=warmup, if_train=False)
    elif task == "allocation":
        from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
        env_train = env_test = StockPortfolioEnv
        if freq == "intraday":
            lookback, ep_len, warmup, rebal, rew_scale = 60, 390, 60, 5, (2**-9)
        else:
            lookback, ep_len, warmup, rebal, rew_scale = 252, 252, 20, 1, 1.0
        env_extra_train = dict(lookback=lookback, ep_len=ep_len, warmup_lookback=warmup, rebalance_every=rebal, transaction_cost_pct=1e-3, reward_scaling=rew_scale, if_train=True)
        env_extra_test = {**env_extra_train, "if_train": False}
    else:
        raise ValueError("task ∈ {'trading','allocation'}")
    return env_train, env_test, interval, s_train, e_train, s_test, e_test, env_extra_train, env_extra_test

def _filter_tickers_intersection(tickers, data_source, interval, s_train, e_train, s_test, e_test):
    dp = DataProcessor(data_source)
    df_tr = dp.clean_data(dp.download_data(tickers, s_train, e_train, interval))
    df_te = dp.clean_data(dp.download_data(tickers, s_test, e_test, interval))
    have_tr = set(df_tr["tic"].unique())
    have_te = set(df_te["tic"].unique())
    return [t for t in tickers if t in have_tr and t in have_te]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo")
    p.add_argument("--task", choices=["trading","allocation"], default="trading")
    p.add_argument("--freq", choices=["daily","intraday"], default="daily")
    p.add_argument("--dataset", choices=["stable","volatile","intraday","bear"], default="stable")
    p.add_argument("--seed", type=int, default=312)
    p.add_argument("--break_step", type=int, default=100000)
    p.add_argument("--outdir", default="results/single_run")
    args = p.parse_args()

    erl_params = ERL_FIXED_PARAMS.copy()
    erl_params["seed"] = args.seed
    env_train, env_test, interval, s_train, e_train, s_test, e_test, env_extra_train, env_extra_test = _pick_env_and_dates(args.task, args.freq, args.dataset)
    os.makedirs(args.outdir, exist_ok=True)
    tickers = _filter_tickers_intersection(DOW_30_TICKER, "yahoofinance", interval, s_train, e_train, s_test, e_test)
    cwd = os.path.join(args.outdir, f"{args.model}_{args.task}_{args.freq}_{args.dataset}_seed{args.seed}")
    os.makedirs(cwd, exist_ok=True)

    with open(os.path.join(cwd, "params.json"), "w") as f:
        json.dump({"erl": erl_params, "task": args.task, "freq": args.freq, "dataset": args.dataset, "interval": interval, "dates": dict(train=(s_train, e_train), test=(s_test, e_test)), "seed": args.seed}, f, indent=2)

    train(start_date=s_train, end_date=e_train, ticker_list=tickers, data_source="yahoofinance", time_interval=interval, technical_indicator_list=INDICATORS, drl_lib="elegantrl", env=env_train, model_name=args.model, cwd=cwd, erl_params=erl_params, break_step=args.break_step, env_extra=env_extra_train)

    test(start_date=s_test, end_date=e_test, ticker_list=tickers, data_source="yahoofinance", time_interval=interval, technical_indicator_list=INDICATORS, drl_lib="elegantrl", env=env_test, model_name=args.model, cwd=cwd, net_dimension=erl_params["net_dimension"], env_extra=env_extra_test)
    print(f"Equity curve saved to: {os.path.join(cwd, 'EquityCurve.jpg')}")

if __name__ == "__main__":
    main()
