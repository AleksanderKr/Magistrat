from __future__ import annotations

import numpy as np
from finrl.config import INDICATORS, SHARPE_PARAMS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

def save_equity_curve(agent_curve: np.ndarray,
                      ref_curve: np.ndarray,
                      path: str,
                      start_date: str = "2020-07-01" ) -> None:

    norm_agent = agent_curve / agent_curve[0]
    norm_ref   = ref_curve  / ref_curve[0]

    n = len(norm_agent)
    x_vals = pd.bdate_range(start=start_date, periods=n)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(x_vals, norm_agent, label="Agent", linewidth=1.6)
    ax.plot(x_vals, norm_ref,   label="Buy & Hold", linewidth=1.4, linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlim(x_vals[0], x_vals[-1])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    env_extra=None,
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor

    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_args = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
    }

    env_config = {
        **env_args,
        "if_train": False,
        **SHARPE_PARAMS
    }
    if getattr(env, "__name__", "") == "StockPortfolioEnv":
        if "cov_list" not in data.columns:
            lookback = int((env_extra or {}).get("lookback", 252))
            data = dp.add_covariance_matrix(data, lookback=lookback)
            assert "cov_list" in data.columns, "add_covariance_matrix() did not add 'cov_list'"

        stock_dim = len(ticker_list)
        action_dim = stock_dim
        tech_dim = len(technical_indicator_list)
        state_space = stock_dim

        max_step_days = int(data["day"].nunique()) if "day" in data.columns else int(
            data["timestamp"].dt.date.nunique()
        )

        env_instance = StockPortfolioEnv(
            df=data,
            stock_dim=stock_dim,
            hmax=(env_extra or {}).get("hmax", 100),
            initial_amount=(env_extra or {}).get("initial_amount", 1e6),
            transaction_cost_pct=(env_extra or {}).get("transaction_cost_pct", 1e-3),
            reward_scaling=(env_extra or {}).get("reward_scaling", 100.0),
            state_space=state_space,
            action_space=action_dim,
            tech_indicator_list=technical_indicator_list,
            turbulence_threshold=(env_extra or {}).get("turbulence_threshold", None),
            lookback=(env_extra or {}).get("lookback", 252),
            day=(env_extra or {}).get("day", 0),
        )

        env_instance.max_step = min(max_step_days, 12345)

        env_args_erl = dict(
            env_name="StockPortfolioEnv",
            state_dim=stock_dim * (stock_dim + tech_dim),
            action_dim=action_dim,
            if_discrete=False,
            max_step=min(max_step_days, 12345),
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
    else:
        env_args = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
        }
        env_config = {**env_args, "if_train": False, **SHARPE_PARAMS}
        env_instance = env(config=env_config)
        env_args_erl = env_args

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
            env_args=env_args_erl
        )
        assets = np.asarray(episode_total_assets, dtype=float)

        # ---------- Buy & Hold ----------
        init_cash = getattr(env_instance, "initial_capital",
                            getattr(env_instance, "initial_amount", 1e6))
        first_px, last_px = price_array[0], price_array[-1]

        equal_cash = init_cash / len(first_px)
        shares = equal_cash / first_px
        bnh_curve = price_array @ shares
        bnh_final = bnh_curve[-1]
        bnh_return = bnh_final / init_cash

        # ---------- Agent  ----------
        daily_ret = np.diff(assets) / assets[:-1]
        ann_vol = daily_ret.std(ddof=0) * np.sqrt(252)
        cagr = (assets[-1] / assets[0]) ** (252 / len(daily_ret)) - 1
        max_dd = (assets / np.maximum.accumulate(assets) - 1).min()
        rf = 0.0
        sharpe = (daily_ret.mean() - rf / 252) / daily_ret.std(ddof=1) * np.sqrt(252) if daily_ret.std(
            ddof=1) > 0 else np.nan

        # ----------  BnH ----------
        bnh_daily_ret = np.diff(bnh_curve) / bnh_curve[:-1]
        bnh_ann_vol = bnh_daily_ret.std(ddof=0) * np.sqrt(252)
        bnh_max_dd = (bnh_curve / np.maximum.accumulate(bnh_curve) - 1).min()

        alpha_pct = assets[-1] / (assets[0] * bnh_return) - 1

        # ---------- print ----------
        print(
            f"Episode return: {assets[-1] / assets[0] - 1:.2%}   |   Sharpe: {sharpe:.3f}")
        print(f"Buy&Hold return: {bnh_return - 1:.2%}            |   Agent vs BnH: {alpha_pct:.2%}")
        print(f"CAGR: {cagr:.2%}   |   AnnVol: {ann_vol:.2%}   |   MaxDD: {max_dd:.2%}")
        print(f"BnH  AnnVol: {bnh_ann_vol:.2%}   |   MaxDD: {bnh_max_dd:.2%}")

        # ---------- plot ----------
        if len(bnh_curve) != len(assets):
            n = min(len(bnh_curve), len(assets))
            bnh_curve = bnh_curve[:n]
            assets = assets[:n]

        save_equity_curve(
            agent_curve=assets,
            ref_curve=bnh_curve,
            path=os.path.join(cwd, "EquityCurve.jpg"),
            start_date=TEST_START_DATE
        )

        return episode_total_assets, (cagr / ann_vol if ann_vol else np.nan), cagr, alpha_pct

    elif drl_lib == "rllib":
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )
        return episode_total_assets
    elif drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
        kwargs=kwargs,
    )

