from __future__ import annotations

import numpy as np
from finrl.config import INDICATORS, SHARPE_PARAMS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
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
    env_instance = env(config=env_config)

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
            env_args=env_args
        )
        assets = np.asarray(episode_total_assets, dtype=float)

        # ---------- Buy & Hold ----------
        init_cash = env_instance.initial_capital
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

        # ----------  BnH ----------
        bnh_daily_ret = np.diff(bnh_curve) / bnh_curve[:-1]
        bnh_ann_vol = bnh_daily_ret.std(ddof=0) * np.sqrt(252)
        bnh_max_dd = (bnh_curve / np.maximum.accumulate(bnh_curve) - 1).min()

        alpha_pct = assets[-1] / (assets[0] * bnh_return) - 1  # „pobicie” BnH

        # ---------- print ----------
        print(
            f"Episode return: {assets[-1] / assets[0] - 1:.2%}   |   Sharpe: {(cagr / ann_vol if ann_vol else np.nan):.3f}")
        print(f"Buy&Hold return: {bnh_return - 1:.2%}            |   Agent vs BnH: {alpha_pct:.2%}")
        print(f"CAGR: {cagr:.2%}   |   AnnVol: {ann_vol:.2%}   |   MaxDD: {max_dd:.2%}")
        print(f"BnH  AnnVol: {bnh_ann_vol:.2%}   |   MaxDD: {bnh_max_dd:.2%}")

        # ---------- plot ----------

        save_equity_curve(
            agent_curve=assets,
            ref_curve=bnh_curve,
            path=os.path.join(cwd, "EquityCurve.jpg"),
            start_date=TEST_START_DATE
        )

        return episode_total_assets, (cagr / ann_vol if ann_vol else np.nan), cagr

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

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # # demo for stable baselines3
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac.zip",
    # )
