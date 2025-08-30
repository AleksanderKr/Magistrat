from __future__ import annotations

from finrl.config import ERL_PARAMS, SHARPE_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER, SINGLE_TICKER, DOW_5_TEST
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

# construct environment


def train(
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
    # download data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    if getattr(env, "__name__", "") == "StockPortfolioEnv":
        if not {"cov_list"}.issubset(set(data.columns)):
            data = dp.add_covariance_matrix(data, lookback=int((env_extra or {}).get("lookback", 252)))

        stock_dim = len(ticker_list)
        action_dim = stock_dim
        state_space = stock_dim

        env = StockPortfolioEnv

        max_step_days = int(data["day"].nunique() - 1) if "day" in data.columns else int(
            data["timestamp"].dt.date.nunique() - 1)
        env_args = dict(
            df=data,
            stock_dim=stock_dim,
            hmax=(env_extra or {}).get("hmax", 100),
            initial_amount=(env_extra or {}).get("initial_amount", 1e6),
            transaction_cost_pct=(env_extra or {}).get("transaction_cost_pct", 1e-3),
            reward_scaling=(env_extra or {}).get("reward_scaling", 100),
            state_space=(env_extra or {}).get("state_space", state_space),
            action_space=(env_extra or {}).get("action_space", action_dim),
            tech_indicator_list=technical_indicator_list,
            turbulence_threshold=(env_extra or {}).get("turbulence_threshold", None),
            lookback=(env_extra or {}).get("lookback", 252),
            day=(env_extra or {}).get("day", 0),
            max_step=min(max_step_days, 12345),
        )
    else:
        env_args = {
            "config": {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": True,
                **SHARPE_PARAMS,
                **(env_extra or {}),
            }
        }

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            env_args=env_args,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model, model_config = agent_rllib.get_model(model_name)
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        kwargs=kwargs,
    )
