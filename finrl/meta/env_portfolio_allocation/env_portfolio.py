from __future__ import annotations

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
        rebalance_every=1,
        ep_len=390,
        warmup_lookback=60,
        if_train=True
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_dim = int(action_space)
        self.tech_indicator_list = tech_indicator_list
        self.rebalance_every = int(rebalance_every)
        self.ep_len = int(ep_len)
        self.warmup = int(warmup_lookback)
        self.if_train = bool(if_train)
        self.start_step = 0
        self.end_step = 0
        self._eval_ptr = 0

        if "step" not in self.df.index.names:
            time_col = "timestamp" if "timestamp" in self.df.columns else "date"
            # sort & deduplicate in case upstream missed it
            self.df = (
                self.df.sort_values([time_col, "tic"])
                .drop_duplicates([time_col, "tic"], keep="last")
            )
            if "step" not in self.df.columns:
                self.df["step"] = pd.factorize(pd.to_datetime(self.df[time_col]))[0].astype(int)
            self.df = self.df.set_index("step"); self.granularity = "intraday" if "timestamp" in self.df.columns else "daily"

        # Cache episode length and max_step (used by evaluator / prediction)
        self.n_steps = int(self.df.index.nunique())
        self.max_step = min(self.n_steps - 1, self.warmup + self.ep_len - 1)

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dim,), dtype=np.float32)
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        obs_dim = self.stock_dim * (self.stock_dim + len(self.tech_indicator_list))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        ).astype(np.float32).reshape(-1)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        #self.date_memory = [self.data.date.unique()[0]]
        first_label = (
            self.data["date"].unique()[0]
            if "date" in self.data.columns
            else pd.to_datetime(self.data["timestamp"]).dt.date.unique()[0]
        )
        self.date_memory = [first_label]

    def _current_time_label(self) -> pd.Timestamp:
        # prefer 'timestamp' (intraday), fallback do 'date' (daily)
        if "timestamp" in self.data.columns:
            ts = pd.to_datetime(self.data["timestamp"], errors="coerce")
            if ts.notna().any():
                return pd.Timestamp(ts.max())
        if "date" in self.data.columns:
            d = self.data["date"].iloc[0]
            return pd.Timestamp(d)

        if "timestamp" in self.df.columns:
            ts = pd.to_datetime(self.df.loc[self.day, "timestamp"], errors="coerce")
            if isinstance(ts, pd.Series):
                ts = ts.max()
            if pd.notna(ts):
                return pd.Timestamp(ts)
        if "date" in self.df.columns:
            d = self.df.loc[self.day, "date"]
            if isinstance(d, pd.Series):
                d = d.iloc[0]
            return pd.Timestamp(d)

        return pd.Timestamp.utcnow()

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= self.max_step
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            """
            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")
            """
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                #print("Sharpe: ", sharpe)
            #print("=================================")

            return self.state, self.reward, self.terminal, False, {}
        else:
            # actions -> portfolio weights; normalize to sum to 1
            last_weights = self.actions_memory[-1]  # weights at the end of the previous step
            new_weights = self.softmax_normalization(actions)  # candidate weights for the next step
            last_day_memory = self.data  # keep prices from day t
            # trade only every N steps (no cost, no change on off-steps)
            if self.rebalance_every > 1 and (self.day % self.rebalance_every) != 0:
                new_weights = last_weights

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            ).astype(np.float32).reshape(-1)

            rel = (self.data.close.values / last_day_memory.close.values) - 1.0
            gross_return = float(np.sum(rel * new_weights))

            c = float(getattr(self, "transaction_cost_pct", 1e-3))
            turnover = float(np.sum(np.abs(new_weights - last_weights)))
            cost_frac = c * turnover
            # net return after paying the rebalancing cost upfront
            effective_return = (1.0 - cost_frac) * (1.0 + gross_return) - 1.0
            # update portfolio value
            self.portfolio_value *= (1.0 + effective_return)
            # save to memory (net returns)
            self.actions_memory.append(new_weights)
            self.portfolio_return_memory.append(effective_return)
            #self.date_memory.append(self.data.date.unique()[0])
            self.date_memory.append(self._current_time_label())
            self.asset_memory.append(self.portfolio_value)
            # reward used for learning (scaled net return)
            self.reward = effective_return * float(self.reward_scaling)

        return self.state, self.reward, self.terminal, False, {}

    def _flatten(self, s):
        return np.asarray(s, dtype=np.float32).reshape(-1)

    def _build_sessions(self):
        if getattr(self, "granularity", "intraday") == "intraday":
            time_col = "timestamp" if "timestamp" in self.df.columns else "date"
            steps = (
                self.df.reset_index()[["step", time_col]]
                .drop_duplicates("step")
                .sort_values("step")
                .copy()
            )
            steps["ts"] = pd.to_datetime(steps[time_col])
            steps["date"] = steps["ts"].dt.date
            open_time = pd.Timestamp("09:30").time()
            open_step_map = (
                steps[steps["ts"].dt.time == open_time]
                .drop_duplicates("date")
                .set_index("date")["step"]
                .to_dict()
            )
            first_step_map = steps.groupby("date")["step"].min().to_dict()
            last_step_map = steps.groupby("date")["step"].max().to_dict()
            session_starts = []
            for d in first_step_map.keys():
                start = open_step_map.get(d, first_step_map[d])
                day_last = last_step_map[d]
                if start + self.warmup + self.ep_len - 1 <= day_last:
                    session_starts.append(start)
            self._session_starts = np.asarray(session_starts, dtype=int)
        else:
            span = int(self.warmup + self.ep_len)
            total = int(self.df.index.nunique())
            last_start = max(total - span, 0)
            self._session_starts = np.arange(0, last_start + 1, dtype=int)

        if self._session_starts.size == 0:
            self._session_starts = np.asarray([0], dtype=int)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        if not hasattr(self, "_session_starts"):
            self._build_sessions()

        if self.if_train:
            base_idx = int(np.random.randint(self._session_starts.size))
        else:
            base_idx = 0

        base_start = int(self._session_starts[base_idx])
        self.start_step = base_start + self.warmup
        self.end_step = self.start_step + self.ep_len - 1

        self.day = self.start_step
        self.max_step = self.end_step

        self.asset_memory = [self.initial_amount]
        self.data = self.df.loc[self.day, :]

        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        ).astype(np.float32).reshape(-1)

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self._current_time_label()]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        x = np.asarray(actions, dtype=np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
