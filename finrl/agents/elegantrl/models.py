"""
DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
"""

from __future__ import annotations

import torch
from elegantrl.agents.MAgentMADDPG import AgentMADDPG
from elegantrl.agents import *
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

MODELS = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
    "ppo": AgentPPO,
    "a2c": AgentA2C,
    "maddpg" : AgentMADDPG,
    "dqn": AgentDQN,
}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array, env_args):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        self.env_args = env_args

    def get_model(self, model_name, model_kwargs):
        self.env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        self.model_kwargs = model_kwargs
        self.gamma = model_kwargs.get("gamma", 0.985)

        env = self.env
        env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        stock_dim = self.price_array.shape[1]
        trading_state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        action_dim = stock_dim
        if getattr(self, "env_args", None) and "df" in self.env_args:
            df = self.env_args["df"]
            if "day" in getattr(df, "columns", []):
                max_step = int(df["day"].nunique())
            else:
                max_step = int(df.index.nunique())
        else:
            max_step = self.price_array.shape[0]

        if self.env_args is not None:
            tech_dim = int(self.tech_array.shape[1] // stock_dim)
            portfolio_state_dim = stock_dim * (stock_dim + tech_dim)
            merged_env_args = {
                **self.env_args,
                "env_name": env.__name__,
                "state_dim": portfolio_state_dim if env.__name__ == "StockPortfolioEnv" else trading_state_dim,
                "action_dim": action_dim,
                "if_discrete": False,
                "max_step": max_step,
            }
            self.env_args = merged_env_args
            self.state_dim = self.env_args["state_dim"]
            self.action_dim = action_dim
        else:
            self.state_dim = trading_state_dim
            self.action_dim = action_dim
            self.env_args = {
                "env_name": env.__name__,
                "config": self.env_config,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "if_discrete": False,
                "max_step": max_step,
            }

        model = Config(agent_class=agent, env_class=env, env_args=self.env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.break_step = int(
                    2e5
                )  # break training if 'total_step > break_step'
                model.net_dims = (
                    128,
                    64,
                )  # the middle layer dimension of MultiLayer Perceptron`
                model.gamma = self.gamma  # discount factor of future rewards
                model.max_step = int(self.env_args["max_step"])
                model.horizon_len = min(1024, model.max_step)
                model.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
                model.learning_rate = model_kwargs.get("learning_rate", 1e-4)
                model.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
                model.eval_times = model_kwargs.get("eval_times", 2**5)
                model.eval_per_step = int(2e4)
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, env_args):
        import torch
        agent_class = MODELS[model_name]
        env = environment

        try:
            state_dim = getattr(env, "state_dim", None) or env.observation_space.shape[0]
        except Exception:
            stock_dim_fb = env_args["price_array"].shape[1]
            state_dim = 1 + 2 + 3 * stock_dim_fb + env_args["tech_array"].shape[1]

        try:
            action_dim = getattr(env, "action_dim", None) or env.action_space.shape[0]
        except Exception:
            action_dim = env_args["price_array"].shape[1]

        if_discrete = getattr(env, "if_discrete", False)
        max_step = int(getattr(env, "max_step", 0)) or int(env_args["price_array"].shape[0] - 1)

        actor_path = f"{cwd}/act.pth"

        loaded = torch.load(actor_path, weights_only=False)

        if isinstance(loaded, torch.nn.Module):
            act = loaded
        else:
            net_dims = [net_dimension] if isinstance(net_dimension, int) else net_dimension
            args = Config(
                agent_class=agent_class,
                env_class=type(env),
                env_args=dict(
                    env_num=1,
                    env_name=type(env).__name__,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    if_discrete=if_discrete,
                    max_step=max_step,
                ),
            )
            args.cwd = cwd
            act = agent_class(net_dims, state_dim, action_dim, gpu_id=0, args=args).act
            state_dict = loaded if isinstance(loaded, dict) else loaded.state_dict()
            act.load_state_dict(state_dict, strict=False)

        device = next(act.parameters()).device

        s0 = env.reset()
        state = s0[0] if isinstance(s0, tuple) else s0

        init_asset = getattr(env, "initial_total_asset",
                             getattr(env, "initial_amount",
                                     getattr(env, "initial_capital", 1e6)))
        episode_total_assets = [float(init_asset)]

        max_step = int(getattr(env, "max_step", max_step))
        for _ in range(max_step):
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_tensor = act(s_tensor)
            action = a_tensor.argmax(dim=1).cpu().numpy()[0] if if_discrete else a_tensor.cpu().numpy()[0]

            out = env.step(action)
            if len(out) == 5:  # gymnasium API
                state, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:  # classic gym API
                state, reward, done, _ = out

            if hasattr(env, "portfolio_value"):
                total_asset = float(env.portfolio_value)  # Allocation
            elif hasattr(env, "asset_memory") and len(env.asset_memory) > 0:
                total_asset = float(env.asset_memory[-1])
            elif hasattr(env, "amount") and hasattr(env, "price_ary") and hasattr(env, "stocks"):
                total_asset = float(env.amount + (env.price_ary[env.day] * env.stocks).sum())  # Trading
            else:
                total_asset = episode_total_assets[-1] * (1.0 + float(reward))

            episode_total_assets.append(total_asset)
            if done:
                break

        print("Test Finished!")
        print("episode_return", episode_total_assets[-1] / episode_total_assets[0])
        return episode_total_assets

