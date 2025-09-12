#!/usr/bin/env python
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

"""
python -m finrl.draw_pareto --task trading --freq daily --dataset stable --model ppo --eps-sharpe 0.01
python -m finrl.draw_pareto --task allocation --freq intraday --dataset intraday --model ppo --log-dir /path/to/logs
"""

def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    for c in ("return", "sharpe"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def _filter(df: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df[df["model"].astype(str).str.lower() == model.lower()]
    df = df[df["mode"].astype(str).str.lower() == "pareto"]
    df = df.dropna(subset=["return", "sharpe"])
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    keys = [c for c in ("series","trial_id","model") if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df[["trial_id","series","mode","model","return","sharpe"]].reset_index(drop=True)

def _pareto_front(df: pd.DataFrame, eps_sharpe: float = 0.0) -> pd.DataFrame:
    d = df.sort_values(["return","sharpe"], ascending=[False,False]).reset_index(drop=True)
    best_s = -float("inf")
    keep = []
    for i, s in enumerate(d["sharpe"]):
        if s > best_s - eps_sharpe:
            keep.append(i)
            if s > best_s:
                best_s = s
    front = d.loc[keep].copy()
    return front.sort_values(["return","sharpe"], ascending=[True,True]).reset_index(drop=True)

def draw_pareto(csv_path: str, model: str, out_path: str | None = None, eps_sharpe: float = 0.0) -> str:
    raw = _read_csv(csv_path)
    if raw.empty:
        raise RuntimeError("CSV is empty or unreadable.")
    df = _filter(raw, model)
    if df.empty:
        raise RuntimeError("No Pareto-mode rows for the given model.")
    front = _pareto_front(df, eps_sharpe=eps_sharpe)

    plt.figure(figsize=(6.5, 4.5), dpi=150)
    plt.scatter(df["return"], df["sharpe"], s=14, alpha=0.35, label="All trials")
    plt.plot(front["return"], front["sharpe"], linewidth=2, label="Pareto front")
    plt.scatter(front["return"], front["sharpe"], s=28)
    plt.xlabel("Return")
    plt.ylabel("Sharpe")
    plt.title(f"Pareto front: PPO")
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend(loc="best", frameon=False)

    # build default path under finrl/figures/pareto
    if out_path is None:
        pkg_dir = os.path.dirname(__file__)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_dir = os.path.join(pkg_dir, "figures", "pareto")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_pareto_{model}.png")
    else:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {os.path.abspath(out_path)}")
    print(f"Points: {len(df)} | On Pareto front: {len(front)}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["trading","allocation"], required=True)
    p.add_argument("--freq", choices=["daily","intraday"], required=True)
    p.add_argument("--dataset", choices=["stable","volatile","intraday","bear"], required=True)
    p.add_argument("--model", required=True, help="ppo | sac | td3 | ddpg")
    p.add_argument("--log-dir", default=".", help="directory with optuna CSV files")
    p.add_argument("--eps-sharpe", type=float, default=0.0)
    args = p.parse_args()

    csv_name = f"OPTUNA_CSV/optuna_{args.task}_{args.freq}_{args.dataset}.csv"
    csv_path = os.path.join(args.log_dir, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {os.path.abspath(csv_path)}")

    draw_pareto(csv_path, args.model, out_path=None, eps_sharpe=args.eps_sharpe)
