"""
Entry point for all simulation runs.

Usage:
    python main.py --policy always_on
    python main.py --policy threshold
    python main.py --policy random
    python main.py --policy cql        (after training)
    python main.py --policy iql        (after training)
    python main.py --policy dqn        (after training)

Timestep split:
    Training slice   : 0   → 663  (4 weeks, used to generate offline dataset)
    Evaluation slice : 664 → 829  (1 week, all methods compared here)

For rule-based policies (always_on, threshold, random):
    Runs on full 830 timesteps but only evaluation slice is used for comparison.

For ML policies (cql, iql, dqn):
    Training happens separately.
    This file runs evaluation on the held-out slice only.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from simulator import RANEnv

#Timestep constants
TRAIN_START = 0
TRAIN_END   = 663
EVAL_START  = 664
EVAL_END    = 829

# Episode runner 
def run_episode(env: RANEnv, policy_fn, policy_name: str) -> pd.DataFrame:
    """
    Runs one full episode using policy_fn.

    policy_fn signature:
        action = policy_fn(obs, env)
        obs  : np.ndarray (322,)
        env  : RANEnv (policy can inspect env.prb, env.mr etc if needed)
        returns: np.ndarray of shape (n_micro,) with binary values

    Returns:
        DataFrame with one row per timestep
    """
    obs, _ = env.reset()
    done = False
    log  = []

    while not done:
        action          = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        info["policy"]  = policy_name
        info["action"]  = action.tolist()
        log.append(info)

    return pd.DataFrame(log)

# Save results 
def save_results(df: pd.DataFrame, policy_name: str, split: str):
    os.makedirs("results", exist_ok=True)
    path = f"results/{policy_name}_{split}.csv"
    df.to_csv(path, index=False)
    print(f"Results saved → {path}")

# Summary printer
def print_summary(df: pd.DataFrame, policy_name: str, split: str):
    print(f"\n{'='*50}")
    print(f"Policy  : {policy_name.upper()}")
    print(f"Split   : {split}")
    print(f"Steps   : {len(df)}")
    print(f"{'='*50}")
    print(f"  Total reward (W·steps) : {df['reward_W'].sum():.2f}")
    print(f"  Mean  reward (W)       : {df['reward_W'].mean():.4f}")
    print(f"  Mean  cells off        : {df['n_cells_off'].mean():.2f} / 39")
    print(f"  Mean  blocked (cap)    : {df['n_blocked_by_capacity'].mean():.2f}")
    print(f"  Mean  blocked (PRB)    : {df['n_blocked_by_micro_threshold'].mean():.2f}")
    print(f"  Mean  actual power (W) : {df['actual_power_W'].mean():.4f}")
    print(f"  Mean  baseline (W)     : {df['baseline_power_W'].mean():.4f}")
    print(f"  --- Scheme 1 (micros only, macros ignored) ---")
    print(f"  Mean  micro power (W)   : {df['actual_power_micro_only_W'].mean():.4f}")
    print(f"  Mean  micro baseline (W): {df['baseline_power_micro_only_W'].mean():.4f}")
    micro_base = df['baseline_power_micro_only_W'].mean()
    micro_act  = df['actual_power_micro_only_W'].mean()
    micro_saving = (1 - micro_act / micro_base) * 100 if micro_base > 0 else 0.0
    print(f"  Micro saving vs base (%) : {micro_saving:.2f}%")
    print(f"  --- Scheme 2 (macros frozen, no handoff cost) ---")
    macro_const_act = df['actual_power_macro_const_W'].mean()
    base = df['baseline_power_W'].mean()
    macro_const_saving = (1 - macro_const_act / base) * 100 if base > 0 else 0.0
    print(f"  Mean  power (macros frozen) (W) : {macro_const_act:.4f}")
    print(f"  Saving vs full baseline (%)     : {macro_const_saving:.2f}%")
    print(f"{'='*50}\n")

# Main
def main():
    parser = argparse.ArgumentParser(description="RAN CSO Simulator")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["always_on", "threshold", "random", "cql", "iql", "dqn"],
        help="Policy to run"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval", "full"],
        help="Timestep split to run on (default: eval)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="PRB threshold for threshold policy"
    )
    args = parser.parse_args()

    # Determine timestep slice
    if args.split == "train":
        start, end = TRAIN_START, TRAIN_END
    elif args.split == "eval":
        start, end = EVAL_START, EVAL_END
    else:  # full
        start, end = TRAIN_START, EVAL_END

    # Load policy
    if args.policy == "always_on":
        from policies import always_on_policy as policy_fn

    elif args.policy == "threshold":
        from policies import make_threshold_policy
        policy_fn = make_threshold_policy(threshold=args.threshold)

    elif args.policy == "random":
        from policies import random_policy as policy_fn

    elif args.policy == "cql":
        from policies import load_cql_policy
        policy_fn = load_cql_policy()

    elif args.policy == "iql":
        from policies import load_iql_policy
        policy_fn = load_iql_policy()

    elif args.policy == "dqn":
        from policies import load_dqn_policy
        policy_fn = load_dqn_policy()

    # Build env 
    env = RANEnv(
        prb_csv="Datasets/Base/simulator_ready_traffic_PRB.csv",
        mr_csv="Datasets/Base/simulator_ready_traffic_MR.csv",
        start=start,
        end=end,
        is_continuous_proxy=(args.policy == "iql"),
    )

    #  Run 
    print(f"\nRunning {args.policy} on {args.split} split "
          f"(timesteps {start}–{end})...")

    df = run_episode(env, policy_fn, policy_name=args.policy)

    print_summary(df, args.policy, args.split)
    save_results(df, args.policy, args.split)

    # Extra: save offline dataset for ML training
    # Only when generating training data with rule-based policies
    if args.policy in ["threshold", "random"] and args.split == "train":
        dataset_path = f"results/offline_dataset_{args.policy}.csv"
        df.to_csv(dataset_path, index=False)
        print(f"Offline dataset saved → {dataset_path}")

if __name__ == "__main__":
    main()