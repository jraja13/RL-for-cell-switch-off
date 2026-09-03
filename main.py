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
    overload_fraction = (df['n_macro_overload'] > 0).mean() * 100   # fraction of timesteps -> mean is correct
    print(f"  Overload timestep (%)  : {overload_fraction:.2f}%")

    # Switching oscillation ("flip-flopping") — genuine ping-pong within 3 timesteps
    actions = np.array(df['action'].tolist())   # (T, 39)
    # A flip-flop = state at t equals state at t-2, but differs from state at t-1
    # i.e. X -> Y -> X pattern (Y != X), caught within a 3-step sliding window
    flip_flop_events = (actions[:-2] == actions[2:]) & (actions[:-2] != actions[1:-1])
    # shape (T-2, 39) boolean — True wherever a flip-flop centred on t-1 occurred
    flip_flops_per_cell = flip_flop_events.sum(axis=0)          # (39,)
    total_flip_flops = int(flip_flops_per_cell.sum())
    mean_flip_flops_per_cell = flip_flops_per_cell.mean()
    pct_cells_flip_flopped = (flip_flops_per_cell > 0).mean() * 100
    print(f"  Flip-flop events (3-step) : {total_flip_flops}")
    print(f"  Mean flip-flops / cell    : {mean_flip_flops_per_cell:.2f}")
    print(f"  Cells that flip-flopped   : {pct_cells_flip_flopped:.1f}%")

    print(f"  Mean  actual power (W) : {df['actual_power_W'].mean():.4f}")
    print(f"  Mean  baseline (W)     : {df['baseline_power_W'].mean():.4f}")

    # Full model saving — total energy over the week, so sum not mean
    total_actual  = df['actual_power_W'].sum()
    total_base    = df['baseline_power_W'].sum()
    full_saving = (1 - total_actual / total_base) * 100 if total_base > 0 else 0.0
    print(f"  Total actual energy (W·steps)   : {total_actual:.2f}")
    print(f"  Total baseline energy (W·steps) : {total_base:.2f}")
    print(f"  Full model saving (%)  : {full_saving:.2f}%   <-- Scheme 1 (headline)")

    print(f"  --- Scheme 2 (micros only, macros ignored) ---")
    micro_base_total = df['baseline_power_micro_only_W'].sum()
    micro_act_total  = df['actual_power_micro_only_W'].sum()
    micro_saving = (1 - micro_act_total / micro_base_total) * 100 if micro_base_total > 0 else 0.0
    print(f"  Total micro power (W·steps)   : {micro_act_total:.2f}")
    print(f"  Total micro baseline (W·steps): {micro_base_total:.2f}")
    print(f"  Micro saving vs base (%) : {micro_saving:.2f}%")

    print(f"  --- Scheme 3 (macros frozen, no handoff cost) ---")
    macro_const_total = df['actual_power_macro_const_W'].sum()
    base_total = df['baseline_power_W'].sum()
    macro_const_saving = (1 - macro_const_total / base_total) * 100 if base_total > 0 else 0.0
    print(f"  Total power (macros frozen) (W·steps) : {macro_const_total:.2f}")
    print(f"  Saving vs full baseline (%)           : {macro_const_saving:.2f}%")
    print(f"{'='*50}\n")

# Main
def main():
    parser = argparse.ArgumentParser(description="RAN CSO Simulator")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["always_on", "threshold", "random", "cql", "iql", "dqn", "qr_cql"],
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
        policy_fn = load_dqn_policy("models/dqn_penalty_42.zip")

    elif args.policy == "qr_cql":
        from policies import load_qr_cql_policy
        policy_fn = load_qr_cql_policy("models/qr_cql_policy_42")  # <-- change this to the correct seed if needed

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