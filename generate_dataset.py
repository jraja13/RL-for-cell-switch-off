"""
Generates an offline dataset for CQL training by rolling out 20
behaviour policies on the training slice (timesteps 0-663).

Behaviour policies :
    PRB-only threshold  : (0.05 to 0.50, step 0.05)
    Dual threshold      :  (PRB + MR conditions)
    Macro-aware         :  (PRB + macro headroom check)
    Distance-weighted   :  (PRB + distance to macro)

Total: 20 x 663 timesteps x 39 micros = ~517,140 transitions

Per-cell transition format:
    obs      : (7,)  float32
    action   : (1,)  float32 — 0.0 or 1.0
    reward   : float32       — exact per-cell reward
    next_obs : (7,)  float32
    terminal : float32       — 1.0 at episode end, else 0.0

Output:
    dataset/offline_dataset.h5
    dataset/offline_dataset_stats.json

Usage:
    python generate_dataset.py
"""

import os
import json
import numpy as np
from simulator import RANEnv
from policies import get_all_behaviour_policies

TRAIN_START = 0
TRAIN_END   = 663
N_FEATURES  = 7
OUTPUT_DIR  = "dataset"
SEED        = 42


def rollout_policy(policy_fn, policy_name: str) -> dict:
    """Runs one policy on the training slice, returns per-cell transitions."""
    env = RANEnv(start=TRAIN_START, end=TRAIN_END)
    obs, _ = env.reset()

    all_obs, all_actions, all_rewards = [], [], []
    all_next_obs, all_terminals = [], []

    done = False
    while not done:
        action = policy_fn(obs, env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        per_cell_rewards = info["per_cell_rewards"]
        final_action     = info["action"]
        terminal_flag    = 1.0 if done else 0.0

        for j, micro_idx in enumerate(env.micro_indices):
            start_idx = micro_idx * N_FEATURES
            end_idx   = start_idx + N_FEATURES

            all_obs.append(obs[start_idx:end_idx].astype(np.float32))
            act_val = 1.0 if float(final_action[j]) == 1.0 else 0.0
            all_actions.append(act_val)
            all_rewards.append(float(per_cell_rewards[j]))
            all_next_obs.append(next_obs[start_idx:end_idx].astype(np.float32))
            all_terminals.append(terminal_flag)

        obs = next_obs

    print(f"  {policy_name}: {len(all_obs)} transitions")

    return {
        "observations": np.array(all_obs,      dtype=np.float32),
        "actions":      np.array(all_actions,  dtype=np.float32).reshape(-1, 1),
        "rewards":      np.array(all_rewards,  dtype=np.float32),
        "next_obs":     np.array(all_next_obs, dtype=np.float32),
        "terminals":    np.array(all_terminals,dtype=np.float32),
    }


def main():
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    policies = get_all_behaviour_policies()
    print(f"Generating offline dataset from {len(policies)} behaviour policies...\n")

    all_obs, all_actions, all_rewards = [], [], []
    all_next_obs, all_terminals = [], []

    for p in policies:
        data = rollout_policy(p["fn"], p["name"])
        all_obs.append(data["observations"])
        all_actions.append(data["actions"])
        all_rewards.append(data["rewards"])
        all_next_obs.append(data["next_obs"])
        all_terminals.append(data["terminals"])

    observations = np.concatenate(all_obs,       axis=0)
    actions      = np.concatenate(all_actions,   axis=0)
    rewards      = np.concatenate(all_rewards,   axis=0)
    next_obs     = np.concatenate(all_next_obs,  axis=0)
    terminals    = np.concatenate(all_terminals, axis=0)

    N = len(observations)
    print(f"\nTotal transitions  : {N:,}")
    print(f"Observations shape : {observations.shape}")
    print(f"Actions shape      : {actions.shape}")
    print(f"Reward mean        : {rewards.mean():.4f}")
    print(f"Reward std         : {rewards.std():.4f}")
    print(f"Action dist        : {np.mean(actions == 1.0)*100:.1f}% ON  "
          f"{np.mean(actions == 0.0)*100:.1f}% OFF")

    # Save as d3rlpy dataset
    try:
        import d3rlpy
        from d3rlpy.dataset import MDPDataset

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )

        dataset_path = os.path.join(OUTPUT_DIR, "offline_dataset.h5")
        with open(dataset_path, "w+b") as f:
            dataset.dump(f)
        print(f"\nDataset saved → {dataset_path}")

    except ImportError:
        print("\nd3rlpy not found — saving as .npz instead.")
        np_path = os.path.join(OUTPUT_DIR, "offline_dataset.npz")
        np.savez(np_path, observations=observations, actions=actions,
                 rewards=rewards, next_obs=next_obs, terminals=terminals)
        print(f"Dataset saved → {np_path}")

    # Save stats
    stats = {
        "n_transitions":  int(N),
        "n_policies":     len(policies),
        "policy_names":   [p["name"] for p in policies],
        "obs_shape":      list(observations.shape),
        "action_shape":   list(actions.shape),
        "reward_mean":    float(rewards.mean()),
        "reward_std":     float(rewards.std()),
        "reward_min":     float(rewards.min()),
        "reward_max":     float(rewards.max()),
        "pct_action_on":  float(np.mean(actions == 1.0) * 100),
        "pct_action_off": float(np.mean(actions == 0.0) * 100),
        "seed":           SEED,
    }

    stats_path = os.path.join(OUTPUT_DIR, "offline_dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved  → {stats_path}")


if __name__ == "__main__":
    main()