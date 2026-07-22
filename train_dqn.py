"""
Trains a DQN policy on the per-cell decomposed CSO problem.

Run this ONCE to produce models/dqn_policy.zip, which is then picked
up by policies.py's load_dqn_policy() for evaluation via main.py.

Usage:
    python train_dqn.py

Note on total_timesteps:
    Each SB3 "step" here = ONE micro cell decision (not one simulator
    timestep). The training slice has 663 timesteps x 39 micros =
    25,857 decisions per single pass over the data. DQN typically
    needs many passes to learn well, so total_timesteps is set well
    above one pass by default. Increase TOTAL_TIMESTEPS if training
    reward hasn't stabilised (check the printed episode reward trend).
"""

import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from dqn_wrapper import CellWiseDQNEnv
import torch
import random
import numpy as np

# Config
TRAIN_START      = 0
TRAIN_END        = 663
TOTAL_TIMESTEPS  = 400_000   # ~7-8 passes over the 25,857-step training data
MODEL_SAVE_PATH  = "models/dqn_policy.zip"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # Build training environment
    env = CellWiseDQNEnv(
        prb_csv="Datasets/Base/simulator_ready_traffic_PRB.csv",
        mr_csv="Datasets/Base/simulator_ready_traffic_MR.csv",
        start=TRAIN_START,
        end=TRAIN_END,
    )
    env = Monitor(env, filename="logs/dqn_train_monitor.csv")

    # Configure DQN
    # Small network — problem is only 7-dim input, 2 actions, so a
    # large network would just overfit / waste training time.
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.3,     # 30% of training spent exploring
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        tensorboard_log="logs/tensorboard/",
        seed=42
    )

    # Train
    print(f"Training DQN for {TOTAL_TIMESTEPS:,} steps "
          f"(~{TOTAL_TIMESTEPS / (663 * 39):.1f} passes over training data)...\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")
    print("Next: python main.py --policy dqn --split eval")


if __name__ == "__main__":
    main()