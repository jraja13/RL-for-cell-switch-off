"""
Trains a per-cell DQN policy on the CSO problem with a MANUAL training loop.

Usage:
    python train_dqn.py

Output:
    models/dqn_policy.zip  — saved SB3 DQN, picked up by main.py --policy dqn
"""

import os
import csv
import numpy as np
import torch
import random
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import polyak_update, LinearSchedule, configure_logger
import gymnasium as gym
from gymnasium import spaces

from simulator import RANEnv

TRAIN_START = 0
TRAIN_END = 663                      # inclusive → env.T = 664 timesteps per pass
TOTAL_DECISIONS = 800_000           
MODEL_SAVE_PATH = "models/dqn_policy.zip"

# DQN hyperparameters (unchanged from original)
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
LEARNING_STARTS = 1_000
BATCH_SIZE = 64
GAMMA = 0.99
TRAIN_FREQ = 4                       # 1 gradient step per 4 collected decisions
TARGET_UPDATE_INTERVAL = 1_000       # collected decisions between target syncs
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05
EXPLORATION_FRACTION = 0.3          # spend first 30% of training annealing eps
NET_ARCH = [64, 64]

# Per-cell decision is binary: 0 = request OFF, 1 = request ON.
# (RANEnv's JOINT action space is MultiBinary(39); this constant is the
# per-cell DQN action space size, NOT env.action_space.n which would be 39.)
N_ACTIONS = 2
N_FEATURES_PER_CELL = 7
LOG_EVERY = 5_000                    # decisions between progress prints
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class _DummyCellEnv(gym.Env):
    """Throwaway env with the right spaces, only to construct the DQN model.
    The real data env (RANEnv) is driven manually in the training loop."""

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(N_FEATURES_PER_CELL,), dtype=np.float32)
    action_space = spaces.Discrete(2)  # 0 = request OFF, 1 = request ON

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def build_model():
    """Construct the SB3 DQN over a dummy vec env. We never call .learn() on it."""
    vec_env = DummyVecEnv([lambda: _DummyCellEnv()])
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,   # informational only — we gate manually
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        exploration_fraction=EXPLORATION_FRACTION,
        policy_kwargs=dict(net_arch=NET_ARCH),
        verbose=0,
        seed=SEED,
    )
    # We bypass model.learn(), which normally configures the logger (and lr
    # schedule) during its setup phase. Replicate the logger init so that
    # model.train() can record the learning-rate and dump logs without an
    # AttributeError on the (un-initialised) _logger.
    model._logger = configure_logger(model.verbose, model.tensorboard_log, "DQN", True)
    return model


def slice_cell_obs(full_obs, micro_idx):
    """Slice the 7-dim observation for one micro cell out of the 322-dim state."""
    start = micro_idx * N_FEATURES_PER_CELL
    end = start + N_FEATURES_PER_CELL
    return full_obs[start:end]


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # The REAL data environment — joint 39-dim action, returns per_cell_rewards.
    env = RANEnv(
        prb_csv="Datasets/Base/processed_cell_PRB.csv",
        mr_csv="Datasets/Base/processed_cell_MR.csv",
        start=TRAIN_START,
        end=TRAIN_END,
    )
    n_micro = env.n_micro
    micro_indices = env.micro_indices
    decisions_per_pass = env.T * n_micro

    model = build_model()
    device = model.device

    # Epsilon schedule exactly as SB3 DQN would use internally.
    exploration_schedule = LinearSchedule(
        EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_FRACTION
    )
    rng = np.random.default_rng(SEED)

    # Counters / accumulators (use thresholds, not modulo — we collect 39 at a time).
    collected = 0
    steps_since_train = 0
    collected_since_target = 0
    ep_reward_window = []  # mean per-cell reward per simulator timestep, for logging

    full_obs, _ = env.reset()

    print(f"Training DQN for {TOTAL_DECISIONS:,} decisions "
          f"(~{TOTAL_DECISIONS / decisions_per_pass:.1f} passes over "
          f"{decisions_per_pass:,} decisions/pass, {env.T} timesteps x {n_micro} micros)...\n")

    log_path = "logs/dqn_train_progress.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["decisions", "eps", "buffer_size", "mean_reward_window",
             "pct_off_requested", "passes"]
        )

    while collected < TOTAL_DECISIONS:
        # Build all 39 per-cell observations for this timestep
        cell_obs_batch = np.empty((n_micro, N_FEATURES_PER_CELL), dtype=np.float32)
        for j, micro_idx in enumerate(micro_indices):
            cell_obs_batch[j] = slice_cell_obs(full_obs, micro_idx)

        # Batched epsilon-greedy over all 39 cells
        with torch.no_grad():
            obs_tensor = torch.as_tensor(cell_obs_batch, dtype=torch.float32, device=device)
            q_values = model.q_net(obs_tensor)                       # (39, 2)
            greedy = q_values.argmax(dim=1).cpu().numpy()             # (39,)

        progress = max(0.0, 1.0 - collected / TOTAL_DECISIONS)
        model._current_progress_remaining = progress
        eps = float(exploration_schedule(progress))
        model.exploration_rate = eps

        random_mask = rng.random(n_micro) < eps
        random_actions = rng.integers(0, N_ACTIONS, size=n_micro)
        requested_actions = np.where(random_mask, random_actions, greedy).astype(np.int32)

        # Step the REAL simulator once with the full joint action
        next_full_obs, _timestep_reward, terminated, truncated, info = env.step(requested_actions)
        done = bool(terminated or truncated)
        per_cell_rewards = np.asarray(info["per_cell_rewards"], dtype=np.float32)  # (39,)

        #  CORRECTLY-PAIRED transitions into the replay buffer
        for j, micro_idx in enumerate(micro_indices):
            next_cell_obs = slice_cell_obs(next_full_obs, micro_idx)
            model.replay_buffer.add(
                cell_obs_batch[j].reshape(1, N_FEATURES_PER_CELL),                  # obs  (1, 7)
                next_cell_obs.reshape(1, N_FEATURES_PER_CELL),                       # next (1, 7)
                np.array([[int(requested_actions[j])]], dtype=np.int64),             # action (1, 1)
                np.array([float(per_cell_rewards[j])], dtype=np.float32),           # reward (1,)
                np.array([done], dtype=bool),                                        # done   (1,)
                [{}],                                                                # infos  (len 1)
            )
            collected += 1
            # Only count toward train/target cadence AFTER the warmup window,
            # so we don't backfill a burst of gradient steps the moment
            # learning starts (matches SB3 learning_starts semantics).
            if collected > LEARNING_STARTS:
                steps_since_train += 1
                collected_since_target += 1

        # Logging window: mean per-cell reward this simulator timestep.
        ep_reward_window.append(float(per_cell_rewards.mean()))

        # Gradient updates: catch up on train_freq
        if collected > LEARNING_STARTS:
            n_grad = steps_since_train // TRAIN_FREQ
            steps_since_train = steps_since_train % TRAIN_FREQ
            if n_grad > 0:
                # Refresh progress with the post-add count so the LR schedule
                # and any SB3-internal progress reads see the latest value.
                model.num_timesteps = collected
                model._current_progress_remaining = max(
                    0.0, 1.0 - collected / TOTAL_DECISIONS
                )
                model.train(gradient_steps=n_grad, batch_size=BATCH_SIZE)

            # ── 6. Target network sync (DQN.train() does NOT do this) ─────
            # while-subtract keeps cadence from drifting across the 39-at-a-
            # time collection boundaries.
            while collected_since_target >= TARGET_UPDATE_INTERVAL:
                collected_since_target -= TARGET_UPDATE_INTERVAL
                polyak_update(
                    model.q_net.parameters(),
                    model.q_net_target.parameters(),
                    model.tau,
                )

        # Advance state / reset at episode end
        if done:
            full_obs, _ = env.reset()
        else:
            full_obs = next_full_obs

        # Progress logging
        if collected % LOG_EVERY < n_micro or collected >= TOTAL_DECISIONS:
            recent = ep_reward_window[-min(len(ep_reward_window), 200):]
            mean_r = float(np.mean(recent)) if recent else 0.0
            pct_off = float(np.mean(requested_actions == 0)) * 100
            passes = collected / decisions_per_pass
            print(f"decisions={collected:>8,}/{TOTAL_DECISIONS:,} | eps={eps:.3f} | "
                  f"buf={model.replay_buffer.size():>6,} | "
                  f"mean_reward={mean_r:+.3f} | off_req={pct_off:4.1f}% | passes={passes:.2f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [collected, f"{eps:.4f}", model.replay_buffer.size(),
                     f"{mean_r:.4f}", f"{pct_off:.2f}", f"{passes:.4f}"]
                )
    
    # Save 
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")
    print(f"Progress log → {log_path}")
    print("Next: python main.py --policy dqn --split eval")

if __name__ == "__main__":
    main()