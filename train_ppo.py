"""
Trains a per-cell PPO policy on the CSO problem.

PPO is on-policy: unlike DQN's manual replay-buffer loop, this uses SB3's
native training loop via a custom VecEnv wrapper (RANCellVecEnv) that
presents the 39 micro cells of the SHARED simulator as 39 parallel
single-cell sub-environments to PPO. Internally, only ONE RANEnv.step()
call happens per simulator timestep -- the same joint 39-dim action /
per_cell_rewards mechanism used by DQN, CQL, IQL, and QR-CQL is reused
here, just packaged into vec-env shape (39, obs_dim) instead of a
manual for-loop.

This gives PPO a directly comparable per-cell GAE-based advantage signal:
each of the 39 "sub-envs" accumulates its own trajectory of
(obs, action, reward, done) over an n_steps=664 rollout (one full pass
over the training slice), and SB3 computes GAE independently per
column -- matching the per-cell factorisation used everywhere else in
this thesis, with no custom advantage-aggregation logic required.

Usage:
    python train_ppo.py

Output:
    models/ppo_policy.zip  -- saved SB3 PPO, picked up by main.py --policy ppo
"""

import os
import csv
import random
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from simulator import RANEnv

TRAIN_START = 0
TRAIN_END = 663                      # inclusive -> env.T = 664 timesteps per pass
TOTAL_DECISIONS = 800_000            # matches DQN / CQL / IQL / QR-CQL budget
MODEL_SAVE_PATH = "models/ppo_policy.zip"

# PPO hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 664          # one full pass over the training slice per rollout
BATCH_SIZE = 64        # matches DQN's batch size for comparability
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2       # the "controllability" mechanism -- bounds policy update size
ENT_COEF = 0.0
NET_ARCH = [64, 64]

N_FEATURES_PER_CELL = 7
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class RANCellVecEnv(VecEnv):
    """
    Presents RANEnv's 39 micro cells as 39 parallel single-cell envs to
    PPO. Internally, ONE shared RANEnv.step() call happens per timestep --
    the joint 39-dim action is assembled from all 39 sub-env actions each
    step, and RANEnv's exact per_cell_rewards array is unpacked back out
    as each sub-env's individual reward.
    """

    def __init__(self, ran_env: RANEnv):
        self.env = ran_env
        n_envs = ran_env.n_micro
        obs_space = spaces.Box(low=0.0, high=1.0,
                                shape=(N_FEATURES_PER_CELL,), dtype=np.float32)
        act_space = spaces.Discrete(2)   # 0 = request OFF, 1 = request ON
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None

    def _slice_all_cells(self, full_obs: np.ndarray) -> np.ndarray:
        return np.stack([
            full_obs[idx * N_FEATURES_PER_CELL:(idx + 1) * N_FEATURES_PER_CELL]
            for idx in self.env.micro_indices
        ]).astype(np.float32)

    def reset(self):
        full_obs, _ = self.env.reset()
        return self._slice_all_cells(full_obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int32)

    def step_wait(self):
        next_full_obs, _timestep_reward, terminated, truncated, info = \
            self.env.step(self._actions)
        done = bool(terminated or truncated)

        rewards = np.asarray(info["per_cell_rewards"], dtype=np.float32)   # (39,)
        next_obs = self._slice_all_cells(next_full_obs)
        dones = np.full(self.num_envs, done, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        if done:
            # SB3 VecEnv autoreset convention: infos[i]["terminal_observation"]
            # carries the TRUE terminal obs for correct GAE bootstrapping,
            # while the returned obs is already the reset observation.
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = next_obs[i]
            reset_obs, _ = self.env.reset()
            next_obs = self._slice_all_cells(reset_obs)

        return next_obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("env_method not needed for training")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs


class ProgressLogger(BaseCallback):
    """Mirrors the progress-print / CSV-log style of train_dqn.py, adapted
    to PPO's rollout-based (not per-decision) update cadence."""

    def __init__(self, log_path: str, decisions_per_pass: int, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.decisions_per_pass = decisions_per_pass
        self._rollout_count = 0

        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["decisions", "mean_reward_window", "pct_off_requested",
                 "passes", "rollouts"]
            )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        buf = self.model.rollout_buffer
        mean_r = float(buf.rewards.mean())
        actions_taken = buf.actions.reshape(-1)          # Discrete -> flatten
        pct_off = float(np.mean(actions_taken == 0)) * 100
        decisions = self.num_timesteps
        passes = decisions / self.decisions_per_pass

        print(f"decisions={decisions:>8,}/{TOTAL_DECISIONS:,} | "
              f"rollout={self._rollout_count:>4d} | "
              f"mean_reward={mean_r:+.3f} | off_req={pct_off:4.1f}% | "
              f"passes={passes:.2f}")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [decisions, f"{mean_r:.4f}", f"{pct_off:.2f}", f"{passes:.4f}",
                 self._rollout_count]
            )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = RANEnv(
        prb_csv="Datasets/Base/processed_cell_PRB.csv",
        mr_csv="Datasets/Base/processed_cell_MR.csv",
        start=TRAIN_START,
        end=TRAIN_END,
    )
    n_micro = env.n_micro
    decisions_per_pass = env.T * n_micro
    vec_env = RANCellVecEnv(env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        policy_kwargs=dict(net_arch=NET_ARCH),
        verbose=0,
        seed=SEED,
    )

    log_path = "logs/ppo_train_progress.csv"
    callback = ProgressLogger(log_path, decisions_per_pass)

    print(f"Training PPO for {TOTAL_DECISIONS:,} decisions "
          f"(~{TOTAL_DECISIONS / decisions_per_pass:.1f} passes over "
          f"{decisions_per_pass:,} decisions/pass, {env.T} timesteps x {n_micro} micros)...\n")
    print(f"Rollout size: {N_STEPS} steps x {n_micro} envs = "
          f"{N_STEPS * n_micro:,} decisions/rollout\n")

    model.learn(total_timesteps=TOTAL_DECISIONS, callback=callback)

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved -> {MODEL_SAVE_PATH}")
    print(f"Progress log -> {log_path}")
    print("Next: python main.py --policy ppo --split eval")


if __name__ == "__main__":
    main()