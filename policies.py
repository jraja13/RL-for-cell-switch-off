"""
All policies share the same signature:

    action = policy_fn(obs, env)

    obs : np.ndarray (184,)  — current state vector
    env : RANEnv             — env reference (policies can inspect
                               env.prb, env.mr, env.t, env.n_micro etc)

    returns: np.ndarray (n_micro,) binary, 1=ON 0=OFF

Rule-based policies (available now):
    - always_on_policy
    - random_policy
    - make_threshold_policy(threshold)

RL policies (stubs, implemented later):
    - load_iql_policy()
    - load_dqn_policy()
"""

import numpy as np


# Always-On
def always_on_policy(obs: np.ndarray, env) -> np.ndarray:
    """
    All micro cells stay ON every timestep.
    This is the baseline — maximum power consumption.
    """
    return np.ones(env.n_micro, dtype=np.int32)


# Random 
def random_policy(obs: np.ndarray, env) -> np.ndarray:
    """
    Random binary action per micro cell.
    Used to add diversity to the offline dataset.
    """
    return env.action_space.sample()


# Threshold 
def make_threshold_policy(threshold: float = 0.3):
    """
    Factory function — returns a threshold policy with the given PRB threshold.

    Logic:
        For each micro cell:
            if PRB load < threshold → switch OFF (action = 0)
            else                    → keep ON   (action = 1)

    PRB values are read directly from env.prb[env.t] at the current timestep.
    obs is not used — raw PRB is cleaner and more interpretable.

    Args:
        threshold : float in [0, 1], default 0.3

    Returns:
        policy_fn compatible with run_episode()
    """
    def threshold_policy(obs: np.ndarray, env) -> np.ndarray:
        prb_now = env.prb[env.t]   # (46,) current PRB loads
        action  = np.ones(env.n_micro, dtype=np.int32)

        for j, micro_idx in enumerate(env.micro_indices):
            if prb_now[micro_idx] < threshold:
                action[j] = 0   # switch off

        return action

    threshold_policy.__name__ = f"threshold_{threshold}"
    return threshold_policy


# IQL (stub) 
def load_iql_policy(model_path: str = "models/iql_policy.pt"):
    """
    Loads a trained IQL policy from disk.
    Returns a policy_fn compatible with run_episode().

    To be implemented after IQL training with d3rlpy.
    Model is expected to be saved via d3rlpy's save_policy() method.

    Args:
        model_path : path to saved IQL policy

    Returns:
        policy_fn compatible with run_episode()
    """
    try:
        import d3rlpy
        policy = d3rlpy.load_learnable(model_path)

        def iql_policy(obs: np.ndarray, env) -> np.ndarray:
            # d3rlpy expects (1, obs_dim) input
            obs_input    = obs.reshape(1, -1)
            action_raw   = policy.predict(obs_input)[0]   # continuous in [-1, 1]
            # Binarise: > 0 → ON, else → OFF
            action_binary = (action_raw > 0).astype(np.int32)
            return action_binary

        print(f"IQL policy loaded from {model_path}")
        return iql_policy

    except Exception as e:
        raise RuntimeError(f"Could not load IQL policy: {e}")


# DQN (stub)
def load_dqn_policy(model_path: str = "models/dqn_policy.zip"):
    try:
        from stable_baselines3 import DQN

        model = DQN.load(model_path)
        N_FEATURES_PER_CELL = 7

        def dqn_policy(obs: np.ndarray, env) -> np.ndarray:
            action = np.zeros(env.n_micro, dtype=np.int32)

            for j, micro_idx in enumerate(env.micro_indices):
                start_idx = micro_idx * N_FEATURES_PER_CELL
                end_idx   = start_idx + N_FEATURES_PER_CELL
                cell_obs  = obs[start_idx:end_idx].astype(np.float32).reshape(1, -1)

                cell_action, _ = model.predict(cell_obs, deterministic=True)
                action[j] = int(cell_action[0])

            return action

        print(f"DQN policy loaded from {model_path}")
        return dqn_policy

    except Exception as e:
        raise RuntimeError(f"Could not load DQN policy: {e}")