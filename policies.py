"""
All policies share the same signature:

    action = policy_fn(obs, env)

    obs : np.ndarray (322,)  — current state vector
    env : RANEnv             — env reference (policies can inspect
                               env.prb, env.mr, env.t, env.n_micro etc)

    returns: np.ndarray (n_micro,) binary, 1=ON 0=OFF

Rule-based policies:
    - always_on_policy
    - random_policy
    - make_threshold_policy(threshold)
    - make_dual_threshold_policy(prb_threshold, mr_threshold)
    - make_macro_aware_policy(prb_threshold, macro_headroom)
    - make_distance_policy(prb_threshold, max_distance)
    - get_all_behaviour_policies()  ← returns all policies for dataset generation

RL policies:
    - load_cql_policy()
    - load_dqn_policy()
    - load_iql_policy()
"""

import numpy as np

# Always-On
def always_on_policy(obs: np.ndarray, env) -> np.ndarray:
    """All micro cells stay ON every timestep. Baseline — max power."""
    return np.ones(env.n_micro, dtype=np.int32)


# Random
def random_policy(obs: np.ndarray, env) -> np.ndarray:
    """Random binary action per micro cell."""
    return env.action_space.sample()


#  PRB-only threshold
def make_threshold_policy(threshold: float = 0.3):
    """
    Switch off micro if PRB load < threshold.
    Threshold in [0, 1].
    """
    def threshold_policy(obs: np.ndarray, env) -> np.ndarray:
        prb_now = env.prb[env.t]
        action  = np.ones(env.n_micro, dtype=np.int32)
        for j, micro_idx in enumerate(env.micro_indices):
            if prb_now[micro_idx] < threshold:
                action[j] = 0
        return action

    threshold_policy.__name__ = f"threshold_{threshold}"
    return threshold_policy


# Dual threshold (PRB AND MR) 
def make_dual_threshold_policy(prb_threshold: float, mr_threshold: float):
    """
    Switch off micro only if BOTH PRB < prb_threshold AND MR < mr_threshold.
    More conservative than PRB-only — requires both load and user count low.
    """
    def dual_policy(obs: np.ndarray, env) -> np.ndarray:
        prb_now = env.prb[env.t]
        mr_now  = env.mr[env.t]
        action  = np.ones(env.n_micro, dtype=np.int32)
        for j, micro_idx in enumerate(env.micro_indices):
            if (prb_now[micro_idx] < prb_threshold and
                    mr_now[micro_idx] < mr_threshold):
                action[j] = 0
        return action

    dual_policy.__name__ = f"dual_prb{prb_threshold}_mr{mr_threshold}"
    return dual_policy


# Macro-aware threshold
def make_macro_aware_policy(prb_threshold: float, macro_headroom: float):
    """
    Switch off micro only if:
        - micro PRB < prb_threshold      (micro is lightly loaded)
        - macro PRB < macro_headroom     (macro has capacity to absorb)
    Directly checks whether parent macro can safely absorb the handoff.
    """

    def macro_aware_policy(obs: np.ndarray, env) -> np.ndarray:
        prb_now = env.prb[env.t]
        action  = np.ones(env.n_micro, dtype=np.int32)
        for j, micro_idx in enumerate(env.micro_indices):
            macro_idx = env.micro_to_macro_idx[j]
            if (prb_now[micro_idx] < prb_threshold and
                    prb_now[macro_idx] < macro_headroom):
                action[j] = 0
        return action

    macro_aware_policy.__name__ = f"macro_aware_prb{prb_threshold}_head{macro_headroom}"
    return macro_aware_policy

# Distance-weighted threshold
def make_distance_policy(prb_threshold: float, max_distance: float):
    """
    Switch off micro only if:
        - micro PRB < prb_threshold      (micro is lightly loaded)
        - normalised distance < max_distance  (close to its macro)

    Physically motivated: far micros harder for macro to cover after
    switch-off due to higher path loss. Only switch off close micros.
    """

    def distance_policy(obs: np.ndarray, env) -> np.ndarray:
        prb_now = env.prb[env.t]
        action  = np.ones(env.n_micro, dtype=np.int32)
        for j, micro_idx in enumerate(env.micro_indices):
            if (prb_now[micro_idx] < prb_threshold and
                    env.distances[micro_idx] < max_distance):
                action[j] = 0
        return action

    distance_policy.__name__ = f"distance_prb{prb_threshold}_dist{max_distance}"
    return distance_policy

# Behaviour policy registry (20 policies) 
def get_all_behaviour_policies() -> list:
    """
    Returns all behaviour policies for offline dataset generation.

    Categories:
        PRB-only threshold (0.05 to 0.50, step 0.05)
        Dual threshold (PRB + MR conditions)
        Macro-aware  (PRB + macro headroom)
        Distance-weighted  (PRB + distance to macro)

    """
    policies = []

    # PRB-only 
    for thresh in [round(x * 0.05, 2) for x in range(1, 20)]:
        policies.append({
            "name": f"threshold_{thresh}",
            "fn":   make_threshold_policy(threshold=thresh),
        })

    # Dual threshold 
    dual_configs = [
        (0.2, 0.2),   
        (0.3, 0.3),   
        (0.4, 0.4),
        (0.5, 0.5),
        (0.6, 0.6),
        (0.7, 0.7),  
        (0.2, 0.3),
        (0.2, 0.4),
        (0.3, 0.2),
        (0.3, 0.4),
        (0.4, 0.2),
        (0.4, 0.3),
        (0.5, 0.3),
        (0.5, 0.4),
        (0.6, 0.4),
        (0.6, 0.5),
        (0.7, 0.5),
        (0.7, 0.6),
        (0.7, 0.6),
    ]
    for prb_t, mr_t in dual_configs:
        policies.append({
            "name": f"dual_prb{prb_t}_mr{mr_t}",
            "fn":   make_dual_threshold_policy(
                        prb_threshold=prb_t, mr_threshold=mr_t),
        })

    # Macro-aware 
    macro_configs = [
        (0.2, 0.7),
        (0.2, 0.8),   
        (0.2, 0.9),   
        (0.3, 0.7),   
        (0.3, 0.8),  
        (0.3, 0.9), 
        (0.4, 0.7),   
        (0.4, 0.8),  
        (0.4, 0.9),
        (0.5, 0.7),
        (0.5, 0.8),
        (0.5, 0.9),
        (0.6, 0.7),
        (0.6, 0.8),
        (0.6, 0.9),
        (0.7, 0.7),
        (0.7, 0.8),
        (0.7, 0.9),
    ]
    for prb_t, headroom in macro_configs:
        policies.append({
            "name": f"macro_aware_prb{prb_t}_head{headroom}",
            "fn":   make_macro_aware_policy(
                        prb_threshold=prb_t, macro_headroom=headroom),
        })

    # Distance-weighted
    distance_configs = [
        (0.3, 0.9), 
        (0.4, 0.9),   
        (0.3, 0.8),
        (0.4, 0.8),  
        (0.3, 0.7),   
        (0.4, 0.7),  
    ]
    for prb_t, max_d in distance_configs:
        policies.append({
            "name": f"distance_prb{prb_t}_dist{max_d}",
            "fn":   make_distance_policy(
                        prb_threshold=prb_t, max_distance=max_d),
        })
    return policies

# CQL
def load_cql_policy(model_path: str = "models/cql_policy"):
    """
    Loads a trained CQL policy from disk (d3rlpy saved directory).
    Returns a policy_fn compatible with run_episode().

    Operates per-cell: loops through all 39 micros, slices each
    cell's 7-dim obs from the full 322-dim state, predicts individually,
    assembles into joint 39-dim action vector.
    """
    try:
        import d3rlpy
        policy = d3rlpy.load_learnable(model_path)
        N_FEATURES_PER_CELL = 7

        def cql_policy(obs: np.ndarray, env) -> np.ndarray:
            action = np.zeros(env.n_micro, dtype=np.int32)
            for j, micro_idx in enumerate(env.micro_indices):
                start_idx = micro_idx * N_FEATURES_PER_CELL
                end_idx   = start_idx + N_FEATURES_PER_CELL
                cell_obs  = obs[start_idx:end_idx].astype(np.float32).reshape(1, -1)
                cell_action = policy.predict(cell_obs)[0]
                action[j]   = int(cell_action)
            return action

        print(f"CQL policy loaded from {model_path}")
        return cql_policy

    except Exception as e:
        raise RuntimeError(f"Could not load CQL policy: {e}")

def load_iql_policy(model_path: str = "models/iql_policy"):
    try:
        import d3rlpy
        policy = d3rlpy.load_learnable(model_path)
        N_FEATURES_PER_CELL = 7

        def iql_policy(obs: np.ndarray, env) -> np.ndarray:
            # Must be float32 to hold continuous proxy values (-1.0 to 1.0)
            action = np.zeros(env.n_micro, dtype=np.float32) 
            
            for j, micro_idx in enumerate(env.micro_indices):
                start_idx = micro_idx * N_FEATURES_PER_CELL
                end_idx   = start_idx + N_FEATURES_PER_CELL
                cell_obs  = obs[start_idx:end_idx].astype(np.float32).reshape(1, -1)
                
                # Predict returns a 1D continuous array [value], extract the scalar float
                cell_action = policy.predict(cell_obs).item()
                action[j]   = cell_action 
                
            return action # Returns the 39-dim continuous float array to RANEnv

        print(f"IQL policy loaded from {model_path}")
        return iql_policy
    except Exception as e:
        raise RuntimeError(f"Could not load IQL policy: {e}")

# DQN
def load_dqn_policy(model_path: str = "models/dqn_policy.zip"):
    """
    Loads a trained per-cell DQN policy from disk (SB3 .zip format).
    Returns a policy_fn compatible with run_episode().
    """
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