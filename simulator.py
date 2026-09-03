import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# # Auer et al power model 
# POWER_PARAMS = {
#     "macro": {"n_trx": 6, "p_max": 20.0, "p0": 130.0, "delta_p": 4.7, "p_sleep": 75.0},
#     "micro": {"n_trx": 2, "p_max": 6.3,  "p0": 56.0,  "delta_p": 2.6, "p_sleep": 39.0},
# }

# different vendor's power model
POWER_PARAMS = {
    "macro": {"n_trx": 1, "p_max": 379.0, "p0": 365.8, "delta_p": 13.2, "p_sleep": 379.0},
    "micro": {"n_trx": 1, "p_max": 172.0, "p0": 164.2, "delta_p": 7.63, "p_sleep": 8.0},
}

# Sector counts 
SECTOR_COUNT = {"macro": 3, "micro": 1}
SECTOR_SCALE = SECTOR_COUNT["micro"] / SECTOR_COUNT["macro"] 

MICRO_CSO_LIMIT    = 0.3  # max PRB load at which a micro is eligible for switch-off
MACRO_CAPACITY_LIMIT = 0.8 # max PRB load at which a macro is eligible to accept transferred load
MACRO_OVERLOAD_THRESHOLD = 0.75  

def auer_power(load: float, cell_type: str, is_on: bool = True) -> float:
    """
    P_in = N_TRX * (P0 + delta_p * load)   for 0 < load <= 1, cell ON
    P_in = N_TRX * P_sleep                  when cell is OFF (sleep mode)
    """
    p = POWER_PARAMS[cell_type]
    if not is_on:
        return p["n_trx"] * p["p_sleep"]
    return p["n_trx"] * (p["p0"] + p["delta_p"] * float(load))

class RANEnv(gym.Env):
    """
    prb_csv  : path to normalised PRB csv  (rows=timesteps, cols=cells)
    mr_csv   : path to normalised MR csv   (rows=timesteps, cols=cells)
    cells_csv: path to cell topology csv (CellID, Latitude, Longitude, Macro)
    start    : first timestep index to use (inclusive)
    end      : last  timestep index to use (inclusive)
    """

    metadata = {"render.modes": []}
    N_FEATURES_PER_CELL = 7   # [prb, mr, is_on, is_macro, distance, macro_prb, macro_mr]

    def __init__(
        self,
        prb_csv:   str = "Datasets/Base/simulator_ready_traffic_PRB.csv",
        mr_csv:    str = "Datasets/Base/simulator_ready_traffic_MR.csv",
        cells_csv: str = "Datasets/Opencellid/final_topology.csv",
        config_path: str = "network_config.json",
        start:     int = 0,
        end:       int = None,
        is_continuous_proxy: bool = False,
    ):
        super().__init__()
        self.is_continuous_proxy = is_continuous_proxy
        # Load traffic data
        prb_full = pd.read_csv(prb_csv).astype(np.float32)
        mr_full  = pd.read_csv(mr_csv).astype(np.float32)

        assert list(prb_full.columns) == list(mr_full.columns), \
            "PRB and MR csvs must have identical column order"

        prb_full = prb_full.clip(0.0, 1.0)
        mr_full  = mr_full.clip(0.0, 1.0)

        total_T = len(prb_full)
        if end is None:
            end = total_T - 1
        assert 0 <= start < total_T, f"start={start} out of range [0, {total_T-1}]"
        assert start <= end < total_T, f"end={end} out of range [start, {total_T-1}]"

        self.prb = prb_full.iloc[start:end + 1].values   # (T, 46)
        self.mr  = mr_full.iloc[start:end + 1].values     # (T, 46)
        self.T   = self.prb.shape[0]

        # Cell identity / type
        self.cell_ids = list(prb_full.columns)            # e.g. ['M2',...,'C1','C2',...]
        self.n_cells  = len(self.cell_ids)                # 46

        self.cell_types = [
            "macro" if cid.startswith("M") else "micro"
            for cid in self.cell_ids
        ]

        self.macro_indices = [i for i, t in enumerate(self.cell_types) if t == "macro"]
        self.micro_indices = [i for i, t in enumerate(self.cell_types) if t == "micro"]
        self.n_macro = len(self.macro_indices)   # 7
        self.n_micro = len(self.micro_indices)   # 39

        # fast lookup: cell index (0..45) -> position within micro_indices (0..38)
        self.micro_pos = {idx: j for j, idx in enumerate(self.micro_indices)}

        # Map each micro -> its parent macro's index in the 46-cell array
        with open(config_path) as f:
            config = json.load(f)

        macro_id_to_idx = {
            c["cell_id"]: self.cell_ids.index(c["cell_id"])
            for c in config["cells"] if c["cell_type"] == "macro"
        }

        self.micro_to_macro_idx = []   # length 39
        for c in config["cells"]:
            if c["cell_type"] == "micro":
                self.micro_to_macro_idx.append(macro_id_to_idx[c["macro_id"]])

        # Precompute normalised distance of each micro to its macro
        coords = pd.read_csv(cells_csv).set_index("CellID")
        coords = coords.loc[self.cell_ids]   # reorder to match cell_ids exactly

        distances = np.zeros(self.n_cells, dtype=np.float32)
        for j, micro_idx in enumerate(self.micro_indices):
            macro_idx = self.micro_to_macro_idx[j]
            lat1, lon1 = coords.iloc[micro_idx][["Latitude", "Longitude"]]
            lat2, lon2 = coords.iloc[macro_idx][["Latitude", "Longitude"]]
            distances[micro_idx] = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

        max_d = distances.max()
        if max_d > 0:
            distances = distances / max_d
        self.distances = distances   # macro cells stay 0.0

        # Gym spaces
        n_features = self.n_cells * self.N_FEATURES_PER_CELL   # 46 * 7 = 322
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_features,), dtype=np.float32
        )

        if self.is_continuous_proxy:
            # For IQL: A continuous box matching the shape of your 39 micro-cells
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(self.n_micro,), 
                dtype=np.float32
            )
        else:
            # For DQN, CQL, and random baseline execution
            self.action_space = spaces.MultiBinary(self.n_micro)

        # Static is_macro feature
        self.is_macro_feat = np.array(
            [1.0 if t == "macro" else 0.0 for t in self.cell_types],
            dtype=np.float32
        )

        # Episode state
        self.t         = 0
        self.on_status = np.ones(self.n_cells, dtype=np.float32)

        print(f"RANEnv ready — {self.T} timesteps "
              f"({self.n_macro} macros, {self.n_micro} micros)")

    def _macro_prb_mr_per_cell(self, prb: np.ndarray, mr: np.ndarray) -> tuple:
        """For every cell, look up its PARENT macro's prb and mr.
        Macro cells report their own values."""
        macro_prb = np.zeros(self.n_cells, dtype=np.float32)
        macro_mr  = np.zeros(self.n_cells, dtype=np.float32)

        for i in range(self.n_cells):
            if self.cell_types[i] == "macro":
                macro_prb[i] = prb[i]
                macro_mr[i]  = mr[i]
            else:
                j = self.micro_pos[i]
                mac_idx = self.micro_to_macro_idx[j]
                macro_prb[i] = prb[mac_idx]
                macro_mr[i]  = mr[mac_idx]

        return macro_prb, macro_mr

    def _build_obs(self) -> np.ndarray:
        """Flatten [prb, mr, is_on, is_macro, distance, macro_prb, macro_mr]
        per cell -> (322,)"""
        prb = self.prb[self.t]
        mr  = self.mr[self.t]
        macro_prb, macro_mr = self._macro_prb_mr_per_cell(prb, mr)

        obs = np.stack(
            [prb, mr, self.on_status, self.is_macro_feat,
             self.distances, macro_prb, macro_mr],
            axis=1
        ).flatten()
        return obs.astype(np.float32)

    def _process_timestep(self, requested_action: np.ndarray) -> dict:
        """
        THE single unified per-cell sequential pass. For each micro
        (in index order), checks macro feasibility against the macro's
        RUNNING committed load (base + already-committed transfers
        from earlier micros under the same macro this timestep), then
        commits or blocks accordingly, and computes that cell's exact
        reward contribution.

        Returns a dict with:
            final_action       : (39,) int array, after feasibility overrides
            per_cell_rewards   : (39,) float array
            n_blocked          : int, count of requested-OFF but forced-ON
            baseline_power_W   : float, all-46-cells-ON power this timestep
            actual_power_W     : float, power with final_action applied
        """
        prb = self.prb[self.t]   # (46,)

        # Baseline: everything ON, all 46 cells
        baseline_power = sum(
            auer_power(prb[i], self.cell_types[i], is_on=True)
            for i in range(self.n_cells)
        )

        # Scheme 1 baseline: all micros ON, macros ignored entirely.
        baseline_power_micro_only = sum(
            auer_power(float(prb[i]), "micro", is_on=True)
            for i in self.micro_indices
        )

        final_action     = requested_action.copy()
        per_cell_rewards = np.zeros(self.n_micro, dtype=np.float32)
        macro_running_extra = np.zeros(self.n_cells, dtype=np.float32)   # committed transfers so far
        n_blocked = 0   # blocked by macro capacity 
        n_blocked_by_micro_threshold = 0  # blocked by MICRO_CSO_LIMIT

        for j, micro_idx in enumerate(self.micro_indices):
            macro_idx       = self.micro_to_macro_idx[j]
            micro_prb       = float(prb[micro_idx])
            macro_base_load = float(prb[macro_idx])

            if requested_action[j] == 1:
                # Cell stays ON — no change vs baseline, no transfer.
                final_action[j]     = 1
                per_cell_rewards[j] = 0.0
                continue

            if micro_prb > MICRO_CSO_LIMIT:
                final_action[j]     = 1
                per_cell_rewards[j] = 0.0
                n_blocked_by_micro_threshold += 1   
                continue

            transferred = micro_prb * SECTOR_SCALE
            projected_load = (
                macro_base_load + macro_running_extra[macro_idx] + transferred
            )

            if projected_load > MACRO_CAPACITY_LIMIT:
                # Not enough room on the macro — force this micro to stay ON
                final_action[j]     = 1
                per_cell_rewards[j] = 0.0
                n_blocked += 1
                continue

            # Feasible — commit the switch-off
            final_action[j] = 0
            macro_running_extra[macro_idx] += transferred

            # Exact marginal reward for this cell (order-independent,
            # due to linearity of the Auer power model):
            own_saving = (
                auer_power(micro_prb, "micro", is_on=True)
                - auer_power(0.0, "micro", is_on=False)   # sleep power
            )
            macro_params = POWER_PARAMS["macro"]
            macro_cost = macro_params["n_trx"] * macro_params["delta_p"] * transferred

            per_cell_rewards[j] = own_saving - macro_cost

        # Actual power: apply final_action, compute true total
        actual_power = 0.0
        actual_power_micro_only = 0.0   # Scheme 1: micro power only, macros ignored
        actual_power_macro_const = 0.0   # Scheme 2: macros frozen at baseline, no handoff cost
        n_macro_overload = 0   # NEW

        for i, cell_type in enumerate(self.cell_types):
            if cell_type == "macro":
                total_load = float(prb[i]) + float(macro_running_extra[i])
                total_load = min(total_load, 1.0)
                if total_load > MACRO_OVERLOAD_THRESHOLD:   # NEW
                    n_macro_overload += 1                    # NEW
                actual_power += auer_power(total_load, "macro", is_on=True)
                actual_power_macro_const += auer_power(float(prb[i]), "macro", is_on=True)
            else:
                j = self.micro_pos[i]
                is_on = bool(final_action[j] == 1)
                micro_p = auer_power(float(prb[i]), "micro", is_on=is_on)
                actual_power += micro_p
                actual_power_micro_only += micro_p
                actual_power_macro_const += micro_p  

        # NEW: per-micro overload flag — was this micro's parent macro overloaded
        # this timestep? Used only for DQN's QoS-penalty reward shaping.
        overload_per_micro = np.zeros(self.n_micro, dtype=np.float32)
        macro_overloaded = {}  # macro_idx -> bool, computed once per macro
        for i in self.macro_indices:
            total_load = float(prb[i]) + float(macro_running_extra[i])
            macro_overloaded[i] = total_load > MACRO_OVERLOAD_THRESHOLD

        for j, micro_idx in enumerate(self.micro_indices):
            macro_idx = self.micro_to_macro_idx[j]
            overload_per_micro[j] = float(macro_overloaded[macro_idx])
                
        return {
            "final_action":      final_action,
            "per_cell_rewards":  per_cell_rewards,
            "n_blocked":         n_blocked,
            "n_blocked_by_micro_threshold": n_blocked_by_micro_threshold,
            "n_macro_overload":  n_macro_overload,   # NEW
            "baseline_power_W":  float(baseline_power),
            "actual_power_W":    float(actual_power),
            "baseline_power_micro_only_W": float(baseline_power_micro_only),
            "actual_power_micro_only_W":   float(actual_power_micro_only),
            "actual_power_macro_const_W":  float(actual_power_macro_const), 
            "overload_per_micro":  overload_per_micro,   # NEW
        }
    
    # openai/gym API
    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.on_status = np.ones(self.n_cells, dtype=np.float32)
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        """
        action : binary array of length n_micro (39), requested ON/OFF.
                 May be partially overridden to ON by the macro capacity
                 feasibility check — see _process_timestep().

        Returns: obs, reward, terminated, truncated, info
            reward = timestep-level total (baseline - actual), unchanged
                     in meaning from earlier versions — used for the
                     cumulative power comparison plot and for any
                     training signal that wants a single scalar.
            info["per_cell_rewards"] = exact (39,) per-cell breakdown,
                     summing to the same value as `reward`.
        """
        action = np.array(action, dtype=np.int32)
        assert len(action) == self.n_micro, \
            f"Action must be length {self.n_micro}, got {len(action)}"

        if self.is_continuous_proxy:
            binary_action = np.where(action > 0.0, 0, 1).astype(np.int32)
        else:
            binary_action = action.astype(np.int32)
        result = self._process_timestep(binary_action)

        final_action = result["final_action"]
        for j, micro_idx in enumerate(self.micro_indices):
            self.on_status[micro_idx] = float(final_action[j])

        reward = result["baseline_power_W"] - result["actual_power_W"]

        info = {
            "timestep":          self.t,
            "baseline_power_W":  round(result["baseline_power_W"], 4),
            "actual_power_W":    round(result["actual_power_W"],   4),
            "baseline_power_micro_only_W": round(result["baseline_power_micro_only_W"], 4),
            "actual_power_micro_only_W":   round(result["actual_power_micro_only_W"],   4),
            "actual_power_macro_const_W": round(result["actual_power_macro_const_W"], 4),
            "reward_W":          round(reward, 4),
            "n_cells_off":       int(np.sum(final_action == 0)),
            "n_blocked_by_capacity": result["n_blocked"],
            "n_blocked_by_micro_threshold": result["n_blocked_by_micro_threshold"],
            "n_macro_overload":  result["n_macro_overload"], 
            "action":            final_action.tolist(),
            "per_cell_rewards":  result["per_cell_rewards"].tolist(),
            "overload_per_micro": result["overload_per_micro"].tolist(),
        }

        self.t += 1
        done = self.t >= self.T

        obs = self._build_obs() if not done else np.zeros(
            self.n_cells * self.N_FEATURES_PER_CELL, dtype=np.float32
        )

        return obs, float(reward), done, False, info

    def render(self, mode="human"):
        prb = self.prb[self.t - 1]
        n_off = int(np.sum(self.on_status[self.micro_indices] == 0))
        print(f"t={self.t-1:4d} | cells_off={n_off:2d} | "
              f"mean_micro_PRB={prb[self.micro_indices].mean():.3f}")