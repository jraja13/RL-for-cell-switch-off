"""
RAN Cell Switch-Off Gym Environment

Loads PRB and MR traffic data directly from CSVs.
Infers cell type from column name prefix (M = macro, C = micro).

State vector (per timestep):
    For each of the 46 cells, 7 features:
        [prb_load, mr_load, is_on, is_macro, distance_to_macro, macro_prb, macro_mr]  
    Total: 46 × 7 = 322-dim flat vector

Action vector:
    Binary vector of length n_micro (39)
    1 = cell ON, 0 = cell OFF

Reward:
    Net energy saved (Watts)
    = power consumed if all ON - power consumed with CSO decisions applied
    Macro compensation overhead is implicitly accounted for:
        switched-off micro's PRB load is added to its parent macro
        before computing macro power via Auer model.
"""

import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# REPLACE the entire POWER_PARAMS dict and auer_power function with:

POWER_PARAMS = {
    "macro": {"n_trx": 6, "p_max": 20.0, "p0": 130.0, "delta_p": 4.7, "p_sleep": 75.0},
    "micro": {"n_trx": 2, "p_max": 6.3,  "p0": 56.0,  "delta_p": 2.6, "p_sleep": 39.0},
}

SECTOR_COUNT = {"macro": 3, "micro": 1}

def auer_power(load: float, cell_type: str, is_on: bool = True) -> float:
    """
    Auer et al. (2011) EARTH linear power model, Eq. 1:
        P_in = N_TRX * (P0 + delta_p * P_out)   for 0 < P_out <= P_max
        P_in = N_TRX * P_sleep                  when P_out = 0 (cell off/sleep)

    load: P_out/P_max fraction in [0, 1]
    is_on: whether the cell is actively transmitting or in sleep mode
    """
    p = POWER_PARAMS[cell_type]
    if not is_on or load <= 0:
        return p["n_trx"] * p["p_sleep"]
    return p["n_trx"] * (p["p0"] + p["delta_p"] * load)

class RANEnv(gym.Env):
    """
    Parameters
    prb_csv  : path to normalised PRB csv  (rows=timesteps, cols=cells)
    mr_csv   : path to normalised MR csv   (rows=timesteps, cols=cells)
    start    : first timestep index to use (inclusive)
    end      : last  timestep index to use (inclusive)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        prb_csv: str = "prb.csv",
        mr_csv:  str = "mr.csv",
        start:   int = 0,
        end:     int = None,
    ):
        super().__init__()

        # Load traffic data
        prb_full = pd.read_csv("Datasets/Base/processed_cell_PRB.csv").astype(np.float32)
        mr_full  = pd.read_csv("Datasets/Base/processed_cell_MR.csv").astype(np.float32)

        assert list(prb_full.columns) == list(mr_full.columns), \
            "PRB and MR csvs must have identical column order"

        # Clip to [0, 1]
        prb_full = prb_full.clip(0.0, 1.0)
        mr_full  = mr_full.clip(0.0, 1.0)

        # Apply timestep slice 
        total_T = len(prb_full)
        if end is None:
            end = total_T - 1

        assert 0 <= start < total_T, f"start={start} out of range [0, {total_T-1}]"
        assert start <= end < total_T, f"end={end} out of range [start, {total_T-1}]"

        self.prb = prb_full.iloc[start:end+1].values   # (T, 46)
        self.mr  = mr_full.iloc[start:end+1].values    # (T, 46)
        self.T   = self.prb.shape[0]

        # Infer cell types from column names
        self.cell_ids  = list(prb_full.columns)        # e.g. ['M1','M2',...,'C2','C3',...]
        self.n_cells   = len(self.cell_ids)            # 46

        print("Cell IDs from CSV:", self.cell_ids[:10])

        self.cell_types = [
            "macro" if cid.startswith("M") else "micro"
            for cid in self.cell_ids
        ]

        # Separate indices
        self.macro_indices = [i for i, t in enumerate(self.cell_types) if t == "macro"]
        self.micro_indices = [i for i, t in enumerate(self.cell_types) if t == "micro"]
        self.n_macro = len(self.macro_indices)   # 7
        self.n_micro = len(self.micro_indices)   # 39

        # Map each micro → its parent macro index 
        # Load from network_config.json for macro_id lookup
        with open("network_config.json") as f:
            config = json.load(f)

        print("Cell IDs from config:", [c["cell_id"] for c in config["cells"]][:10])
        macro_id_to_idx = {
            c["cell_id"]: self.cell_ids.index(c["cell_id"])
            for c in config["cells"] if c["cell_type"] == "macro"
        }

        self.micro_to_macro_idx = []   # length 39, each entry is index in 46-cell array
        for c in config["cells"]:
            if c["cell_type"] == "micro":
                self.micro_to_macro_idx.append(macro_id_to_idx[c["macro_id"]])

        # Precompute normalised distance to parent macro
        coords = pd.read_csv("Datasets/Opencellid/final_topology.csv", index_col=0)
        # ensure same order as self.cell_ids
        coords = coords.loc[self.cell_ids]

        distances = np.zeros(self.n_cells, dtype=np.float32)
        for j, micro_idx in enumerate(self.micro_indices):
            macro_idx = self.micro_to_macro_idx[j]
            lat1, lon1 = coords.iloc[micro_idx][["Latitude", "Longitude"]]
            lat2, lon2 = coords.iloc[macro_idx][["Latitude", "Longitude"]]
            distances[micro_idx] = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

        # normalise to [0, 1]
        max_d = distances.max()
        if max_d > 0:
            distances /= max_d
        self.distances = distances  # macro cells stay 0


        #  Gym spaces 
        n_features = self.n_cells * 7   # 7 features 
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_features,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.n_micro)

        # Static is_macro feature (never changes) 
        self.is_macro_feat = np.array(
            [1.0 if t == "macro" else 0.0 for t in self.cell_types],
            dtype=np.float32
        )

        self.micro_pos = {idx: j for j, idx in enumerate(self.micro_indices)}

        # Episode state 
        self.t         = 0
        self.on_status = np.ones(self.n_cells, dtype=np.float32)  # all ON at start

        print(f"RANEnv ready — {self.T} timesteps "
              f"({self.n_macro} macros, {self.n_micro} micros)")

    # Internal helpers

    def _build_obs(self) -> np.ndarray:
        prb = self.prb[self.t]        # (46,)
        mr  = self.mr[self.t]         # (46,)

        # macro PRB and MR for each cell (look up parent macro)
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

        obs = np.stack(
            [prb, mr, self.on_status, self.is_macro_feat,
            self.distances, macro_prb, macro_mr],
            axis=1
        ).flatten()   # (322,)

        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple:
        """
        Reward = baseline_power - actual_power

        Baseline : all 46 cells ON at current PRB load
        Actual   : micros OFF contribute 0 power;
                   their PRB load is added to their parent macro
                   before computing macro power
        """
        prb = self.prb[self.t]   # (46,)

        #  Baseline: everything ON 
        baseline_power = sum(
            auer_power(prb[i], self.cell_types[i], is_on=True)
            for i in range(self.n_cells)
        )

        # Extra load pushed onto each macro by switched-off micros
        macro_extra = np.zeros(self.n_cells, dtype=np.float32)
        sector_scale = SECTOR_COUNT["micro"] / SECTOR_COUNT["macro"]   # 1/3

        for j, micro_idx in enumerate(self.micro_indices):
            if action[j] == 0:   # this micro is OFF
                macro_idx = self.micro_to_macro_idx[j]
                macro_extra[macro_idx] += prb[micro_idx] * sector_scale

        # NEW
        actual_power = 0.0
        for i, cell_type in enumerate(self.cell_types):
            if cell_type == "macro":
                total_load = float(prb[i]) + float(macro_extra[i])
                total_load = min(total_load, 1.0)
                actual_power += auer_power(total_load, "macro", is_on=True)
            else:
                j = self.micro_pos[i]
                is_micro_on = bool(action[j] == 1)
                actual_power += auer_power(float(prb[i]), "micro", is_on=is_micro_on)
                # is_on=False → uses p_sleep instead of 0

        reward = baseline_power - actual_power

        info = {
            "timestep":         self.t,
            "baseline_power_W": round(baseline_power, 4),
            "actual_power_W":   round(actual_power,   4),
            "reward_W":         round(reward,          4),
            "n_cells_off":      int(np.sum(action == 0)),
        }

        return float(reward), info

    # Gym API

    def reset(self) -> np.ndarray:
        self.t = 0
        self.on_status = np.ones(self.n_cells, dtype=np.float32)
        return self._build_obs()

    def step(self, action: np.ndarray):
        """
        action : binary array of length n_micro (39)
                 1 = ON, 0 = OFF
        """
        action = np.array(action, dtype=np.int32)
        assert len(action) == self.n_micro, \
            f"Action must be length {self.n_micro}, got {len(action)}"

        # Update on_status for micro cells
        for j, micro_idx in enumerate(self.micro_indices):
            self.on_status[micro_idx] = float(action[j])

        reward, info = self._compute_reward(action)

        self.t += 1
        done = self.t >= self.T

        obs = self._build_obs() if not done else np.zeros(
            self.n_cells * 7, dtype=np.float32
        )

        return obs, reward, done, info

    def render(self, mode="human"):
        prb = self.prb[self.t - 1]
        n_off = int(np.sum(self.on_status[self.micro_indices] == 0))
        print(f"t={self.t-1:4d} | cells_off={n_off:2d} | "
              f"mean_micro_PRB={prb[self.micro_indices].mean():.3f}")