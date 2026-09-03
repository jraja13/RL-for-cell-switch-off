import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import RANEnv


class CellWiseDQNEnv(gym.Env):
    """
    Per-cell wrapper around RANEnv for SB3 DQN training/inference.

    Parameters
    ----------
    prb_csv, mr_csv, cells_csv, config_path : passed straight through to RANEnv
    start, end : timestep slice passed straight through to RANEnv
    """

    metadata = {"render.modes": []}
    N_FEATURES_PER_CELL = 7   # [prb, mr, is_on, is_macro, distance, macro_prb, macro_mr]

    def __init__(
        self,
        prb_csv:     str = "Datasets/Base/simulator_ready_traffic_PRB.csv",
        mr_csv:      str = "Datasets/Base/simulator_ready_traffic_MR.csv",
        cells_csv:   str = "Datasets/Opencellid/final_topology.csv",
        config_path: str = "network_config.json",
        start:       int = 0,
        end:         int = None,
    ):
        super().__init__()

        self.env = RANEnv(
            prb_csv=prb_csv, mr_csv=mr_csv,
            cells_csv=cells_csv, config_path=config_path,
            start=start, end=end,
        )
        self.n_micro = self.env.n_micro   # 39

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.N_FEATURES_PER_CELL,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)   # 0 = OFF (requested), 1 = ON

        # Bookkeeping
        self._full_obs       = None   # current 322-dim obs from RANEnv
        self._micro_cursor   = 0      # which of the 39 micros we're deciding now (0..38)
        self._pending_action = None   # accumulates this timestep's 39-dim action

        # Per-cell rewards from the PREVIOUS completed timestep, indexed
        # by micro position (0..38). Emitted one position at a time as
        # the cursor reaches that position again in the current timestep.
        # All zeros at episode start (no previous timestep exists yet).
        self._prev_per_cell_rewards = np.zeros(self.n_micro, dtype=np.float32)

    #helpers
    def _extract_cell_obs(self, full_obs: np.ndarray, micro_idx: int) -> np.ndarray:
        """Slices the 7 features for a single cell out of RANEnv's flat obs."""
        start_idx = micro_idx * self.N_FEATURES_PER_CELL
        end_idx   = start_idx + self.N_FEATURES_PER_CELL
        return full_obs[start_idx:end_idx].astype(np.float32)

    def _current_micro_cell_index(self) -> int:
        """Maps the cursor (0..38) to the actual index in the 46-cell array."""
        return self.env.micro_indices[self._micro_cursor]

    # openaigym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._full_obs, _    = self.env.reset()
        self._micro_cursor   = 0
        self._pending_action = np.ones(self.n_micro, dtype=np.int32)  # default ON
        self._prev_per_cell_rewards = np.zeros(self.n_micro, dtype=np.float32)

        cell_idx = self._current_micro_cell_index()
        obs = self._extract_cell_obs(self._full_obs, cell_idx)
        return obs, {}

    def step(self, action: int):
        """
        action : 0 or 1, requested decision for the CURRENT micro cell.

        Returns the EXACT reward this same micro position earned in the
        PREVIOUS completed timestep (see module docstring for why this
        one-timestep lag is necessary and exact, not approximate).
        """
        reward_to_emit = float(self._prev_per_cell_rewards[self._micro_cursor])

        self._pending_action[self._micro_cursor] = int(action)
        self._micro_cursor += 1

        if self._micro_cursor < self.n_micro:
            # Still mid-timestep — more micros to decide before we can
            # call the real env.
            cell_idx = self._current_micro_cell_index()
            obs = self._extract_cell_obs(self._full_obs, cell_idx)
            return obs, reward_to_emit, False, False, {}

        # ── All 39 micros decided — advance the real simulator ───────
        next_full_obs, _timestep_reward, terminated, truncated, info = \
            self.env.step(self._pending_action)

        # Store this timestep's exact per-cell rewards to be emitted
        # one-by-one as the NEXT timestep's cursor reaches each position.
        self._prev_per_cell_rewards = np.array(
            info["per_cell_rewards"], dtype=np.float32
        )

        self._full_obs       = next_full_obs
        self._micro_cursor   = 0
        self._pending_action = np.ones(self.n_micro, dtype=np.int32)

        done = terminated or truncated

        if done:
            # Episode over. Nothing left to carry the buffered rewards
            # into — emit the mean of this final timestep's per-cell
            # rewards as a one-off closing signal rather than discarding
            # 39 valid reward values entirely.
            dummy_obs = np.zeros(self.N_FEATURES_PER_CELL, dtype=np.float32)
            final_reward = float(np.mean(self._prev_per_cell_rewards))
            return dummy_obs, final_reward, True, False, info

        cell_idx = self._current_micro_cell_index()
        obs = self._extract_cell_obs(self._full_obs, cell_idx)
        return obs, reward_to_emit, False, False, info

    def render(self, mode="human"):
        self.env.render(mode=mode)