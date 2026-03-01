"""
single_agent_wrapper.py
-----------------------
A gym.Env wrapper around ChefsHatEnv that makes it work with MaskablePPO.

THE ZERO REWARD PROBLEM (and how we fix it)
============================================
The default ChefsHatEnv ALWAYS returns reward=0.
Every single step. Even when you win. This means PPO never learns anything.

This is a known issue — the assignment hint says:
  "See ChefsHatGym.reward.only_winning for an example reward function"

We implement our OWN reward function here, inside this wrapper:

  1. WIN/LOSS reward (the critical one — fixes the zero reward problem):
       +1.0 if RL agent finishes 1st  (Chef)
       +0.5 if RL agent finishes 2nd  (Sous-Chef)
       -0.3 if RL agent finishes 3rd  (Waiter)
       -1.0 if RL agent finishes 4th  (Dishwasher)

     This is given ONCE at the end of each match (when terminated=True).
     Without this, reward is always zero and PPO cannot learn.

  2. POTENTIAL-BASED SHAPING reward (bonus dense signal):
       F = gamma * phi(next_state) - phi(current_state)
       phi(s) = -(cards_in_hand / 17) * shaping_scale

     This gives a small reward each time the agent plays a card,
     encouraging card-shedding behaviour (good heuristic for Chef's Hat).
     This is POLICY-INVARIANT (Ng et al., 1999) — it speeds up learning
     but cannot change which policy is optimal.

OBSERVATION LAYOUT (from ChefsHatEnv source code)
==================================================
  obs[0:11]   board state (11 card slots, normalised)
  obs[11:28]  RL agent's hand (17 card slots, normalised)
  obs[28:228] valid action mask (200 values: 1=valid, 0=invalid)

ACTION SPACE
============
  PPO outputs an integer 0-199.
  The env expects a 200-element one-hot float32 vector.
  We convert: action_vec[action_index] = 1.0

OTHER KEY NOTES
===============
  - startExperiment() MUST be called once before reset() — we do this in __init__
  - import chefshatgym MUST come before gym.make() — it registers the env
  - gameType="MATCHES", stopCriteria=1 means 1 match per episode
"""

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: this import MUST come before gym.make()
# It registers "ChefsHat-v0" in the gym registry.
import chefshatgym

from gym import spaces


# ── Reward values per finishing position ──────────────────────────────────────
#
# These mirror what ChefsHatGym.reward.only_winning does:
# give a positive reward for winning, negative for losing.
#
# Position 0 = 1st place = Chef     (best)
# Position 1 = 2nd place = Sous-Chef
# Position 2 = 3rd place = Waiter
# Position 3 = 4th place = Dishwasher (worst)
#
POSITION_REWARDS = {
    0: +1.0,   # Chef        — won the match
    1: +0.5,   # Sous-Chef   — second place
    2: -0.3,   # Waiter      — third place
    3: -1.0,   # Dishwasher  — lost the match
}

RL_SEAT = 0   # The RL agent always sits at seat 0


class SingleAgentWrapper(gym.Env):
    """
    Wraps ChefsHatEnv so MaskablePPO can train a single agent.

    Parameters
    ----------
    gamma : float
        Discount factor. MUST match the gamma you pass to MaskablePPO.
        Default 0.99.
    shaping_scale : float
        How strong the card-shedding shaping reward is.
        0.1 = recommended starting value.
        0.0 = no shaping (use only win/loss reward).
    seed : int
        Random seed for opponent actions.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, gamma: float = 0.99,
                 shaping_scale: float = 0.1,
                 seed: int = 42,
                 opponent_type: str = "random"):
        super().__init__()

        self.gamma         = gamma
        self.shaping_scale = shaping_scale
        self._rng          = np.random.default_rng(seed)
        self.opponent_type = opponent_type   # "random" or "greedy"

        # ── Create the base ChefsHat gym environment ──────────────────────
        self.base_env = gym.make("ChefsHat-v0")

        # ── Observation and action spaces ─────────────────────────────────
        # 228-dim float32 vector (matches base env exactly)
        self.observation_space = spaces.Box(
            low  = np.zeros(228, dtype=np.float32),
            high = np.ones(228,  dtype=np.float32),
            dtype = np.float32,
        )
        # 200 discrete actions (PPO picks one integer)
        self.action_space = spaces.Discrete(200)

        # ── IMPORTANT: startExperiment() must be called before reset() ────
        # Sets up player names, game type, logging, etc.
        # We call it ONCE here so every subsequent reset() works.
        import os
        os.makedirs("log", exist_ok=True)
        self.base_env.startExperiment(
            playerNames    = ["RL_Agent", "Random_1", "Random_2", "Random_3"],
            logDirectory   = "log",
            verbose        = 0,
            saveLog        = False,
            saveDataset    = False,
            gameType       = "MATCHES",
            stopCriteria   = 1,        # 1 match = 1 episode
            maxInvalidActions = 5,
        )

        # ── Internal state ─────────────────────────────────────────────────
        self._last_obs       = None   # most recent obs (needed for masking)
        self._prev_obs       = None   # obs before RL agent's action (for shaping)
        self._last_info      = {}     # most recent info dict
        self._episode_done   = False

    # ──────────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Start a new match. Returns first observation for the RL agent."""

        self._episode_done = False

        # Start a new match
        result = self.base_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs = np.array(obs, dtype=np.float32)

        # Auto-play opponents until it is the RL agent's turn
        obs, done = self._skip_to_rl_turn(obs)

        self._last_obs = obs
        self._prev_obs = obs.copy()
        self._last_info = {}

        return obs, {}

    # ──────────────────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action_index: int):
        """
        Apply the RL agent's action, auto-play opponents, return result.

        Parameters
        ----------
        action_index : int
            An integer 0-199 chosen by MaskablePPO.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # Save current obs for reward shaping calculation
        self._prev_obs = self._last_obs.copy()

        # ── Convert integer action to one-hot vector ──────────────────────
        # The base env expects a 200-element float32 array like:
        #   [0, 0, 0, ..., 1, ..., 0]  where the 1 is at action_index
        action_vec = np.zeros(200, dtype=np.float32)
        action_vec[int(action_index)] = 1.0

        # ── Apply the RL agent's action to the environment ────────────────
        obs, _env_reward, terminated, truncated, info = self.base_env.step(action_vec)
        # NOTE: _env_reward is ALWAYS 0 — we ignore it and compute our own below
        obs = np.array(obs, dtype=np.float32)

        # ── Auto-play opponents until RL agent's turn or game ends ────────
        if not (terminated or truncated or self.base_env.gameFinished):
            obs, opp_terminated, info = self._skip_to_rl_turn_with_info(obs)
            terminated = terminated or opp_terminated

        terminated = terminated or self.base_env.gameFinished
        self._last_obs  = obs
        self._last_info = info

        # ── Compute our custom reward (fixes the zero reward problem) ─────
        reward = self._compute_reward(obs, terminated, info)

        return obs, reward, bool(terminated), False, info

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION MASKS  (called automatically by MaskablePPO)
    # ──────────────────────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean array of length 200.
        True = this action is legal. False = illegal.

        The valid actions are already embedded in the observation:
          obs[28:228]  →  1.0 if valid, 0.0 if invalid

        Method name MUST be exactly 'action_masks' for MaskablePPO to find it.
        """
        if self._last_obs is None:
            return np.ones(200, dtype=bool)

        mask = self._last_obs[28:228].astype(bool)

        # Safety: always allow 'pass' (index 199) if everything is masked out
        if not mask.any():
            mask[199] = True

        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # REWARD COMPUTATION
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, next_obs: np.ndarray,
                        terminated: bool, info: dict) -> float:
        """
        Compute the total reward for this step.

        This replaces the env's built-in reward (which is always 0).

        Component 1: Win/loss terminal reward
        --------------------------------------
        Given once when the match ends (terminated=True).
        +1.0 for winning (1st place), down to -1.0 for losing (4th place).

        This is the CRITICAL component that fixes the zero reward problem.
        Without this, the agent never learns.

        Component 2: Potential-based shaping (optional)
        ------------------------------------------------
        Given every step. Rewards card reduction.
        Formula: F = gamma * phi(s') - phi(s)
        phi(s) = -(cards_in_hand / 17) * shaping_scale

        Policy-invariant per Ng et al. (1999).
        """
        reward = 0.0

        # ── Component 1: Win/loss reward (THE FIX for zero reward) ───────
        if terminated:
            position = self._get_rl_finishing_position(info)
            reward  += POSITION_REWARDS.get(position, 0.0)

        # ── Component 2: Potential-based shaping ──────────────────────────
        if self.shaping_scale > 0.0 and self._prev_obs is not None:
            phi_s  = self._phi(self._prev_obs)   # potential before action
            phi_s2 = self._phi(next_obs)          # potential after action
            reward += self.gamma * phi_s2 - phi_s

        return float(reward)

    def _get_rl_finishing_position(self, info: dict) -> int:
        """
        Work out what position (0=1st, 3=4th) the RL agent finished in.

        Uses info["Match_Score"] from the env.
        Match_Score is a list of scores per player seat.
        Score 3 = Chef (1st), Score 0 = Dishwasher (4th).
        So: position = 3 - score
        """
        if "Match_Score" in info:
            scores   = info["Match_Score"]   # e.g. [3, 2, 1, 0]
            rl_score = scores[RL_SEAT]        # score for seat 0
            position = 3 - int(rl_score)     # convert to finishing position
            return max(0, min(3, position))  # clamp to 0-3

        # Fallback: if info is missing, check players directly
        try:
            pos = self.base_env.players[RL_SEAT].finishing_position
            if pos is not None and pos >= 0:
                return int(pos)
        except Exception:
            pass

        return 3   # worst case assumption

    def _phi(self, obs: np.ndarray) -> float:
        """
        Potential function for reward shaping.

        phi(s) = -(cards_in_hand / 17.0) * shaping_scale

        obs[11:28] = the RL agent's hand (17 slots).
        Non-zero values = cards still held.
        More cards = more negative potential.
        Shedding cards increases potential → positive shaping reward.
        """
        hand       = obs[11:28]
        cards_left = float(np.count_nonzero(hand))
        return -(cards_left / 17.0) * self.shaping_scale

    # ──────────────────────────────────────────────────────────────────────────
    # OPPONENT AUTO-PLAY HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _skip_to_rl_turn(self, obs: np.ndarray):
        """
        Auto-play opponents until it is the RL agent's turn.
        Returns (obs, done).
        Used in reset() where we don't need info.
        """
        obs, done, _ = self._skip_to_rl_turn_with_info(obs)
        return obs, done

    def _skip_to_rl_turn_with_info(self, obs: np.ndarray):
        """
        Auto-play opponents until it is the RL agent's turn.
        Returns (obs, done, last_info).

        Each opponent picks a random valid action from obs[28:228].
        We keep stepping until:
          - It is the RL agent's (seat 0) turn, OR
          - The game ends
        """
        last_info = {}
        max_steps = 500   # safety limit — a Chef's Hat match is ~100-200 steps

        # Mixed opponent strategies — each seat has a fixed strategy
        # Seat 1 = random, Seat 2 = greedy, Seat 3 = lowest card
        MIXED_STRATEGIES = {1: "random", 2: "greedy", 3: "lowest"}

        for _ in range(max_steps):
            # Stop if it is the RL agent's turn
            if self.base_env.currentPlayer == RL_SEAT:
                break

            # Work out which strategy this opponent uses
            current_seat = self.base_env.currentPlayer
            if self.opponent_type == "mixed":
                strategy = MIXED_STRATEGIES.get(current_seat, "random")
            else:
                strategy = self.opponent_type

            # Pick an action based on strategy
            mask      = obs[28:228].astype(bool)
            valid_idx = np.where(mask)[0]
            non_pass  = valid_idx[valid_idx != 199]

            if len(valid_idx) == 0:
                action_idx = 199   # forced pass

            elif strategy == "greedy":
                # Always play highest value non-pass action
                if len(non_pass) > 0:
                    action_idx = int(non_pass[-1])
                else:
                    action_idx = 199

            elif strategy == "lowest":
                # Always play lowest value non-pass action
                if len(non_pass) > 0:
                    action_idx = int(non_pass[0])
                else:
                    action_idx = 199

            else:
                # Random: pick uniformly from valid actions
                action_idx = int(self._rng.choice(valid_idx))

            action_vec             = np.zeros(200, dtype=np.float32)
            action_vec[action_idx] = 1.0

            obs, _, terminated, truncated, last_info = self.base_env.step(action_vec)
            obs = np.array(obs, dtype=np.float32)

            if terminated or truncated or self.base_env.gameFinished:
                return obs, True, last_info

        return obs, False, last_info

    # ──────────────────────────────────────────────────────────────────────────
    # STANDARD GYM METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, mode="human"):
        pass

    def close(self):
        self.base_env.close()
