"""
test_env.py
-----------
Run this FIRST before any training to check your installation is correct.

HOW TO RUN:
    python test_env.py

EXPECTED OUTPUT (all lines should show a tick):
    [OK] gym imported
    [OK] chefshatgym imported
    [OK] Environment created
    [OK] startExperiment() succeeded
    [OK] reset() returned obs shape (228,)
    [OK] Action mask has N valid actions
    [OK] step() worked
    [OK] MaskablePPO importable
    [OK] PyTorch available
    All checks passed - ready to train!
"""

import sys
import numpy as np

print("=" * 60)
print("  Chef's Hat RL - Installation Check")
print("=" * 60)

errors = []

# ── Check 1: classic gym ──────────────────────────────────────
try:
    import gym
    print(f"[OK] gym imported  (version: {gym.__version__})")
except ImportError:
    print("[FAIL] gym not found")
    print("       Fix: pip install gym==0.26.2")
    errors.append("gym")

# ── Check 2: chefshatgym ──────────────────────────────────────
try:
    import chefshatgym          # this line triggers gym.register("ChefsHat-v0")
    print("[OK] chefshatgym imported")
except ImportError:
    print("[FAIL] chefshatgym not found")
    print("       Fix: pip install chefshatgym")
    errors.append("chefshatgym")

if errors:
    print("\nFix the errors above then re-run this script.")
    sys.exit(1)

# ── Check 3: create environment ───────────────────────────────
try:
    env = gym.make("ChefsHat-v0")
    print(f"[OK] Environment created: {type(env).__name__}")
except Exception as e:
    print(f"[FAIL] gym.make('ChefsHat-v0') failed: {e}")
    print("       Make sure you ran: import chefshatgym  BEFORE gym.make()")
    sys.exit(1)

# ── Check 4: startExperiment (REQUIRED before reset) ─────────
# The ChefsHatEnv MUST have startExperiment() called before reset().
# Without this, reset() crashes because players are not set up yet.
try:
    import os
    os.makedirs("log", exist_ok=True)
    env.startExperiment(
        playerNames=["RL_Agent", "Random_1", "Random_2", "Random_3"],
        logDirectory="log",
        verbose=0,
        saveLog=False,
        saveDataset=False,
        gameType="MATCHES",
        stopCriteria=3,
        maxInvalidActions=5,
    )
    print("[OK] startExperiment() succeeded")
except Exception as e:
    print(f"[FAIL] startExperiment() failed: {e}")
    sys.exit(1)

# ── Check 5: reset ────────────────────────────────────────────
try:
    result = env.reset()
    # gym 0.26 returns (obs, info) tuple
    obs = result[0] if isinstance(result, tuple) else result
    assert obs.shape == (228,), f"Expected shape (228,) but got {obs.shape}"
    print(f"[OK] reset() returned obs shape: {obs.shape}")
except Exception as e:
    print(f"[FAIL] reset() failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Check 6: action mask in observation ──────────────────────
# The observation layout (from ChefsHatEnv source code):
#   obs[0:11]   = board state
#   obs[11:28]  = player's hand (17 card slots)
#   obs[28:228] = valid action mask (200 values, 0=invalid 1=valid)
mask = obs[28:].astype(bool)
n_valid = int(mask.sum())
assert n_valid > 0, "No valid actions found - something is wrong"
print(f"[OK] Action mask: {n_valid} valid actions out of 200")

# ── Check 7: one step ─────────────────────────────────────────
try:
    # PPO gives an integer; env.step() needs a 200-dim one-hot vector
    valid_indices = np.where(mask)[0]
    action_index = int(valid_indices[0])

    action_vec = np.zeros(200, dtype=np.float32)
    action_vec[action_index] = 1.0

    result = env.step(action_vec)
    next_obs, reward, terminated, truncated, info = result
    print(f"[OK] step() worked - reward={reward}, terminated={terminated}")
    print(f"     next_obs shape: {next_obs.shape}")
    print(f"     example info keys: {list(info.keys())[:4]}")
except Exception as e:
    print(f"[FAIL] step() failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Check 8: MaskablePPO ──────────────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    print("[OK] MaskablePPO importable from sb3_contrib")
except ImportError:
    print("[FAIL] sb3-contrib not found")
    print("       Fix: pip install sb3-contrib==2.3.0")
    sys.exit(1)

# ── Check 9: PyTorch ─────────────────────────────────────────
try:
    import torch
    print(f"[OK] PyTorch available (version: {torch.__version__})")
except ImportError:
    print("[FAIL] torch not found")
    print("       Fix: pip install torch")
    sys.exit(1)

print()
print("=" * 60)
print("  All checks passed - ready to train!")
print("  Next step: python train.py")
print("=" * 60)
