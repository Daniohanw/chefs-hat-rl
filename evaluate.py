"""
evaluate.py
-----------
Evaluates all four trained agents and prints a comparison table.

Agents evaluated:
  1. random_init      — Random-Init PPO vs random opponents
  2. genai_init       — GenAI-Init  PPO vs random opponents
  3. mixed_random     — Random-Init PPO vs mixed opponents
  4. mixed_genai      — GenAI-Init  PPO vs mixed opponents

HOW TO RUN:
    python evaluate.py

    or with fewer episodes:
    python evaluate.py --episodes 50
"""

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from sb3_contrib import MaskablePPO
from single_agent_wrapper import SingleAgentWrapper


def evaluate_agent(model_path: str, n_episodes: int = 100,
                   opponent_type: str = "random") -> dict:
    """Load a model and evaluate it for n_episodes."""
    print(f"\n  Evaluating: {model_path} (opponents: {opponent_type})")

    if not os.path.exists(model_path + ".zip"):
        print(f"  [SKIP] File not found: {model_path}.zip")
        return {}

    env   = SingleAgentWrapper(gamma=0.99, shaping_scale=0.0,
                               opponent_type=opponent_type)
    model = MaskablePPO.load(model_path, env=env)

    rewards            = []
    positions          = []
    wins               = []
    performance_scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_rew = 0.0

        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(),
                                      deterministic=True)
            obs, rew, done, _, info = env.step(action)
            ep_rew += rew

        rewards.append(ep_rew)
        wins.append(1 if ep_rew > 0 else 0)

        if "Match_Score" in info:
            rl_score = info["Match_Score"][0]
            position = 3 - int(rl_score)
        else:
            position = 3
        positions.append(position)

        try:
            perf = env.base_env.players[0].acumulated_performance_score
            performance_scores.append(float(perf))
        except Exception:
            performance_scores.append(0.0)

        if (ep + 1) % 20 == 0:
            print(f"    Episode {ep+1}/{n_episodes}  "
                  f"win_rate={np.mean(wins):.3f}  "
                  f"avg_reward={np.mean(rewards):.3f}")

    env.close()

    return {
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "win_rate":        float(np.mean(wins)),
        "avg_position":    float(np.mean(positions)),
        "position_dist":   np.bincount(positions, minlength=4) / n_episodes,
        "mean_perf_score": float(np.mean(performance_scores)) if performance_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Chef's Hat RL - Evaluation")
    print("="*60)
    print(f"  Episodes per agent: {args.episodes}")

    results = {}

    # ── Experiment 1: vs Random Opponents ─────────────────────
    print("\n--- Experiment 1: vs Random Opponents ---")
    results["random_init"] = evaluate_agent(
        "models/random_init_agent", args.episodes, opponent_type="random")
    results["genai_init"]  = evaluate_agent(
        "models/genai_init_agent",  args.episodes, opponent_type="random")

    # ── Experiment 2: vs Mixed Opponents ──────────────────────
    print("\n--- Experiment 2: vs Mixed Opponents ---")
    results["mixed_random"] = evaluate_agent(
        "models/mixed_random_agent", args.episodes, opponent_type="mixed")
    results["mixed_genai"]  = evaluate_agent(
        "models/mixed_genai_agent",  args.episodes, opponent_type="mixed")

    # ── Print 2x2 comparison table ────────────────────────────
    print("\n" + "="*70)
    print("  EVALUATION RESULTS — 2x2 Experiment Grid")
    print("="*70)

    print(f"\n  {'Metric':<25}  {'Rand-Init':>10}  {'GenAI-Init':>10}  {'Diff':>8}")
    print("  " + "-"*58)
    print("  vs RANDOM OPPONENTS:")
    for label, key, fmt in [
        ("  Win Rate",              "win_rate",        ".3f"),
        ("  Avg Position",          "avg_position",     ".2f"),
        ("  Env Performance Score", "mean_perf_score",  ".4f"),
    ]:
        r = results.get("random_init", {}).get(key, 0)
        g = results.get("genai_init",  {}).get(key, 0)
        d = g - r
        print(f"  {label:<25}  {r:>10{fmt}}  {g:>10{fmt}}  {d:>+8.3f}")

    print()
    print("  vs MIXED OPPONENTS (Random + Greedy + Lowest-Card):")
    for label, key, fmt in [
        ("  Win Rate",              "win_rate",        ".3f"),
        ("  Avg Position",          "avg_position",     ".2f"),
        ("  Env Performance Score", "mean_perf_score",  ".4f"),
    ]:
        r = results.get("mixed_random", {}).get(key, 0)
        g = results.get("mixed_genai",  {}).get(key, 0)
        d = g - r
        print(f"  {label:<25}  {r:>10{fmt}}  {g:>10{fmt}}  {d:>+8.3f}")

    print()
    print("  DIFFICULTY COMPARISON (Random-Init only):")
    r_rand  = results.get("random_init",  {}).get("win_rate", 0)
    r_mixed = results.get("mixed_random", {}).get("win_rate", 0)
    print(f"    vs Random opponents: {r_rand:.3f}")
    print(f"    vs Mixed  opponents: {r_mixed:.3f}")
    print(f"    Difficulty drop:     {r_mixed - r_rand:+.3f}")

    # ── Position distributions ─────────────────────────────────
    print()
    print("  Position distribution (P0=Chef ... P3=Dishwasher):")
    for label, key in [
        ("random_init",  "random_init"),
        ("genai_init",   "genai_init"),
        ("mixed_random", "mixed_random"),
        ("mixed_genai",  "mixed_genai"),
    ]:
        dist = results.get(key, {}).get("position_dist", np.zeros(4))
        print(f"    {label:<15}: " + "  ".join(
            f"P{i}={dist[i]:.2f}" for i in range(4)))

    # ── Save results ───────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    np.savez("models/evaluation_results.npz",
             **{f"{agent}_{k}": v
                for agent, res in results.items()
                for k, v in res.items()
                if isinstance(v, (float, int, np.ndarray))})
    print("\n  Saved: models/evaluation_results.npz")
    print("  Next step: python plot_results.py")


if __name__ == "__main__":
    main()
