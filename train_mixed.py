"""
train_mixed.py
--------------
Trains TWO agents against mixed opponents (Random + Greedy + Lowest-Card):
  1. Random-Init vs Mixed Opponents
  2. GenAI-Init  vs Mixed Opponents

This completes the 2x2 experiment grid:
  ┌─────────────────┬──────────────────┬──────────────────┐
  │                 │  Random Opponents │  Mixed Opponents │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Random-Init PPO │  random_init      │ mixed_random     │
  │ GenAI-Init  PPO │  genai_init       │ mixed_genai      │
  └─────────────────┴──────────────────┴──────────────────┘

HOW TO RUN:
    python train_mixed.py

    or with fewer steps for a quick test:
    python train_mixed.py --timesteps 50000
"""

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from single_agent_wrapper import SingleAgentWrapper
from vae_init import VAETrainer


class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.win_flags = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                self.episode_rewards.append(float(r))
                self.win_flags.append(1 if r > 0 else 0)
        return True


def train_agent(name, total_timesteps, genai_init=False, vae_trainer=None):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Opponents: Random + Greedy + Lowest-Card (mixed)")
    print(f"  GenAI init: {genai_init}")
    print(f"{'='*60}")

    env = SingleAgentWrapper(gamma=0.99, shaping_scale=0.1,
                             opponent_type="mixed")

    model = MaskablePPO(
        "MlpPolicy", env,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log=f"logs/{name}",
    )

    if genai_init and vae_trainer is not None:
        print("  Applying VAE encoder weights to policy network...")
        vae_trainer.apply_to_policy(model)
        print("  VAE weights applied.")

    cb = MetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)

    wr = float(np.mean(cb.win_flags[-100:])) if cb.win_flags else 0.0
    print(f"\n  Training complete. Final win rate: {wr:.3f}")

    return model, cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    print("\n" + "="*60)
    print("  Mixed Opponents Experiment")
    print("  Opponents: Random + Greedy + Lowest-Card")
    print(f"  Timesteps per agent: {args.timesteps:,}")
    print("="*60)

    # Pre-train VAE (needed for GenAI-Init agent)
    print("\n[Step 1/3] Pre-training VAE...")
    vae_trainer = VAETrainer(obs_dim=228, hidden_dim=256, latent_dim=64)
    vae_trainer.pretrain(n_samples=50_000, epochs=30)
    print("[Step 1/3] VAE pre-training complete.")

    # Train Random-Init vs Mixed Opponents
    rand_model, rand_cb = train_agent(
        name="mixed_random",
        total_timesteps=args.timesteps,
        genai_init=False,
    )
    rand_model.save("models/mixed_random_agent")
    np.savez("models/mixed_random_metrics.npz",
             episode_rewards=np.array(rand_cb.episode_rewards),
             win_flags=np.array(rand_cb.win_flags))
    print("  Saved: models/mixed_random_agent.zip")

    # Train GenAI-Init vs Mixed Opponents
    genai_model, genai_cb = train_agent(
        name="mixed_genai",
        total_timesteps=args.timesteps,
        genai_init=True,
        vae_trainer=vae_trainer,
    )
    genai_model.save("models/mixed_genai_agent")
    np.savez("models/mixed_genai_metrics.npz",
             episode_rewards=np.array(genai_cb.episode_rewards),
             win_flags=np.array(genai_cb.win_flags))
    print("  Saved: models/mixed_genai_agent.zip")

    # Summary
    r_wr = float(np.mean(rand_cb.win_flags[-100:]))  if rand_cb.win_flags  else 0.0
    g_wr = float(np.mean(genai_cb.win_flags[-100:])) if genai_cb.win_flags else 0.0

    print("\n" + "="*60)
    print("  MIXED OPPONENTS TRAINING COMPLETE")
    print("="*60)
    print(f"  Random-Init vs Mixed:  {r_wr:.3f}")
    print(f"  GenAI-Init  vs Mixed:  {g_wr:.3f}")
    print(f"  GenAI improvement:    +{g_wr - r_wr:.3f}")
    print()
    print("  Full 2x2 results (from earlier training):")
    print(f"  Random-Init vs Random: 0.990  (from train.py)")
    print(f"  GenAI-Init  vs Random: 0.970  (from train.py)")
    print()
    print("  Next step: python evaluate.py")
    print("  Then:      python plot_results.py")


if __name__ == "__main__":
    main()
