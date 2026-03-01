"""
train.py
--------
Trains two MaskablePPO agents on Chef's Hat and saves them:
  1. Random-Init agent  (standard random weight initialisation)
  2. GenAI-Init agent   (weights warm-started from a pre-trained VAE)

HOW TO RUN:
    python train.py

    or with fewer steps for a quick test:
    python train.py --timesteps 50000

OUTPUTS (saved to models/ folder):
    models/random_init_agent.zip      trained Random-Init model
    models/genai_init_agent.zip       trained GenAI-Init model
    models/random_init_metrics.npz    training metrics for plotting
    models/genai_init_metrics.npz     training metrics for plotting

HOW LONG IT TAKES:
    50,000 timesteps  →  ~5-10 minutes  (good for testing)
    500,000 timesteps →  ~1-2 hours     (reasonable results)
    1,000,000 timesteps → ~3-4 hours    (recommended for assignment)
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


# ── Hyperparameters ───────────────────────────────────────────
GAMMA        = 0.99      # discount factor (used by both PPO and reward shaping)
LEARNING_RATE= 3e-4      # Adam learning rate
N_STEPS      = 2048      # steps per PPO update (collect this many before training)
BATCH_SIZE   = 64        # mini-batch size for gradient updates
N_EPOCHS     = 10        # number of gradient passes per update
SHAPING_SCALE= 0.1       # reward shaping strength (0 = off)


# ── Callback: records metrics during training ─────────────────

class MetricsCallback(BaseCallback):
    """
    Records per-episode rewards and win/loss flags during training.
    These are saved and used for plotting learning curves.
    """

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.win_flags       = []   # 1 if agent finished as Chef, else 0

    def _on_step(self) -> bool:
        # SB3 puts episode info into self.locals["infos"] when an episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(float(ep_reward))

                # Win = positive reward (Chef or Sous-Chef)
                self.win_flags.append(1 if ep_reward > 0 else 0)

        return True   # True = keep training


# ── Training function ─────────────────────────────────────────

def train_agent(name: str, total_timesteps: int,
                genai_init: bool = False,
                vae_trainer=None,
                opponent_type: str = "random") -> tuple:
    """
    Train one agent and return (model, callback).

    Parameters
    ----------
    name           : display name for printing
    total_timesteps: how many env steps to train for
    genai_init     : if True, warm-start weights from VAE encoder
    vae_trainer    : trained VAETrainer (required if genai_init=True)
    opponent_type  : "random" or "greedy"
    """
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  GenAI init: {genai_init}")
    print(f"  Opponent type: {opponent_type}")
    print(f"{'='*60}")

    # Create a fresh environment for this agent
    env = SingleAgentWrapper(gamma=GAMMA, shaping_scale=SHAPING_SCALE,
                             opponent_type=opponent_type)

    # Create MaskablePPO model
    # "MlpPolicy" = multi-layer perceptron (correct for flat obs vector)
    # gamma must match the gamma used in reward shaping inside the wrapper
    model = MaskablePPO(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        verbose=1,
        tensorboard_log=f"logs/{name}",
    )

    # Apply VAE warm-start if requested (GenAI variant)
    if genai_init and vae_trainer is not None:
        print(f"  Applying VAE encoder weights to policy network...")
        vae_trainer.apply_to_policy(model)
        print(f"  VAE weights applied.")

    # Create callback to record metrics
    callback = MetricsCallback()

    # Train!
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
    )

    print(f"\n  Training complete for {name}.")
    if callback.win_flags:
        final_wr = np.mean(callback.win_flags[-100:])
        print(f"  Final 100-episode win rate: {final_wr:.3f}")

    return model, callback


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Chef's Hat RL agents")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training timesteps per agent (default: 500000)")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    TOTAL_TIMESTEPS = args.timesteps

    print("\n" + "="*60)
    print("  Chef's Hat RL Training")
    print("  Coventry University Assignment")
    print("="*60)
    print(f"  Each agent trains for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Gamma (discount): {GAMMA}")
    print(f"  Shaping scale:    {SHAPING_SCALE}")

    # ── STEP 1: Pre-train VAE (Generative AI component) ───────
    print("\n[Step 1/3] Pre-training VAE (Generative AI Component)...")
    vae_trainer = VAETrainer(obs_dim=228, hidden_dim=256, latent_dim=64)
    vae_losses  = vae_trainer.pretrain(n_samples=50_000, epochs=30)
    vae_trainer.save("models/vae_pretrained.pt")
    print("[Step 1/3] VAE pre-training complete.")

    # ── STEP 2: Train Random-Init agent ───────────────────────
    rand_model, rand_cb = train_agent(
        name            = "random_init",
        total_timesteps = TOTAL_TIMESTEPS,
        genai_init      = False,
        opponent_type   = "random",
    )
    rand_model.save("models/random_init_agent")
    np.savez("models/random_init_metrics.npz",
             episode_rewards = np.array(rand_cb.episode_rewards),
             win_flags       = np.array(rand_cb.win_flags))
    print("  Saved: models/random_init_agent.zip")

    # ── STEP 3: Train GenAI-Init agent ────────────────────────
    genai_model, genai_cb = train_agent(
        name            = "genai_init",
        total_timesteps = TOTAL_TIMESTEPS,
        genai_init      = True,
        vae_trainer     = vae_trainer,
        opponent_type   = "random",
    )
    genai_model.save("models/genai_init_agent")
    np.savez("models/genai_init_metrics.npz",
             episode_rewards = np.array(genai_cb.episode_rewards),
             win_flags       = np.array(genai_cb.win_flags),
             vae_losses      = np.array(vae_losses))
    print("  Saved: models/genai_init_agent.zip")

    # ── STEP 4: Train Mixed-Opponents agent (new experiment) ──
    mixed_model, mixed_cb = train_agent(
        name            = "mixed_opponents",
        total_timesteps = TOTAL_TIMESTEPS,
        genai_init      = False,
        opponent_type   = "mixed",
    )
    mixed_model.save("models/mixed_opponents_agent")
    np.savez("models/mixed_opponents_metrics.npz",
             episode_rewards = np.array(mixed_cb.episode_rewards),
             win_flags       = np.array(mixed_cb.win_flags))
    print("  Saved: models/mixed_opponents_agent.zip")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    r_wr  = float(np.mean(rand_cb.win_flags[-100:]))   if rand_cb.win_flags   else 0.0
    g_wr  = float(np.mean(genai_cb.win_flags[-100:]))  if genai_cb.win_flags  else 0.0
    m_wr  = float(np.mean(mixed_cb.win_flags[-100:]))  if mixed_cb.win_flags  else 0.0
    print(f"  Random-Init (vs random) win rate:   {r_wr:.3f}")
    print(f"  GenAI-Init  (vs random) win rate:   {g_wr:.3f}")
    print(f"  Mixed-Opponents win rate:            {m_wr:.3f}")
    print(f"  GenAI improvement over Random-Init: +{g_wr - r_wr:.3f}")
    print()
    print("  Next step: python evaluate.py")
    print("  Then:      python plot_results.py")


if __name__ == "__main__":
    main()
