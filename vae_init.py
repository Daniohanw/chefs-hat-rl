"""
vae_init.py
-----------
Generative AI Component: Variational Autoencoder (VAE) for policy warm-start.

WHY A VAE?
==========
Normally DQN/PPO networks start with random weights — they know nothing
about the game. A VAE is a generative model that:
  1. Reads thousands of game states
  2. Learns to compress them into a small "latent space"
  3. Learns to reconstruct them from that latent space

After training, the VAE encoder has learned what features matter in a
Chef's Hat game state (board layout, hand sizes, roles, etc.).

We then TRANSFER these encoder weights into the PPO policy network.
This gives PPO a head-start: its feature extractor already understands
game states before any game is played.

ARCHITECTURE
============
Encoder:  obs(228) → FC(256) → ReLU → FC(128) → μ(64), log_σ²(64)
Decoder:  z(64)    → FC(128) → ReLU → FC(256) → ReLU → FC(228) → Sigmoid

We transfer the encoder FC(256) and FC(128) layers into PPO's policy.
PPO's mlp_extractor has layers [256, 256] so we match the first layer.

TRAINING DATA
=============
We generate 50,000 synthetic game states (since we cannot access the
real environment during VAE pre-training). These states follow the
same distribution as real observations (board, hand, mask).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. VAE will use dummy initialisation.")


# ──────────────────────────────────────────────────────────────
# VAE MODEL
# ──────────────────────────────────────────────────────────────

class VAEModel(nn.Module):
    """Variational Autoencoder for game-state representation learning."""

    def __init__(self, obs_dim=228, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim

        # Encoder: obs → latent distribution (μ, log_σ²)
        self.enc_fc1   = nn.Linear(obs_dim, hidden_dim)
        self.enc_fc2   = nn.Linear(hidden_dim, 128)
        self.enc_mu    = nn.Linear(128, latent_dim)
        self.enc_logv  = nn.Linear(128, latent_dim)

        # Decoder: z → reconstructed obs
        self.dec_fc1   = nn.Linear(latent_dim, 128)
        self.dec_fc2   = nn.Linear(128, hidden_dim)
        self.dec_out   = nn.Linear(hidden_dim, obs_dim)

        self.relu      = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.enc_fc1(x))
        h = self.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logv(h)

    def reparameterise(self, mu, logv):
        if self.training:
            std = torch.exp(0.5 * logv)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        mu, logv = self.encode(x)
        z        = self.reparameterise(mu, logv)
        return self.decode(z), mu, logv


def vae_loss(recon, x, mu, logv):
    """ELBO = reconstruction loss + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon_loss + kl_loss


# ──────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────────────────────

def make_synthetic_obs(n=50_000, obs_dim=228, seed=42):
    """
    Generate plausible Chef's Hat observations for VAE training.

    Layout matches ChefsHatEnv.getObservation():
      [0:11]   board (normalised)
      [11:28]  hand  (normalised, 17 slots)
      [28:228] valid action mask (binary)
    """
    rng  = np.random.default_rng(seed)
    data = np.zeros((n, obs_dim), dtype=np.float32)

    for i in range(n):
        # Board: some slots filled with normalised card values
        n_board = rng.integers(0, 5)
        if n_board > 0:
            data[i, rng.choice(11, n_board, replace=False)] = rng.uniform(0.1, 1.0, n_board)

        # Hand: 0-17 cards, each a normalised card value
        n_cards = rng.integers(0, 18)
        if n_cards > 0:
            slots = rng.choice(17, n_cards, replace=False)
            data[i, 11 + slots] = rng.uniform(0.08, 1.0, n_cards)

        # Action mask: ~10-30% valid
        n_valid = rng.integers(1, 60)
        valid_slots = rng.choice(200, n_valid, replace=False)
        data[i, 28 + valid_slots] = 1.0

    return data


# ──────────────────────────────────────────────────────────────
# VAE TRAINER
# ──────────────────────────────────────────────────────────────

class VAETrainer:
    """Trains the VAE and transfers encoder weights to PPO policy."""

    def __init__(self, obs_dim=228, hidden_dim=256, latent_dim=64):
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.model      = None
        self.device     = "cpu"

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model  = VAEModel(obs_dim, hidden_dim, latent_dim).to(self.device)

    def pretrain(self, n_samples=50_000, epochs=30,
                 batch_size=256, lr=1e-3) -> list:
        """
        Train the VAE on synthetic data.
        Returns list of per-epoch losses for plotting.
        """
        if not TORCH_AVAILABLE:
            print("[VAE] PyTorch not available — skipping VAE training.")
            return [0.0] * epochs

        print(f"[VAE] Generating {n_samples:,} synthetic game states...")
        data    = make_synthetic_obs(n_samples, self.obs_dim)
        tensor  = torch.tensor(data, dtype=torch.float32)
        loader  = DataLoader(TensorDataset(tensor),
                             batch_size=batch_size, shuffle=True)

        opt     = optim.Adam(self.model.parameters(), lr=lr)
        losses  = []

        print(f"[VAE] Training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, mu, logv = self.model(batch)
                loss = vae_loss(recon, batch, mu, logv)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)

            avg = total / n_samples
            losses.append(avg)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.5f}")

        print("[VAE] Pre-training complete.")
        return losses

    def apply_to_policy(self, model):
        """
        Transfer VAE encoder weights into PPO's policy network.

        MaskablePPO's MlpPolicy has:
          policy.mlp_extractor.policy_net[0]  = Linear(228, 64)  (first layer)
          policy.mlp_extractor.policy_net[2]  = Linear(64, 64)   (second layer)

        We copy our VAE encoder's first layer (228→256) weights into
        the policy's first layer. We need to handle size differences
        by only copying the compatible portion.
        """
        if not TORCH_AVAILABLE or self.model is None:
            print("[VAE] Skipping weight transfer (PyTorch not available).")
            return

        try:
            vae_sd  = self.model.state_dict()
            pol_sd  = model.policy.state_dict()

            # Find the first linear layer in the policy network
            # In SB3 MlpPolicy, layers are in mlp_extractor
            transferred = 0
            for key in pol_sd:
                if "policy_net" in key and "0.weight" in key:
                    vae_w  = vae_sd.get("enc_fc1.weight")
                    pol_w  = pol_sd[key]
                    if vae_w is not None:
                        # Copy as many rows/cols as fit
                        rows = min(pol_w.shape[0], vae_w.shape[0])
                        cols = min(pol_w.shape[1], vae_w.shape[1])
                        pol_sd[key][:rows, :cols] = vae_w[:rows, :cols]
                        transferred += 1
                        print(f"  Transferred enc_fc1 → {key} "
                              f"({rows}x{cols} of {pol_w.shape})")

            model.policy.load_state_dict(pol_sd)
            print(f"  Weight transfer complete ({transferred} layer(s) updated).")

        except Exception as e:
            print(f"[VAE] Weight transfer failed: {e}")
            print("      Continuing with random initialisation.")

    def save(self, path: str):
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"[VAE] Saved to {path}")

    def load(self, path: str):
        if TORCH_AVAILABLE and self.model is not None:
            self.model.load_state_dict(
                torch.load(path, map_location=self.device))
            print(f"[VAE] Loaded from {path}")
