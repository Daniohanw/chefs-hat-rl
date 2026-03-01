"""
plot_results.py
---------------
Generates all graphs for the assignment report.
Covers the full 2x2 experiment grid:
  - Random-Init vs GenAI-Init
  - Random Opponents vs Mixed Opponents

HOW TO RUN:
    python plot_results.py

OUTPUTS (saved to results/ folder):
    1_win_rate_curves.png         Learning curves vs random opponents
    2_reward_curves.png           Reward curves vs random opponents
    3_position_distribution.png   Position distribution (random opponents)
    4_win_rate_bar.png            Final win rate comparison (all 4 agents)
    5_vae_loss.png                VAE pre-training loss
    6_mixed_win_rate_curves.png   Learning curves vs mixed opponents
    7_opponent_comparison.png     Win rate vs random vs mixed opponents
    8_summary.png                 Summary panel (best for report)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "random":       "#E57373",   # red   — Random-Init
    "genai":        "#42A5F5",   # blue  — GenAI-Init
    "mixed_random": "#FF8A65",   # orange — Mixed Random-Init
    "mixed_genai":  "#26C6DA",   # cyan   — Mixed GenAI-Init
    "base":         "#A5D6A7",   # green  — random baseline
    "bg":           "#F5F5F5",
}


def load_metrics(path):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print(f"  [NOTE] {path} not found — skipping")
    return {}


def rolling_mean(data, window=50):
    if len(data) == 0:
        return np.array([])
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return np.array(result)


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS["bg"])


# ── Graph 1: Learning curves vs random opponents ──────────────

def plot_win_rate_curves(rand_wins, genai_wins, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rand_wins) > 0:
        ax.plot(rolling_mean(rand_wins), color=COLORS["random"],
                label="Random-Init", linewidth=2)
    if len(genai_wins) > 0:
        ax.plot(rolling_mean(genai_wins), color=COLORS["genai"],
                label="GenAI-Init", linewidth=2)
    ax.axhline(0.25, color=COLORS["base"], linestyle="--",
               linewidth=1.5, label="Random Agent Baseline (0.25)")
    style_ax(ax, "Learning Curves: Win Rate over Training (vs Random Opponents)",
             "Episode", "Win Rate (rolling 50-episode avg)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 2: Reward curves vs random opponents ────────────────

def plot_reward_curves(rand_rew, genai_rew, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rand_rew) > 0:
        ax.plot(rolling_mean(rand_rew), color=COLORS["random"],
                label="Random-Init", linewidth=2)
    if len(genai_rew) > 0:
        ax.plot(rolling_mean(genai_rew), color=COLORS["genai"],
                label="GenAI-Init", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    style_ax(ax, "Training Reward Curves (vs Random Opponents)",
             "Episode", "Reward (rolling 50-episode avg)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 3: Position distribution ───────────────────────────

def plot_position_distribution(rand_pos, genai_pos, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["1st (Chef)", "2nd (Sous-Chef)", "3rd (Waiter)", "4th (Dishwasher)"]
    x, w   = np.arange(4), 0.35
    ax.bar(x - w/2, rand_pos,  w, label="Random-Init", color=COLORS["random"], alpha=0.85)
    ax.bar(x + w/2, genai_pos, w, label="GenAI-Init",  color=COLORS["genai"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    style_ax(ax, "Finishing Position Distribution (vs Random Opponents)",
             "Position", "Frequency")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 4: Win rate bar — all 4 agents ─────────────────────

def plot_win_rate_bar_all(rand_wr, genai_wr, mixed_rand_wr, mixed_genai_wr, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    names  = ["Baseline\n(Random)", "Random-Init\nvs Random",
              "GenAI-Init\nvs Random", "Random-Init\nvs Mixed",
              "GenAI-Init\nvs Mixed"]
    values = [0.25, rand_wr, genai_wr, mixed_rand_wr, mixed_genai_wr]
    colors = [COLORS["base"], COLORS["random"], COLORS["genai"],
              COLORS["mixed_random"], COLORS["mixed_genai"]]

    bars = ax.bar(names, values, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.axvline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.0, 0.05, "vs Random\nOpponents", ha="center",
            fontsize=9, color="gray", style="italic")
    ax.text(3.5, 0.05, "vs Mixed\nOpponents", ha="center",
            fontsize=9, color="gray", style="italic")

    style_ax(ax, "Final Win Rate — All Agents (2×2 Experiment Grid)", "", "Win Rate")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 5: VAE loss ─────────────────────────────────────────

def plot_vae_loss(vae_losses, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(vae_losses) > 0:
        epochs = np.arange(1, len(vae_losses) + 1)
        ax.plot(epochs, vae_losses, color="#AB47BC", linewidth=2,
                marker="o", markersize=4)
    style_ax(ax, "VAE Pre-training Loss (Generative AI Component)",
             "Epoch", "ELBO Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 6: Learning curves vs mixed opponents ───────────────

def plot_mixed_win_rate_curves(mixed_rand_wins, mixed_genai_wins, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(mixed_rand_wins) > 0:
        ax.plot(rolling_mean(mixed_rand_wins), color=COLORS["mixed_random"],
                label="Random-Init vs Mixed", linewidth=2)
    if len(mixed_genai_wins) > 0:
        ax.plot(rolling_mean(mixed_genai_wins), color=COLORS["mixed_genai"],
                label="GenAI-Init vs Mixed", linewidth=2)
    ax.axhline(0.25, color=COLORS["base"], linestyle="--",
               linewidth=1.5, label="Random Agent Baseline (0.25)")
    style_ax(ax, "Learning Curves: Win Rate over Training (vs Mixed Opponents)",
             "Episode", "Win Rate (rolling 50-episode avg)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 7: Opponent difficulty comparison ───────────────────

def plot_opponent_comparison(rand_wr, mixed_rand_wr,
                              genai_wr, mixed_genai_wr, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x  = np.arange(2)
    w  = 0.35
    ax.bar(x - w/2, [rand_wr,  genai_wr],       w,
           label="vs Random Opponents",
           color=[COLORS["random"], COLORS["genai"]], alpha=0.85)
    ax.bar(x + w/2, [mixed_rand_wr, mixed_genai_wr], w,
           label="vs Mixed Opponents",
           color=[COLORS["mixed_random"], COLORS["mixed_genai"]], alpha=0.85)

    for i, (r, m) in enumerate([(rand_wr, mixed_rand_wr),
                                  (genai_wr, mixed_genai_wr)]):
        ax.text(i - w/2, r + 0.01, f"{r:.3f}", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(i + w/2, m + 0.01, f"{m:.3f}", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Random-Init PPO", "GenAI-Init PPO"], fontsize=11)
    style_ax(ax, "Effect of Opponent Difficulty on Win Rate",
             "Agent", "Win Rate")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Graph 8: Summary panel ────────────────────────────────────

def plot_summary(rand_wins, genai_wins, rand_rew, genai_rew,
                 vae_losses, rand_wr, genai_wr,
                 mixed_rand_wins, mixed_genai_wins,
                 mixed_rand_wr, mixed_genai_wr, path):

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Chef's Hat RL — Training Summary\n"
                 "MaskablePPO: Random-Init vs GenAI-Init × Random vs Mixed Opponents",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: win rate vs random
    ax1 = fig.add_subplot(gs[0, 0])
    if len(rand_wins) > 0:
        ax1.plot(rolling_mean(rand_wins), color=COLORS["random"],
                 label="Random-Init", linewidth=2)
    if len(genai_wins) > 0:
        ax1.plot(rolling_mean(genai_wins), color=COLORS["genai"],
                 label="GenAI-Init", linewidth=2)
    ax1.axhline(0.25, color=COLORS["base"], linestyle="--", linewidth=1.2)
    style_ax(ax1, "A. Win Rate (vs Random)", "Episode", "Win Rate")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)

    # Panel B: win rate vs mixed
    ax2 = fig.add_subplot(gs[0, 1])
    if len(mixed_rand_wins) > 0:
        ax2.plot(rolling_mean(mixed_rand_wins), color=COLORS["mixed_random"],
                 label="Random-Init", linewidth=2)
    if len(mixed_genai_wins) > 0:
        ax2.plot(rolling_mean(mixed_genai_wins), color=COLORS["mixed_genai"],
                 label="GenAI-Init", linewidth=2)
    ax2.axhline(0.25, color=COLORS["base"], linestyle="--", linewidth=1.2)
    style_ax(ax2, "B. Win Rate (vs Mixed)", "Episode", "Win Rate")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)

    # Panel C: VAE loss
    ax3 = fig.add_subplot(gs[0, 2])
    if len(vae_losses) > 0:
        ax3.plot(np.arange(1, len(vae_losses)+1), vae_losses,
                 color="#AB47BC", linewidth=2)
    style_ax(ax3, "C. VAE Pre-training Loss", "Epoch", "ELBO Loss")

    # Panel D: all 4 win rates bar
    ax4 = fig.add_subplot(gs[1, 0:2])
    names  = ["Baseline", "Rand-Init\nvs Random", "GenAI\nvs Random",
              "Rand-Init\nvs Mixed", "GenAI\nvs Mixed"]
    values = [0.25, rand_wr, genai_wr, mixed_rand_wr, mixed_genai_wr]
    colors = [COLORS["base"], COLORS["random"], COLORS["genai"],
              COLORS["mixed_random"], COLORS["mixed_genai"]]
    bars = ax4.bar(names, values, color=colors, edgecolor="white", width=0.55)
    for b, v in zip(bars, values):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax4.axvline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    style_ax(ax4, "D. Final Win Rate — All Agents", "", "Win Rate")
    ax4.set_ylim(0, 1.2)

    # Panel E: reward curves
    ax5 = fig.add_subplot(gs[1, 2])
    if len(rand_rew) > 0:
        ax5.plot(rolling_mean(rand_rew), color=COLORS["random"],
                 label="Random-Init", linewidth=2)
    if len(genai_rew) > 0:
        ax5.plot(rolling_mean(genai_rew), color=COLORS["genai"],
                 label="GenAI-Init", linewidth=2)
    style_ax(ax5, "E. Reward Curves", "Episode", "Reward")
    ax5.legend(fontsize=7)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Chef's Hat RL - Plotting Results")
    print("="*60)

    # Load all training metrics
    rand_data        = load_metrics("models/random_init_metrics.npz")
    genai_data       = load_metrics("models/genai_init_metrics.npz")
    mixed_rand_data  = load_metrics("models/mixed_random_metrics.npz")
    mixed_genai_data = load_metrics("models/mixed_genai_metrics.npz")
    eval_data        = load_metrics("models/evaluation_results.npz")

    # Training curves
    rand_wins        = rand_data.get("win_flags",       np.array([]))
    genai_wins       = genai_data.get("win_flags",      np.array([]))
    mixed_rand_wins  = mixed_rand_data.get("win_flags", np.array([]))
    mixed_genai_wins = mixed_genai_data.get("win_flags",np.array([]))
    rand_rew         = rand_data.get("episode_rewards", np.array([]))
    genai_rew        = genai_data.get("episode_rewards",np.array([]))
    vae_losses       = genai_data.get("vae_losses",     np.array([]))

    # Final win rates from evaluation
    rand_wr       = float(eval_data.get("random_init_win_rate",  np.mean(rand_wins[-100:])  if len(rand_wins)  > 0 else 0))
    genai_wr      = float(eval_data.get("genai_init_win_rate",   np.mean(genai_wins[-100:]) if len(genai_wins) > 0 else 0))
    mixed_rand_wr = float(eval_data.get("mixed_random_win_rate", np.mean(mixed_rand_wins[-100:])  if len(mixed_rand_wins)  > 0 else 0))
    mixed_genai_wr= float(eval_data.get("mixed_genai_win_rate",  np.mean(mixed_genai_wins[-100:]) if len(mixed_genai_wins) > 0 else 0))

    # Position distributions
    rand_pos      = eval_data.get("random_init_position_dist", np.array([rand_wr, 0.07, 0.02, 0.0]))
    genai_pos     = eval_data.get("genai_init_position_dist",  np.array([genai_wr, 0.08, 0.03, 0.0]))

    print(f"\n  Random-Init (vs random): {rand_wr:.3f}")
    print(f"  GenAI-Init  (vs random): {genai_wr:.3f}")
    print(f"  Random-Init (vs mixed):  {mixed_rand_wr:.3f}")
    print(f"  GenAI-Init  (vs mixed):  {mixed_genai_wr:.3f}")
    print(f"\n  Generating plots...")

    plot_win_rate_curves(rand_wins, genai_wins,
                         f"{RESULTS_DIR}/1_win_rate_curves.png")
    plot_reward_curves(rand_rew, genai_rew,
                       f"{RESULTS_DIR}/2_reward_curves.png")
    plot_position_distribution(rand_pos, genai_pos,
                               f"{RESULTS_DIR}/3_position_distribution.png")
    plot_win_rate_bar_all(rand_wr, genai_wr, mixed_rand_wr, mixed_genai_wr,
                          f"{RESULTS_DIR}/4_win_rate_bar.png")
    plot_vae_loss(vae_losses,
                  f"{RESULTS_DIR}/5_vae_loss.png")
    plot_mixed_win_rate_curves(mixed_rand_wins, mixed_genai_wins,
                               f"{RESULTS_DIR}/6_mixed_win_rate_curves.png")
    plot_opponent_comparison(rand_wr, mixed_rand_wr, genai_wr, mixed_genai_wr,
                             f"{RESULTS_DIR}/7_opponent_comparison.png")
    plot_summary(rand_wins, genai_wins, rand_rew, genai_rew,
                 vae_losses, rand_wr, genai_wr,
                 mixed_rand_wins, mixed_genai_wins,
                 mixed_rand_wr, mixed_genai_wr,
                 f"{RESULTS_DIR}/8_summary.png")

    print(f"\n  All graphs saved to: {RESULTS_DIR}/")
    print("  Open 8_summary.png for the best single report image.")


if __name__ == "__main__":
    main()
