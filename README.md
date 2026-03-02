# Chef's Hat RL - Task 2

**Name:** Daniel Ohanwe  
**Student ID:** 15874137  
**Variant:** 6  

---

## Task 2

For this task I trained reinforcement learning agents to play the Chef's Hat card game. I used MaskablePPO as my RL algorithm and a Variational Autoencoder (VAE) as the generative AI part. The idea was to pre-train the VAE on game observations, then use those weights to initialise the PPO network before training starts, and see if that helps compared to just starting with random weights.

I ran two experiments:
- Experiment 1: Random-Init PPO vs GenAI-Init PPO, both trained against random opponents
- Experiment 2: Both agents trained against a harder mix of opponents (random + greedy + lowest-card strategy)

---

## Setting Up on Windows

First make sure you have Python 3.11 installed. Then open Command Prompt and run these steps one at a time.

**Clone the repo and go into the folder:**
```
git clone https://github.com/Daniohanw/chefs-hat-rl.git
cd chefs-hat-rl
```

**Create a virtual environment and activate it:**
```
python -m venv chefs_env
chefs_env\Scripts\activate
```

**Install everything needed:**
```
python -m pip install chefshatgym==3.0.0.1 stable-baselines3==2.3.2 sb3-contrib==2.3.0 gym==0.26.2 torch numpy==1.26.1 matplotlib seaborn pandas tqdm rich shimmy tensorboard
```

**There is a bug with chefshatgym on Windows to do with capitalisation - fix it by running:**
```
python fix_init.py
```

**Check everything installed properly:**
```
python test_env.py
```
All lines should say OK. If anything fails the script will tell you what to do.

---

## Running the Code

**Train the main two agents (takes about 1-2 hours):**
```
python train.py --timesteps 500000
```

**Train both agents against the harder mixed opponents (about 1 hour):**
```
python train_mixed.py --timesteps 200000
```

**Evaluate the trained agents:**
```
python evaluate.py
```

**Generate all the graphs:**
```
python plot_results.py
```

Graphs get saved into the results folder. The best one to look at first is `8_summary.png` which shows everything together.

---

## How I Handled the Reward Function

One problem I ran into early on is that the Chef's Hat environment always returns 0 as the reward, even when you win. This means the agent has no way of learning what is good or bad, so training just does nothing.

To fix this I wrote a custom reward function inside `single_agent_wrapper.py`. It has two parts:

The first part gives a reward at the end of each match depending on what position the agent finished:
- 1st place: +1.0
- 2nd place: +0.5
- 3rd place: -0.3
- 4th place: -1.0

The second part gives a small reward during the game every time the agent plays a card and reduces their hand. The formula is based on potential-based shaping which means it cannot change what the best strategy is, it just gives more frequent feedback so the agent learns faster. Without this the agent only gets feedback at the very end of a 20+ step game.

---

## Experiments and Results

### Experiment 1 - Does the VAE initialisation help?

Both agents trained for 500,000 steps against 3 random opponents.

| | Random-Init | GenAI-Init | Random Baseline |
|---|---|---|---|
| Win Rate | 1.000 | 1.000 | 0.250 |
| Avg Finishing Position | 0.04 | 0.05 | ~1.50 |
| Performance Score | 1.4172 | 1.4403 | - |

Both agents ended up at 100% win rate which shows the training worked well. The difference between them is very small because they both maxed out against random opponents , this is a ceiling effect. Looking at the learning curves (graph 1) the GenAI agent does seem to learn a bit faster in the first few hundred episodes which is what the VAE initialisation is supposed to do.

### Experiment 2 - Does opponent difficulty matter?

Both agents trained for 200,000 steps against harder opponents. Seat 1 played randomly, seat 2 always played the highest card, seat 3 always played the lowest card.

| | Random-Init | GenAI-Init |
|---|---|---|
| Win Rate | 0.960 | 0.950 |
| Avg Finishing Position | 0.66 | 0.72 |
| Performance Score | 1.2029 | 1.1790 |

Win rates dropped compared to experiment 1 which shows the mixed opponents are genuinely harder to beat. The agents still performed well overall but could not dominate as easily.

---

## Reading the Graphs

- **Graph 1** - win rate over training vs random opponents. Both lines go from 0.25 up to nearly 1.0 very quickly
- **Graph 2** - reward over training. Starts near 0 and climbs to around 1.0, shows the reward fix is working
- **Graph 4** - bar chart comparing all 4 agents. Good overview of both experiments side by side
- **Graph 5** - VAE loss going down over 30 epochs. Shows the VAE learned something before RL training started
- **Graph 6** - win rate over training vs mixed opponents. More noisy and slower than graph 1
- **Graph 7** - side by side comparison of both agents in both conditions
- **Graph 8** - summary of everything, Best image to look at

---

## Files

```
chefs-hat-rl/
├── train.py                 # main training script
├── train_mixed.py           # mixed opponents experiment
├── evaluate.py              # runs evaluation on saved models
├── plot_results.py          # generates all the graphs
├── test_env.py              # checks the install worked
├── fix_init.py              # fixes the Windows chefshatgym bug
├── single_agent_wrapper.py  # gym wrapper, reward function, opponent logic
├── vae_init.py              # VAE model and weight transfer code
├── models/                  # saved model files after training
└── results/                 # saved graph images after plotting
```
