# Fix test_env.py to use a log directory
path = 'test_env.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the startExperiment call to include a log directory
old = '''    env.startExperiment(
        playerNames=["RL_Agent", "Random_1", "Random_2", "Random_3"],
        logDirectory="",        # empty = no log files created
        verbose=0,
        saveLog=False,
        saveDataset=False,
        gameType="MATCHES",
        stopCriteria=3,         # game ends after 3 matches
        maxInvalidActions=5,
    )'''

new = '''    import os
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
    )'''

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print("Fixed test_env.py!")
