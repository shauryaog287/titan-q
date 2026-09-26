import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
import numpy as np
from sklearn.preprocessing import StandardScaler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
os.makedirs('./checkpoints/', exist_ok=True)

from sortino_reward_env import SortinoStocksEnv

print("==========================================")
print("[MASTER TRAINING: 50 MILLION STEPS (SORTINO EDITION)]")
print("==========================================")

print("--- 1. Fetching Market Data ---")
import time
for attempt in range(5):
    market_data = yf.download("^NSEI", start="2015-01-01", end="2024-01-01", progress=False)
    if not market_data.empty:
        break
    print(f"Yahoo Finance glitch. Retrying {attempt+1}/5...")
    time.sleep(3)
if market_data.empty:
    print("Using RELIANCE.NS as fallback.")
    market_data = yf.download("RELIANCE.NS", start="2015-01-01", end="2024-01-01", progress=False)
df = market_data.copy()
if isinstance(df.columns, pd.MultiIndex): 
    df.columns = [c[0] for c in df.columns]

print("--- 2. Calculating Indicators ---")
df.ta.sma(length=20, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(append=True)
df.ta.atr(append=True)
df.dropna(inplace=True)

print("--- 3. Preparing the Advanced Sortino Simulator ---")
def make_env():
    env = SortinoStocksEnv(
        df=df,
        window_size=30,
        frame_bound=(30, 1500),
        reward_mode="differential",
        eta=0.02,              # ~50-step risk memory
        trade_fee=0.002,       # 20 bp fee
        reward_scale=0.05,
        lambda_dd=1.0,         # Downside aversion
        l_floor=2.5e-5,        
        augment_observation=True,
    )
    return Monitor(env)

venv = DummyVecEnv([make_env])
# Claude's genius tip: Normalize obs, but NOT the custom reward!
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

print("--- 4. Training the AI (50,000,000 steps) ---")
print("Using Claude's advanced Hyperparameters. The AI is learning...")
# RESUME TRAINING FROM 7M CHECKPOINT
import glob
checkpoint_files = glob.glob('./checkpoints/ppo_sortino_*_steps.zip')
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=venv, device="cuda")
else:
    print("No checkpoint found. Starting from scratch.")
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,          
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,      # Counteracts the hold-forever attractor
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=1000000,
    save_path='./checkpoints/',
    name_prefix='ppo_sortino'
)
model.learn(total_timesteps=43000000, callback=checkpoint_callback, reset_num_timesteps=False)

print("\n--- 5. The Final Exam ---")
# Need to use the same normalization logic for test env
def make_test_env():
    env = SortinoStocksEnv(
        df=df,
        window_size=30,
        frame_bound=(1500, len(df)),
        reward_mode="differential",
        eta=0.02, trade_fee=0.002, reward_scale=0.05, lambda_dd=1.0, l_floor=2.5e-5, augment_observation=True,
    )
    return Monitor(env)

venv_test = DummyVecEnv([make_test_env])
venv_test = VecNormalize(venv_test, norm_obs=True, norm_reward=False, clip_obs=10.0)
venv_test.training = False # Do not update moving averages during test!
venv_test.reward_update = False

obs = venv_test.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = venv_test.step(action)
    if dones[0]:
        info = infos[0]
        print("\n=====================================")
        print("MASTER EXAM RESULTS:")
        print("=====================================")
        print(f"AI Sortino Ratio Achieved: {info.get('terminal_sortino', 0.0)}")
        print("=====================================")
        break

# Save the brain!
model.save("master_ai_brain_sortino")
print("\n[SUCCESS] AI Brain saved locally as 'master_ai_brain_sortino.zip'")
