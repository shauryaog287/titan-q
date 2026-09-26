import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

print("--- Phase 1: Fetching Market Data ---")
market_data = yf.download("^NSEI", start="2019-01-01", end="2024-01-01", progress=False)

df = market_data.copy()
if isinstance(df.columns, pd.MultiIndex): 
    df.columns = [c[0] for c in df.columns]

print("--- Phase 2: Adding Technical Indicators (SMA & RSI) ---")
df.ta.sma(length=20, append=True)
df.ta.rsi(length=14, append=True)
df.dropna(inplace=True)

print("--- Phase 3: Building the Virtual Trading Simulator ---")
def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()
    signal_features = env.df.loc[:, ['Close', 'Volume', 'SMA_20', 'RSI_14']].to_numpy()[start:end]
    return prices, signal_features

class NiftyTradingEnv(StocksEnv):
    _process_data = add_signals

# We will train the AI on the first 1000 days of data
env_train = NiftyTradingEnv(df=df, window_size=10, frame_bound=(10, 1000))
env_maker = lambda: env_train
vec_env = DummyVecEnv([env_maker])

print("--- Phase 4: Training the AI Brain ---")
print("Training on 10,000 simulations... please wait.")
model = A2C("MlpPolicy", vec_env, verbose=0)
model.learn(total_timesteps=10000)
print("Training Complete!")

print("--- Phase 5: The Final Exam (Fictional Trading on Unseen Data) ---")
# We test the AI on data it has NEVER seen before (Day 1000 to the end)
env_test = NiftyTradingEnv(df=df, window_size=10, frame_bound=(1000, len(df)))

obs, info = env_test.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated
    if done:
        print("\n=====================================")
        print("📊 FINAL EXAM RESULTS:")
        print("=====================================")
        print(f"Total Profit Ratio: {info['total_profit']:.4f}")
        
        # Calculate actual points gained/lost
        start_price = env_test.df['Close'].iloc[env_test.frame_bound[0]]
        end_price = env_test.df['Close'].iloc[env_test.frame_bound[1] - 1]
        market_return = end_price / start_price
        
        print(f"If you just bought and held NIFTY: {market_return:.4f}x your money")
        print(f"AI Trading Performance: {info['total_profit']:.4f}x your money")
        print("=====================================")
        break
