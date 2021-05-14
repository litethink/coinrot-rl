#================================================================ 
#   Idea by      : PyLessons
#   Created date: 2021-05-14
#   Description : Trading Crypto with Reinforcement Learning 
#================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import numpy as np
import pandas as pd
from collections import deque
from tensorflow.keras.optimizers import Adam

from agent import TrainAgent
from env import CustomEnv
from utils import Normalizing
from indicators import AddIndicators



import pdb
def d():
    pdb.set_trace()


agent_config = {
    "model" :"CNN",
    "lr" : 0.00001,
    "epochs" : 5,
    "batch_size" : 32
}

env_config = {
    "initial_balance" : 1000,
    "render_range" : 100,
    "show_reward" : False,
    "show_indicators" : False,
    "normalize_value" : 40000
}

def train_agent(env, agent, visualize=False, train_episodes = 50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            #line
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
        a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()
    with open("train/newest","w") as f:
        f.write(agent.log_name)
    print("Train done.")

if __name__ == "__main__":            
    df = pd.read_csv('./BTCUSD_1h.csv')
    df = df.dropna()
    df = df.sort_values('Date')
    #d()

    #df columns :['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'sma7', 'sma25','sma99', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'psar', 'RSI']
    df = AddIndicators(df) # insert indicators to df 2021_02_21_17_54_Crypto_trader
    #df = indicators_dataframe(df, threshold=0.5, plot=False) # insert indicators to df 2021_02_18_21_48_Crypto_trader
    depth = len(list(df.columns[1:])) # OHCL + indicators without Date
    df_nomalized = Normalizing(df[99:])[1:].dropna()
    df = df[100:].dropna()
    lookback_window_size = 100
    test_window = 24 * 30 * 3 # 3 months
    #     # split training and testing datasets
    train_df = df[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
    #     # split training and testing normalized datasets
    train_df_nomalized = df_nomalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
    #     # single processing training
    agent = TrainAgent(lookback_window_size=lookback_window_size,depth=depth,**agent_config)
    train_env = CustomEnv(df=train_df, df_normalized=train_df_nomalized, lookback_window_size=lookback_window_size,**env_config)
    train_agent(train_env, agent, visualize=False, train_episodes=500, training_batch_size=500)

