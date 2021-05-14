#================================================================ 
#   Idea by      : PyLessons
#   Created date: 2021-05-14
#   Description : Trading Crypto with Reinforcement Learning 
#================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import pandas as pd
from indicators import AddIndicators
from utils import Normalizing
from agent import TestAgent
from env import CustomEnv

import pdb
def d():
    pdb.set_trace()


 
def test_agent(test_df, test_df_nomalized, visualize=True, test_episodes=10, comment="", show_reward=False, show_indicators=False):
    with open("train/newest","r") as f:
        sub_path = f.read()
        newst_train_folder = "train/{}".format(sub_path)
    params_file = "{}/{}".format(newst_train_folder,"parameters.json")
    with open(params_file, "r") as json_file:
        params = json.load(json_file)
    agent = TestAgent(**params)
    env = CustomEnv(df=test_df, df_normalized=test_df_nomalized, lookback_window_size=params["lookback window size"]
        , show_reward=show_reward, show_indicators=show_indicators)

    agent.load(folder=newst_train_folder)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break

    print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


if __name__ == "__main__":            
    df = pd.read_csv('./BTCUSD_1h.csv')
    df = df.dropna()
    df = df.sort_values('Date')
    df = AddIndicators(df) # insert indicators to df 2021_02_21_17_54_Crypto_trader
    df_nomalized = Normalizing(df[99:])[1:].dropna()
    df = df[100:].dropna()
    lookback_window_size = 100
    test_window = 24 * 30 * 3 # 3 months
    test_df = df[-test_window-lookback_window_size:]
    test_df_nomalized = df_nomalized[-test_window-lookback_window_size:]
    test_agent(test_df,test_df_nomalized)
