import csv
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from os import path
import numpy as np

def read_executions():
        directory = path.join(path.abspath('.'), file_name)
        data = pd.read_csv(directory)
        return data

def graph():
        data = read_executions()
        data_rewards = data.loc[:, ['episode','rewards']]

        #moving average
        window = 100
        data_rewards['moving_average'] = data_rewards.rewards.rolling(window).mean()
        data_rewards['min_value_in_move_average'] = data_rewards.rewards.rolling(window).min()
        data_rewards['max_value_in_move_average'] = data_rewards.rewards.rolling(window).max()
        # print(data_rewards)
        sns.lineplot(x = 'episode', y='rewards', data=data_rewards, label='Reward per episodes')
        sns.lineplot(x = 'episode', y='moving_average', data=data_rewards, label='Reward per episodes')
        
        # plt.fill_between(data_rewards['episode'], data_rewards['min_value_in_move_average'], data_rewards['max_value_in_move_average'], color='green', alpha=0.5) 
        # # # plt.xlabel('Episodes')
        # # # plt.ylabel('Rewards')
        # # # plt.grid()
        # # # plt.savefig('graph_rewards_dqn.png')
        plt.show()

file_name = 'dqn_execution_01.csv'
valores = np.random.randint(0,1000,10) # -457 -506 -917
print(valores)