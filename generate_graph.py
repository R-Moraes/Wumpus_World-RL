import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from os import path
import seaborn as sns

def read_executions():
    dir_abs = path.join(path.abspath('.'), 'gym_game')
    dir_files = [path.join(dir_abs, name_dir, 'executions') for name_dir in ['DDQN', 'DQN', 'Q_Learning']]
    code_files = ['ddqn', 'dqn', 'ql']

    ddqn_directory = path.join(dir_files[0], code_files[0] + file_name)
    ddqn_data = pd.read_csv(ddqn_directory)
    dqn_directory = path.join(dir_files[1], code_files[1] + file_name)
    dqn_data = pd.read_csv(dqn_directory)
    ql_directory = path.join(dir_files[2], code_files[2] + file_name)
    ql_data = pd.read_csv(ql_directory)

    return ddqn_data, dqn_data, ql_data

def graph():
        ddqn_data, dqn_data, ql_data = read_executions()

        #moving average
        window = 100
        ddqn_data['moving_average'] = ddqn_data.rewards.rolling(window).mean()
        dqn_data['moving_average'] = dqn_data.rewards.rolling(window).mean()
        ql_data['moving_average'] = ql_data.rewards.rolling(window).mean()

        print(ddqn_data.rewards.max(), dqn_data.rewards.max(), ql_data.rewards.max())

        # sns.lineplot(x = 'episode', y='rewards', data=ddqn_data, label='Reward per episodes')
        sns.lineplot(x='episode', y='moving_average', data=ddqn_data, label='Move Average Rewards DDQN')
        sns.lineplot(x='episode', y='moving_average', data=dqn_data, label='Move Average Rewards DQN')
        sns.lineplot(x='episode', y='moving_average', data=ql_data, label='Move Average Rewards Q_Learning')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.grid()
        # plt.savefig(f'graph_rewards_ql_{dim}x{dim}.png')
        plt.show()

if __name__ == '__main__':
    dict_max_steps = {4: 100, 8: 150, 10: 200} #size environment is key and value is amount max steps
    dict_values_seed = {4: 123, 8: 99, 10: 917} #size environment is key and value is values seed
    dim = 10
    date = '2022-10-15'
    file_name = f'_execution_{dim}x{dim}-{date}.csv'
    graph()
    