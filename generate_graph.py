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

        steps_data_ql = ql_data.loc[(ql_data['has_gold'] == 1)]
        steps_data_dqn = dqn_data.loc[(dqn_data['has_gold'] == 1)]
        steps_data_ddqn = ddqn_data.loc[(ddqn_data['has_gold'] == 1)]
        # num_win_ql = (ql_data.groupby('killed_wumpus').size()/(20000))*100
        # num_win_dqn = (dqn_data.groupby('killed_wumpus').size()/(20000))*100
        # num_win_ddqn = (ddqn_data.groupby('killed_wumpus').size()/(20000))*100
        # print(f'QL: {num_win_ql}')
        # print(f'DQN: {num_win_dqn}')
        # print(f'DDQN: {num_win_ddqn}')
        # print(ql_data[ql_data['has_gold'] == 1].rewards.min(), dqn_data[dqn_data['has_gold']==1].rewards.min(), ddqn_data[ddqn_data['has_gold']==1].rewards.min())
        # print(ql_data[ql_data['has_gold'] == 1].rewards.max(), dqn_data[dqn_data['has_gold']==1].rewards.max(), ddqn_data[ddqn_data['has_gold']==1].rewards.max())
        # print(round(ql_data[ql_data['has_gold'] == 1].rewards.std(), 4), round(dqn_data[dqn_data['has_gold']==1].rewards.std(), 4), round(ddqn_data[ddqn_data['has_gold']==1].rewards.std(), 4))
        # print(round(ql_data[ql_data['has_gold'] == 1].rewards.mean(), 4), round(dqn_data[dqn_data['has_gold']==1].rewards.mean(), 4), round(ddqn_data[ddqn_data['has_gold']==1].rewards.mean(), 4))

        # sns.lineplot(x = 'episode', y='step_per_episode', data=ddqn_data, label='Reward per episodes')
        sns.lineplot(x='episode', y='moving_average', data=ddqn_data, label='Moving Average Rewards DDQN')
        sns.lineplot(x='episode', y='moving_average', data=dqn_data, label='Moving Average Rewards DQN')
        sns.lineplot(x='episode', y='moving_average', data=ql_data, label='Moving Average Rewards QL')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.grid()
        plt.savefig(f'graph_rewards_ql_{dim}x{dim}.png')
        # plt.savefig(f'graph_step_per_episode_{dim}x{dim}.png')
        plt.show()

if __name__ == '__main__':
    dict_max_steps = {4: 100, 8: 150, 10: 200} #size environment is key and value is amount max steps
    dict_values_seed = {4: 123, 8: 99, 10: 917} #size environment is key and value is values seed
    dim = 15
    date = '2022-10-15'
    file_name = f'_execution_{dim}x{dim}-{date}.csv'
    graph()
    