import sys
from os import path
sys.path.append(path.join(path.abspath('.'), 'gym_game'))
sys.path.append(path.join(path.abspath('.'), 'gym_game','env'))

from random import random
import numpy as np
import time
from custom_env import CustomEnv
from epsilon_methods import exponential_decay_method, decrement_epsilon
import csv
import datetime
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}

class QAgent():
    def __init__(self, env, max_episodes):
        # set hyperparameters
        self.max_episodes = max_episodes   #set max training episodes
        self.max_actions = env.environment.board.max_steps       # set max actions per episodes
        self.learning_rate = 0.83   # for q-learning
        self.discount = 0.93        # for q-learning
        self.exploration_rate = 1.0 # for exploration
        self.exploration_decay = 0.0001 # for exploration
        self.epsilon_min = 0.1

        # get environment
        self.env = env

        # Initialize Q(s, a)
        row = env.observation_space.n
        col = env.action_space.n
        self.Q = np.zeros((row, col))
    
    def _police(self, mode, state, e_rate=0):
        if mode == 'train':
            if random() > e_rate:
                return np.argmax(self.Q[state, :])     # exploitation
            else:
                return self.env.action_space.sample()   # exploration
        elif mode == 'test':
            return np.argmax(self.Q[state, :])  # optimal policy
    
    def train(self):
        #RECORD INFO EPISODES
        list_info_train = []
        info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
                    'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}
        # get hyper-parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        learning_rate = self.learning_rate
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay

        # start training
        total_rewards = 0
        grab_gold = 0
        has_gold_safe_home = 0
        killed_wumpus = 0

        self.env.render()
        time.sleep(5)
        for i in range(max_episodes):
            print(f'Episode {i}')
            amount_steps_environment = 0
            total_rewards = 0
            grab_gold = 0
            killed_wumpus = 0
            has_gold_safe_home = 0
            info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
                    'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}

            state = self.env.reset()

            for a in range(max_actions):
                action = self._police('train', state, exploration_rate)
                next_state, reward, done, info = self.env.step(dict_actions[action])

                # The formulation of updating Q(s, a)
                self.Q[state, action] = self.Q[state, action] + \
                    learning_rate * (reward + discount*np.max(self.Q[next_state, :]) - self.Q[state, action])
                state = next_state
                total_rewards += reward
                amount_steps_environment += 1

                if done:
                    #record information of the episode
                    grab_gold = 1 if self.env.environment.board.components['Agent'].has_gold else 0
                    killed_wumpus = 1 if not self.env.environment.board.components['Agent'].wumpus_alive else 0
                    if next_state == 0 and self.env.environment.board.components['Agent'].has_gold:
                        has_gold_safe_home = 1
                    info_train['episode'] = i
                    info_train['rewards'] = total_rewards
                    info_train['has_gold'] = grab_gold
                    info_train['step_per_episode'] = amount_steps_environment
                    info_train['killed_wumpus'] = killed_wumpus
                    info_train['get_gold_and_return_home'] = has_gold_safe_home
                    list_info_train.append(info_train)

                    break
            print(f'Reward of episode {i}: {total_rewards}\nEpsilon: {exploration_rate}')
            
            # if exploration_rate > 0.001:
            #     # exploration_rate -= exploration_decay
            #     exploration_rate = 0.01 + (exploration_rate-0.01)*np.exp(-exploration_decay*(i+1))
            exploration_rate = exponential_decay_method(i, max_episodes, self.epsilon_min)
            
        self.write_executions(list_info_train) 
    
    def test(self):
        # Setting hyper-parameters
        max_actions = self.max_actions
        state = self.env.reset() # reset the environment
        for a in range(max_actions):
            self.env.render() # show the environment states
            action = np.argmax(self.Q[state,:]) # take action with the Optimal Policy
            next_state, reward, done, info = self.env.step(action) # arrive to next_state after taking the action
            state = next_state # update current state
            if done:
                print("======")
                self.env.render()
                break
            print("======")
        self.env.close()
    
    def write_executions(self, infos_train: list):
        headers = ['episode', 'rewards', 'has_gold', 'step_per_episode', 'killed_wumpus', 'get_gold_and_return_home']

        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\\')
        with open(path.abspath(path.join(directory, file_name)), 'w') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(infos_train)

    def read_executions(self):
        date = '2022-10-15'
        file_name_date_exec = f'ql_execution_{dim}x{dim}-{date}.csv'
        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\\', file_name_date_exec)
        data = pd.read_csv(directory)

        return data

    def graph(self):
        data = self.read_executions()

        #moving average
        window = 100
        data['moving_average'] = data.rewards.rolling(window).mean()

        sns.lineplot(x = 'episode', y='rewards', data=data, label='Reward per episodes')
        sns.lineplot(x='episode', y='moving_average', data=data, label='Move Average Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.grid()
        plt.savefig(f'graph_rewards_ql_{self.env.environment.board.size_env}x{self.env.environment.board.size_env}.png')
        # plt.show()

def reset_data():
    directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\\', file_name)
    open(directory,"wb").close()

if __name__ == '__main__':
    dict_max_steps = {4: 100, 8: 150, 10: 200} #size environment is key and value is amount max steps
    dict_values_seed = {4: 123, 8: 99, 10: 917} #size environment is key and value is values seed
    dim = 4
    date_execution = datetime.datetime.now().strftime('%Y-%m-%d')
    file_name = f'ql_execution_{dim}x{dim}-{date_execution}.csv'
    # reset_data()
    max_episodes = 20000
    env = CustomEnv(nrow=dim,ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])
    agent = QAgent(env, max_episodes=max_episodes)
    # agent.train()
    # agent.graph()
