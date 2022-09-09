from random import random
import numpy as np
import time
from os import path
from matplotlib import pyplot as plt

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
        # get hyper-parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        learning_rate = self.learning_rate
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay

        total_rewards = 0
        # start training
        record_rewards = []
        list_rewards = []
        amount_grab_gold = 0
        amount_dead_wumpus = 0
        has_gold_safe_home = 0
        steps_environment = []
        self.env.render()
        time.sleep(5)
        for i in range(max_episodes):
            print(f'Episode {i}')
            amount_steps_environment = 0
            total_rewards = 0
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
                    amount_grab_gold = amount_grab_gold + 1 if self.env.environment.board.components['Agent'].has_gold else amount_grab_gold
                    amount_dead_wumpus = amount_dead_wumpus + 1 if not self.env.environment.board.components['Agent'].wumpus_alive else amount_dead_wumpus
                    if next_state == 0 and self.env.environment.board.components['Agent'].has_gold:
                        has_gold_safe_home += 1
                    steps_environment.append(amount_steps_environment)
                    break
            print(f'reward of episode {i}: {total_rewards}')        
            record_rewards.append(total_rewards)
            list_rewards.append(total_rewards)
            
            if exploration_rate > 0.001:
                # exploration_rate -= exploration_decay
                exploration_rate = 0.01 + (exploration_rate-0.01)*np.exp(-exploration_decay*(i+1))
            
            if i%100==0 and i>0:
                self.env.render()
                average_rewards = np.mean(np.array(record_rewards))
                record_rewards = []
                print("episodes: %i to %i, average_reward: %.3f, exploration: %.3f" %(i-100, i, average_rewards, exploration_rate))
                print(f'Amount_has_gold: {amount_grab_gold}')
                amount_grab_gold = 0
                print(f'Amount_dead_wumpus: {amount_dead_wumpus}')
                amount_dead_wumpus = 0
                print(f'Has_Gold_And_Safe_Home: {has_gold_safe_home}')
                has_gold_safe_home = 0
                print(f'State last episode: {next_state}\nPosition of Agent in the last episode: {next_state}')
                row, col = self.env.environment.get_pos_agent()
                print('Position(Matrix) of Agent in the last episode: (%i,%i)' %(row,col))
                avg_steps = np.mean(np.array(steps_environment))
                print(f'Average steps in the environment: {avg_steps}')
                print(f'Steps in last episode: {amount_steps_environment}')
                last_action = self.env.environment.board.components['Agent'].last_action
                print(f'last action: {last_action}')
                time.sleep(5)
        self.write_executions(list_rewards)
    
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
    
    def write_executions(self, rewards):
        list_rewards = np.array(rewards)
        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions')
        with open(path.abspath(path.join(directory,'execution_01.npy')), 'ab+') as file:
            np.save(file, list_rewards)

    def read_executions(self):
        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\execution_01.npy')

        with open(directory, 'rb') as file:
            all_rewards = np.load(file)

        return all_rewards

    def graph(self):
        all_rewards = self.read_executions()
        x = np.arange(self.max_episodes)

        fig, ax = plt.subplots()
        ax.plot(x, all_rewards)

        ax.set(xlabel='Episodes', ylabel='Rewards', title='DDQN')
        ax.grid()

        # fig.savefig('ddqn.png')
        plt.show()

    def reset_data(self):
        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\execution_01.npy')
        open(directory,"wb").close()
