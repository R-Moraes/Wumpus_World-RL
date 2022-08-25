import gym 
from experience_replay import ExpReplay
from QNET import QNET
from TNET import TNET
import numpy as np
import tensorflow.compat.v1 as tf
from custom_env import CustomEnv
import time
import os

dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}

class Agent():
    def __init__(self, env):
        # set hyper parameters
        self.max_episodes = 10000
        self.max_actions = 10000
        self.exploration_rate = 1.0
        self.exploration_decay = 0.0001  
        
        # set environment
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        
        # Experience Replay for batch learning
        self.exp = ExpReplay()
        # the number of experience per batch for batch learning
        self.batch_size = 64 
        
        # Deep Q Network
        self.qnet = QNET(self.states, self.actions, self.exp)
        # For execute Deep Q Network
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())
        #self.qnet.set_session(session)
        self.qnet.session = session
        
    def train(self):
        # set hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        batch_size = self.batch_size
        
        # start training
        record_rewards = []
        amount_grab_gold = 0
        amount_dead_wumpus = 0
        has_gold_safe_home = 0
        steps_environment = []
        for i in range(max_episodes):
            print(f'Episode {i}')
            amount_steps_environment = 0
            if i%100==0:
                self.env.render()
            total_rewards = 0
            state = self.env.reset()
            if i%100==0:
                self.env.render()
            state = state.reshape(1, self.states)
            for j in range(max_actions):
                #self.env.render() # Uncomment this line to render the environment
                action = self.qnet.get_action(state, exploration_rate)
                next_state, reward, done, info = self.env.step(dict_actions[action])
                next_state = next_state.reshape(1, self.states)
                total_rewards += reward
                amount_steps_environment += 1
                
                if done:
                    amount_grab_gold = amount_grab_gold + 1 if self.env.environment.board.components['Agent'].has_gold else amount_grab_gold
                    amount_dead_wumpus = amount_dead_wumpus + 1 if not self.env.environment.board.components['Agent'].wumpus_alive else amount_dead_wumpus
                    if state[0][0] == 0 and self.env.environment.board.components['Agent'].has_gold:
                        has_gold_safe_home += 1
                    steps_environment.append(amount_steps_environment)
                    self.exp.add(state, action, reward, next_state, done)
                    self.qnet.batch_train(batch_size)
                    break
                    
                self.exp.add(state, action, reward, next_state, done)
                self.qnet.batch_train(batch_size)
                
                # update target network
                if (j%25)== 0 and j>0:
                    self.qnet.update()
                # next episode
                state = next_state
                
            record_rewards.append(total_rewards)
            exploration_rate = 0.01 + (exploration_rate-0.01)*np.exp(-exploration_decay*(i+1))
            if i%100==0 and i>0:
                average_rewards = np.mean(np.array(record_rewards))
                record_rewards = []
                print("episodes: %i to %i, average_reward: %.3f, exploration: %.3f" %(i-100, i, average_rewards, exploration_rate))
                print(f'Amount_has_gold: {amount_grab_gold}')
                amount_grab_gold = 0
                print(f'Amount_dead_wumpus: {amount_dead_wumpus}')
                amount_dead_wumpus = 0
                print(f'Has_Gold_And_Safe_Home: {has_gold_safe_home}')
                has_gold_safe_home = 0
                print(f'State last episode: {state}\nPosition of Agent in the last episode: {state[0][0]}')
                row, col = self.env.environment.board.components['Agent'].pos
                print('Position(Matrix) of Agent in the last episode: (%i,%i)' %(row,col))
                avg_steps = np.mean(np.array(steps_environment))
                print(f'Average steps in the environment: {avg_steps}')


if __name__ == '__main__':
    tf.disable_eager_execution()
    env = CustomEnv(4,4)
    agent = Agent(env)
    agent.train()