import tensorflow as tf
from tensorflow.compat import v1 as tfv1
import numpy as np
from experience_replay import ExpReplay
from custom_env import CustomEnv
from QNET import QNET

dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}

class DeepQAgent:
    def __init__(self, env, hidden_units=256):
        # set hyper parameters
        self.max_episodes = 10000
        #self.max_actions = 10000
        self.exploration_rate = 1.0
        self.exploration_decay = 0.0001  
        
        # set environment
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.max_actions = self.env.environment.board.max_steps
        
        # Experience Replay for batch learning
        self.exp = ExpReplay()
        # the number of experience per batch for batch learning
        self.batch_size = 64 
        
        # Deep Q Network
        self.qnet = QNET(self.states, self.actions, self.exp)
        # For execute Deep Q Network
        session = tfv1.InteractiveSession()
        session.run(tfv1.global_variables_initializer())
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
        list_rewards = []
        amount_grab_gold = 0
        amount_dead_wumpus = 0
        has_gold_safe_home = 0
        steps_environment = []
        self.env.render()
        for i in range(max_episodes):
            print(f'Episode {i}')
            amount_steps_environment = 0
            total_rewards = 0
            state = self.env.reset()

            state = state.reshape((1, self.states))

            for j in range(max_actions):
                #self.env.render() # Uncomment this line to render the environment
                action = self.qnet.get_action(state, exploration_rate)
                next_state, reward, done, info = self.env.step(dict_actions[action])
                next_state = next_state.reshape((1, self.states))

                total_rewards += reward
                
                if done:
                    steps_environment.append(amount_steps_environment)
                    self.exp.add(state, action, reward, next_state, done)
                    self.qnet.batch_train(batch_size)
                    
                    break
                    
                self.exp.add(state, action, reward, next_state, done)
                self.qnet.batch_train(batch_size)
                
                # update target network
                # if (j%25)== 0 and j>0:
                #     self.qnet.update()
                
                # next episode
                state = next_state
            print(f'reward of episode {i}: {total_rewards}')        
            record_rewards.append(total_rewards)
            list_rewards.append(total_rewards)
            exploration_rate = 0.01 + (exploration_rate-0.01)*np.exp(-exploration_decay*(i+1))
    
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

    def reset_data():
        directory = path.join(path.abspath('.'), 'gym_game\Q_Learning\executions\execution_01.npy')
        open(directory,"wb").close()

tfv1.disable_eager_execution()
env = CustomEnv(4,4, 100)
agent = DeepQAgent(env)

agent.train()