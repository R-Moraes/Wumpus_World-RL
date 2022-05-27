from custom_env import CustomEnv
from dqn import DQNAgent
import time
import os
import numpy as np

dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}

def Random_games():
    #Each of this episode is its own game
    for episode in range(2):
        state = env.reset()
        print('Episode: {}'.format(episode))
        env.render()
        print('Initial position of agent: {}'.format(env.environment.board.components['Agent'].pos))
        print('Initial state of board: {}'.format(state))

        #this is each frame, up to 500...but we wont make it that far with random
        for t in range(500):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            # env.render()

            # This will just create a sample action in any environment
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(dict_actions[action])

            #lets print everything in one line:
            print(t, next_state, reward, done, info, action)
            if done:
                break
        time.sleep(0.5)

env = CustomEnv(4,4)

state_size = env.observation_space.n
action_size = env.action_space.n
batch_size = 64
n_episodes = 1000
output_dir = 'model_output/wumpus/'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

agent = DQNAgent(state_size, action_size)

for e in range(n_episodes):
    print('Episode: {}'.format(e))
    state = env.reset()
    print('Initial state: {}'.format(state))
    state = np.reshape(state, [1, state_size])
    env.render()
    done = False 
    time = 0
    while not done:
        #env.render()
        action = agent.act(state)
        print('Action: {} || Direction: {} || Last Action: {} || Amount_step: {} '.format(dict_actions[action],
            env.environment.board.components['Agent'].direction,
            env.environment.board.components['Agent'].last_action,        
            env.environment.board.components['Agent'].amount_steps))
        next_state, reward, done, _ = env.step(dict_actions[action])
        print(next_state, reward, done, dict_actions[action])
        next_state = np.reshape(next_state, [1, state_size]) 
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += reward
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, n_episodes-1, time, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.train(batch_size)