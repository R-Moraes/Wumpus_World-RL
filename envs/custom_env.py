import gym
from gym import spaces
import numpy
from wumpus_world import WumpusWorld
import time

class CustomEnv(gym.Env):
    def __init__(self, nrow, ncol):
        self.environment = WumpusWorld(nrow)
        self.action_space = spaces.Discrete(5)  # 5 actions: FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT
        # 8 observations: POSITION_AGENT, DIRECTION, BREEZE, STENCH, GLITTER, ARROW, WUMPUS IS ALIVE, DISTANCE_TO_GOLD
        self.observation_space = spaces.Discrete(7)
    
    def reset(self):
        initial_state = self.environment.reset_environment()
        return initial_state

    def step(self, action):
        self.environment.move(action)
        reward = self.environment.evaluate()
        state = self.environment.observe()
        done = self.environment.is_done()
        return state, reward, done, {}

    def render(self, mode='human', close=False):
        mat = self.environment.board.get_matrix_env()
        print(self.environment.board.get_board_str(mat))

if __name__ == '__main__':
    env = WumpusWorld(4)
    env.board.components['Pit0'].pos = (3,1)
    env.board.components['Pit1'].pos = (1,2)
    env.board.components['Pit2'].pos = (0,2)
    env.board.components['Gold'].pos = (2,1)
    env.board.components['Wumpus'].pos = (0,3)
    mat = env.board.get_matrix_env()
    mat_sen = env.board.get_matrix_sensations()
    print(env.board.get_board_str(mat))
    print(env.observe())
    env.board.components['Agent'].pos = (1,3)
    mat = env.board.get_matrix_env()
    mat_sen = env.board.get_matrix_sensations()
    print(env.board.get_board_str(mat))
    print(env.observe())
    print('==SHOOT==')
    env.move('SHOOT')
    print(env.observe())
    print(f'is done: {env.is_done()}')
    print(f'reward: {env.evaluate()}, Wumpus alive: {env.board.components["Agent"].wumpus_alive}')
    print('==TURN RIGHT==')
    env.move('TURN_RIGHT')
    print(env.observe())
    print(f'is done: {env.is_done()}')
    print(f'reward: {env.evaluate()}, Wumpus alive: {env.board.components["Agent"].wumpus_alive}')
    print('==SHOOT==')
    env.move('SHOOT')
    print(env.observe())
    print(f'is done: {env.is_done()}')
    print(f'reward: {env.evaluate()}, Wumpus alive: {env.board.components["Agent"].wumpus_alive}')
    print('==TURN_LEFT==')
    env.move('TURN_LEFT')
    print(env.observe())
    print(f'is done: {env.is_done()}')
    print(f'reward: {env.evaluate()}, Wumpus alive: {env.board.components["Agent"].wumpus_alive}')
    print('==FORWARD==')
    env.move('FORWARD')
    print(env.observe())
    print(f'is done: {env.is_done()}')
    print(f'reward: {env.evaluate()}, Wumpus alive: {env.board.components["Agent"].wumpus_alive}')