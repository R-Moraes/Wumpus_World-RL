import gym
from gym import spaces
import numpy
from wumpus_world import WumpusWorld

class CustomEnv(gym.Env):

    def __init__(self, nrow, ncol):
        self.environment = WumpusWorld(nrow)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(nrow*ncol)
    
    def reset(self):
        self.environment.reset_environment()

    def step(self, action):
        self.environment.move(action)
        state = self.environment.observe()
        return state

    def render(self, mode='human', close=False):
        mat = self.environment.board.get_matrix_env()
        print(self.environment.board.get_board_str(mat))

if __name__ == '__main__':
    env = CustomEnv(4,4)
    env.render()