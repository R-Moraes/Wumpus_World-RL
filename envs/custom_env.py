import gym
from gym import spaces
import numpy
from wumpus_world import WumpusWorld

class CustomEnv(gym.Env):
    def __init__(self, nrow, ncol):
        self.environment = WumpusWorld(nrow)
        self.action_space = spaces.Discrete(5)  # 5 actions: FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT
        self.observation_space = spaces.Discrete(4)
    
    def reset(self):
        initial_state = self.environment.reset_environment()
        return initial_state

    def step(self, action):
        print(action)
        self.environment.move(action)
        state = self.environment.observe()
        reward = self.environment.evaluate()
        done = self.environment.is_done()
        return state, reward, done, {}

    def render(self, mode='human', close=False):
        mat = self.environment.board.get_matrix_env()
        print(self.environment.board.get_board_str(mat))

if __name__ == '__main__':
    env = CustomEnv(4,4)
    env.step('UP')