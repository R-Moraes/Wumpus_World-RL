import gym
from io import StringIO
from contextlib import closing
from gym import spaces
import numpy
from wumpus_world import WumpusWorld
import time
import numpy as np

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
        outfile = StringIO()
        mat = self.environment.board.get_matrix_env()
        # print(self.environment.board.get_board_str(mat))
        mat_env = self.environment.board.get_board_str_two(mat)
        desc = np.asarray(mat_env, dtype="c")
        r = desc.tolist()
        r = [[c.decode("utf-8") for c in line] for line in r]
        i = 0
        for values in  r:
            if 'A' in values:
                j = values.index('A')
                print(f'i: {i} and j: {j}')
                break
            else:
                i += 1
        r[i][j] = gym.utils.colorize(r[i][j], "red", highlight=True)
        outfile.write("\n".join("".join(line) for line in r) + "\n")
        with closing(outfile):
            print(outfile.getvalue())


if __name__ == '__main__':
    env = CustomEnv(4,4)
    env.environment.board.components['Pit0'].pos = (3,1)
    env.environment.board.components['Pit1'].pos = (1,2)
    env.environment.board.components['Pit2'].pos = (0,2)
    env.environment.board.components['Gold'].pos = (2,1)
    env.environment.board.components['Wumpus'].pos = (0,3)
    env.render()
    env.environment.board.components['Agent'].pos = (1,1)
    env.step('TURN_LEFT')
    print(env.environment.board.components['Agent'].direction)
    env.step('FORWARD')
    env.render()
    env.step('FORWARD')
    env.render()