import gym
from io import StringIO
from contextlib import closing
from gym import spaces
import numpy
from wumpus_world import WumpusWorld
import time
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, nrow, ncol, max_steps, value_seed):
        self.environment = WumpusWorld(nrow, max_steps, value_seed)
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
                break
            else:
                i += 1
        r[i][j] = gym.utils.colorize(r[i][j], "red", highlight=True)
        outfile.write("\n".join("".join(line) for line in r) + "\n")
        with closing(outfile):
            print(outfile.getvalue())


# if __name__ == '__main__':
#     size = 10
#     env = CustomEnv(size,size, 200)
#     env.render()
