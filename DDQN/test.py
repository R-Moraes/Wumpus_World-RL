from os import path
import numpy as np


def read_executions():
    directory = path.join(path.abspath('.'), 'gym_game\DDQN\executions\execution_01.npy')

    with open(directory, 'rb') as file:
        all_rewards = np.load(file)

    return all_rewards

x = [r for r in read_executions() if r < -500]
print(x)