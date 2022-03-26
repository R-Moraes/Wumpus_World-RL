from custom_env import CustomEnv



env = CustomEnv(4,4)

env.render()
pos = env.environment.board.components['Agent'].pos
print(env.environment.board.get_sensations(pos))
state = env.step('UP')
print(state)
print(env.environment.board.components['Agent'].pos)
env.render()
pos = env.environment.board.components['Agent'].pos
print(env.environment.board.get_sensations(pos))
state = env.step('UP')
print(state)
print(env.environment.board.components['Agent'].pos)
env.render()
pos = env.environment.board.components['Agent'].pos
print(env.environment.board.get_sensations(pos))
