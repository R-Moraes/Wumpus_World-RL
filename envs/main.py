from custom_env import CustomEnv



env = CustomEnv(4,4)
env.render()
pos = env.environment.get_pos_agent()
print(env.environment.board.get_sensations(pos))
state = env.step('UP')
print(state)
env.render()
pos = env.environment.get_pos_agent()
print(env.environment.board.get_sensations(pos))
