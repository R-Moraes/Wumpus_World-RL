from Q_Agent import QAgent
from custom_env import CustomEnv


if __name__ == '__main__':
    max_episodes = 20000
    max_steps = 100
    env = CustomEnv(4,4, max_steps)
    agent = QAgent(env, max_episodes)
    # agent.reset_data()
    # agent.train()
    agent.graph()