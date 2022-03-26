from importlib.metadata import entry_points
from gym.envs.registration import register


register(
    id='WumpusWorld-V0',
    entry_points='gym_game.envs:CustomEnv',
    max_episode_steps=2000
)