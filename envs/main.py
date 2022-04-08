from custom_env import CustomEnv
import time

dict_actions = {1:'FORWARD', 2:'TURN_LEFT', 3:'TURN_RIGHT', 4: 'GRAB', 5:'SHOOT'}

env = CustomEnv(4,4)

def Random_games():
    #Each of this episode is its own game
    for episode in range(100):
        env.reset()
        #this is each frame, up to 500...but we wont make it that far with random
        for t in range(500):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(dict_actions[action])

            #lets print everything in one line:
            print(t, next_state, reward, done, info, action)
            if done:
                break
        time.sleep(0.5)
