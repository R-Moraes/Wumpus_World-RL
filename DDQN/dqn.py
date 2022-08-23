import random
from turtle import st
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import random
from argparse import ArgumentParser
from custom_env import CustomEnv

class DeepQAgent:
    def __init__(self, args, env) -> None:
        #setting hyper-parameters and initialize NN model
        # set hyperparameters
        self.max_episodes = 20000
        self.max_actions = 99
        self.discount = 0.93
        self.exploration_rate = 1.0
        self.exploration_decay = 1.0/20000
        # get envirionment
        self.env = env
    
        # nn_model parameters
        self.in_units = env.observation_space.n
        self.out_units = env.action_space.n
        self.hidden_units = int(args.hidden_units)
        
        # construct nn model
        self._nn_model()
    
        # save nn model
        self.saver = tf.compat.v1.train.Saver()
    
    def _nn_model(self):
        #build NN model

        self.a0 = tf.compat.v1.placeholder(tf.float32, shape=[1,self.in_units]) #input layer
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[1,self.out_units]) #output layer

        # from input layer to hidden layer
        self.w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32)) # weight
        self.b1 = tf.Variable(tf.random.uniform([self.hidden_units], 0, 0.01, dtype=tf.float32)) # bias
        self.a1 = tf.nn.relu(tf.matmul(self.a0, self.w1) + self.b1) # the output of hidden layer

        # from hidden layer to output layer
        self.w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32)) # weight
        self.b2 = tf.Variable(tf.random.uniform([self.out_units], 0, 0.01, dtype=tf.float32)) # bias

        # Q-Value and Action
        self.a2 = tf.matmul(self.a1, self.w2) + self.b2 # the predict_y (Q-Value) of four actions
        self.action = tf.argmax(self.a2, 1) # the agent would take the action which has maximum Q-Value


        # loss function
        self.loss = tf.reduce_sum(tf.square(self.a2-self.y))

        #update model
        self.update_model = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self.loss)
    
    def train(self):
        #training the model
        #get hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay

        #start training
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer()) #initialize tf variables
            for i in range(max_episodes):
                state = env.reset() # reset the environment per episodes
                for j in range(max_actions):
                    env.render()
                    # get action and Q-values of all actions
                    print(f'Action: {self.action}\nA2: {self.a2}\nState: {state}\n')
                    
                    action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:state})
                    exit()
                    # if exploration, then taking a random action instead
                    if np.random.rand() < exploration_rate:
                        action[0] = env.action_space.sample()
                    
                    # get nextQ in given next_state
                    next_state, reward, done, info = env.step(action[0])
                    nextQ = sess.run(self.a2, feed_dict={self.a0:np.eye(16)[next_state:next_state+1]})

                    #update
                    update_Q = pred_Q
                    update_Q [0, action[0]] = reward + discount * np.max(nextQ)

                    sess.run([self.update_model], 
                            feed_dict={self.a0:np.identity(16)[state:state+1], self.y:update_Q})
                    
                    state = next_state

                    # if fall in the hole or arrive to the goal, then this episode is terminated.
                    if done:
                        if exploration_rate > 0.001:
                            exploration_rate -= exploration_decay
                        break

            # save model
            save_path = self.saver.save(sess, "./nn_model.ckpt")

    
    def test(self):
        #testing the agent
        # get hyper-parameters
        max_actions = self.max_actions
        # start testing
        with tf.compat.v1.Session() as sess:
            # restore the model
            sess.run(tf.compat.v1.global_variables_initializer())
            saver=tf.compat.v1.train.import_meta_graph("./nn_model.ckpt.meta") # restore model
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))# restore variables
            
            # testing result
            state = env.reset()
            for j in range(max_actions):
                env.render() # show the environments
                # always take optimal action
                action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:np.eye(16)[state:state+1]})
                # update
                next_state, rewards, done, info = env.step(action[0])
                state = next_state
                if done:
                    env.render()
                    break
    
    def display(self):
        #show information
        pass

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--max_episodes", help="max training episode", default=20000)
    parser.add_argument("--max_actions", help="max actions per episode", default=99)
    parser.add_argument("--discount", help="discount factor for Q-learning", default=0.95)
    parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
    parser.add_argument("--hidden_units", help="hidden units", default=10)
    return parser.parse_args()

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    args = arg_parse() # get hyper-parameters
    env = gym.make('CartPole-v1') # construct the environment
    agent = DeepQAgent(args, env) # get agent
    print("START TRAINING...")
    agent.train()
    print("\n==============\nTEST==============\n")
    agent.test()