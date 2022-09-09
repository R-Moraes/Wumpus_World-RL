import tensorflow as tf
from tensorflow.compat import v1 as tfv1
# from TNET import TNET
import numpy as np


class QNET():
    def __init__(self, in_units, out_units, exp, hidden_units=256):
        # Target Network
        # self.tnet = TNET(in_units, out_units)

        # experience replay
        self.exp = exp
        
        # Q network architecture
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units

        self.session = None
        self._model()
        self._batch_learning_model()
        # self._tnet_update()
        
    def _model(self):
        """ Q-network architecture """
        with tfv1.variable_scope('qnet'):
            self.x = tfv1.placeholder(tf.float32, shape=(None, self.in_units))
            
            W1 = tfv1.get_variable('W1', shape=(self.in_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W2 = tfv1.get_variable('W2', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W3 = tfv1.get_variable('W3', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W4 = tfv1.get_variable('W4', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            W5 = tfv1.get_variable('W5', shape=(self.hidden_units, self.out_units), initializer=tf.random_normal_initializer())
            
            b1 = tfv1.get_variable('b1', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            b2 = tfv1.get_variable('b2', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            b3 = tfv1.get_variable('b3', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            b4 = tfv1.get_variable('b4', shape=(self.hidden_units), initializer=tf.zeros_initializer())
 
            h1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
            h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
            h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)
            h4 = tf.nn.tanh(tf.matmul(h3, W4) + b4)
            self.q = tf.matmul(h4, W5)
            
            
    def update(self):
        """Execution for Target network update"""
        self.session.run(self.update_opt)
    
    def get_action(self, state, e_rate):
        """ for training stage of the Agent, exploitation or exploration"""
        if np.random.random()<e_rate: # exploration
            return np.random.choice(self.out_units)
        else: # exploitation
            return np.argmax(self.session.run(self.q, feed_dict={self.x: state}))

    def _batch_learning_model(self):
        """For batch learning"""
        with tfv1.variable_scope('qnet'):
            # TD-target
            self.target = tfv1.placeholder(tf.float32, shape=(None, ))
            # Action index
            self.selected_idx = tfv1.placeholder(tf.int32, shape=(None, 2))
            # Q-value
            self.selected_q = tf.gather_nd(self.q, self.selected_idx)
            
            self.params = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES, scope='qnet')
            
            # Q-network optimization alogrithms
            loss = tf.losses.mean_squared_error(self.target, self.selected_q)
            gradients = tf.gradients(loss, self.params)
            self.train_opt = tfv1.train.AdamOptimizer(3e-4).apply_gradients(zip(gradients, self.params))
    
    def batch_train(self, batch_size=64):
        """Implement Double DQN Algorithm, batch training"""
        if self.exp.get_num() < self.exp.get_min():
            #The number of experiences is not enough for batch training
            return

        # get a batch of experiences
        state, action, reward, next_state, done = self.exp.get_batch(batch_size)
        state = state.reshape(batch_size, self.in_units)
        next_state = next_state.reshape(batch_size, self.in_units)
        
        # get actions by Q-network
        qnet_q_values = self.session.run(self.q, feed_dict={self.x:next_state})
        qnet_actions = np.argmax(qnet_q_values, axis=1)
        #take pega os valores da lista qnet_q_values nas posicoes qnet_actions
        qnet_q = [np.take(qnet_q_values[i], qnet_actions[i]) for i in range(batch_size)]
        
        # Update Q-values of Q-network
        qnet_update_q = [r+0.95*q if not d else r for r, q, d in zip(reward, qnet_q, done)]
        
        # optimization
        indices=[[i,action[i]] for i in range(batch_size)]
        feed_dict={self.x:state, self.target:qnet_update_q, self.selected_idx:indices}
        self.session.run(self.train_opt, feed_dict)