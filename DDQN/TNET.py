import tensorflow as tf
from tensorflow.compat import v1 as tfv1

class TNET():
    '''
    Target network is for calculating the maximum estimated Q-value in given action a.
    '''

    def __init__(self, in_units, out_units, hidden_units=256):
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units
        self._model()
    
    def _model(self):
        with tf.compat.v1.variable_scope('tnet'):
            # input layer
            self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.in_units))

            #from input layer to hidden layer1
            W1 = tfv1.get_variable('W1', shape=(self.in_units, self.hidden_units), initializer=tf.random_normal_initializer())
            #from hidden layer1 to hidden layer2
            W2 = tfv1.get_variable('W2', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            #from hidden layer2 to hidden layer3
            W3 = tfv1.get_variable('W3', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            #from hidden layer2 to hidden layer3
            W4 = tfv1.get_variable('W4', shape=(self.hidden_units, self.hidden_units), initializer=tf.random_normal_initializer())
            #from hidden layer3 to output layer
            W5 = tfv1.get_variable('W5', shape=(self.hidden_units, self.out_units), initializer=tf.random_normal_initializer())

            #the bias of hidden layer1
            b1 = tfv1.get_variable('b1', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            #the bias of hidden layer2
            b2 = tfv1.get_variable('b2', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            #the bias of hidden layer3
            b3 = tfv1.get_variable('b3', shape=(self.hidden_units), initializer=tf.zeros_initializer())
            #the bias of hidden layer3
            b4 = tfv1.get_variable('b4', shape=(self.hidden_units), initializer=tf.zeros_initializer())

            #the output of hidden layer1
            h1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
            #the output of hidden layer2
            h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
            #the output of hidden layer3
            h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)
            #the output of hidden layer3
            h4 = tf.nn.tanh(tf.matmul(h3, W4) + b4)

            #the output of output layer, that is the Q-value
            self.q = tf.matmul(h4, W5)

            self.parms = tfv1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='tnet')