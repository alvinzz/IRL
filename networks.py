import tensorflow as tf

class MLP:
    def __init__(
        self,
        name,
        in_dim,
        out_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        in_layer=None,
        reuse=False
    ):
        self.params = {}
        self.layers = {}
        with tf.variable_scope(name, reuse=reuse):
            if in_layer is not None:
                self.layers['in'] = in_layer
            else:
                self.layers['in'] = tf.placeholder(tf.float32, shape=[None, in_dim], name='in')

            if hidden_dims:
                self.params['W0'] = tf.get_variable('W0', [in_dim, hidden_dims[0]], initializer=weight_init())
                self.params['b0'] = tf.get_variable('b0', [hidden_dims[0]], initializer=bias_init())
                self.layers['h0'] = hidden_activation(tf.matmul(self.layers['in'], self.params['W0']) + self.params['b0'])

                for i in range(1, len(hidden_dims)):
                    self.params['W{}'.format(i)] = tf.get_variable('W{}'.format(i), [hidden_dims[i-1], hidden_dims[i]], initializer=weight_init())
                    self.params['b{}'.format(i)] = tf.get_variable('b{}'.format(i), [hidden_dims[i]], initializer=bias_init())
                    self.layers['h{}'.format(i)] = hidden_activation(tf.matmul(self.layers['h{}'.format(i-1)], self.params['W{}'.format(i)]) + self.params['b{}'.format(i)])

                i = len(hidden_dims)
                self.params['W{}'.format(i)] = tf.get_variable('W{}'.format(i), [hidden_dims[-1], out_dim], initializer=weight_init())
                self.params['b{}'.format(i)] = tf.get_variable('b{}'.format(i), [out_dim], initializer=bias_init())
                if out_activation is None:
                    self.layers['out'] = tf.matmul(self.layers['h{}'.format(i-1)], self.params['W{}'.format(i)]) + self.params['b{}'.format(i)]
                else:
                    self.layers['out'] = out_activation(tf.matmul(self.layers['h{}'.format(i-1)], self.params['W{}'.format(i)]) + self.params['b{}'.format(i)])
            else:
                self.params['W0'] = tf.get_variable('W0', [in_dim, out_dim], initializer=weight_init())
                self.params['b0'] = tf.get_variable('b0', [out_dim], initializer=bias_init())
                self.layers['out'] = hidden_activation(tf.matmul(self.layers['in'], self.params['W0']) + self.params['b0'])
