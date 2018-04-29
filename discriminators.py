import tensorflow as tf
import numpy as np
from networks import MLP
from utils import sample_minibatch, sample_basis_minibatch

class AIRLDiscriminator:
    def __init__(
        self,
        name,
        ob_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.elu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        discount=0.99,
        learning_rate=1e-3
    ):
        self.last_loss = -np.log(0.5)
        self.noise_param = 0
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')
            self.next_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='next_obs')
            # reward network. assumes rewards are functions of state only
            self.reward_network = MLP('reward', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.rewards = self.reward_network.layers['out']
            # value network
            self.value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.next_value_network = MLP('value', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.next_obs, reuse=True)
            self.values = self.value_network.layers['out']
            self.next_values = self.next_value_network.layers['out']

            # estimate p(a | s, expert) as follows:
            self.expert_action_log_probs = self.rewards + discount*self.next_values - self.values

            self.policy_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='policy_action_log_probs')
            self.expert_log_probs = self.expert_action_log_probs - tf.reduce_logsumexp(tf.stack((self.expert_action_log_probs, self.policy_action_log_probs)), axis=0)

            # training
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels') # 1 if from expert else 0
            # self.expert_loss = -tf.log(self.expert_probs)
            # self.policy_loss = -tf.log(1-self.expert_probs)
            self.loss = -tf.reduce_mean(self.labels*self.expert_log_probs + (1-self.labels)*tf.log(1-tf.exp(self.expert_log_probs)+1e-8))
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def expert_log_prob(self, obs, next_obs, policy_action_log_probs, global_session):
        expert_log_probs = global_session.run(
            self.expert_log_probs,
            feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_log_probs: policy_action_log_probs}
        )
        return expert_log_probs

    def reward(self, obs, global_session):
        rewards = global_session.run(
            self.rewards,
            feed_dict={self.obs: obs}
        )
        return rewards

    def train(self,
        expert_obs, expert_next_obs, expert_actions,
        policy_obs, policy_next_obs, policy_actions,
        policy, global_session,
        n_iters=100, batch_size=32
    ):
        # add instance noise to the inputs if the signal for the generator is too weak
        if self.last_loss < -np.log(0.9):
            print('increasing noise param to weaken discriminator')
            self.noise_param += 0.01
        else:
            self.noise_param *= 0.9
        expert_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_obs, axis=0), size=expert_obs.shape)
        expert_next_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_next_obs, axis=0), size=expert_next_obs.shape)
        expert_actions += np.random.normal(loc=0, scale=self.noise_param*np.std(expert_actions, axis=0), size=expert_actions.shape)
        policy_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_obs, axis=0), size=policy_obs.shape)
        policy_next_obs += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_next_obs, axis=0), size=policy_next_obs.shape)
        policy_actions += np.random.normal(loc=0, scale=self.noise_param*np.std(policy_actions, axis=0), size=policy_actions.shape)

        policy_action_log_probs = global_session.run(
            policy.action_log_probs,
            feed_dict={policy.obs: policy_obs, policy.actions: policy_actions}
        )
        policy_action_log_probs = np.expand_dims(policy_action_log_probs, axis=1)

        expert_action_log_probs_under_policy = global_session.run(
            policy.action_log_probs,
            feed_dict={policy.obs: expert_obs, policy.actions: expert_actions}
        )
        expert_action_log_probs_under_policy = np.expand_dims(expert_action_log_probs_under_policy, axis=1)

        labels = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

        for iter_ in range(n_iters):
            mb_expert_obs, mb_expert_next_obs, mb_expert_action_log_probs_under_policy = sample_minibatch(expert_obs, expert_next_obs, expert_action_log_probs_under_policy, batch_size)
            mb_policy_obs, mb_policy_next_obs, mb_policy_action_log_probs = sample_minibatch(policy_obs, policy_next_obs, policy_action_log_probs, batch_size)
            mb_obs = np.concatenate((mb_expert_obs, mb_policy_obs))
            mb_next_obs = np.concatenate((mb_expert_next_obs, mb_policy_next_obs))
            mb_policy_action_log_probs = np.concatenate((mb_expert_action_log_probs_under_policy, mb_policy_action_log_probs))
            global_session.run(
                self.train_op,
                feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.labels: labels}
            )

        self.last_loss = global_session.run(
            self.loss,
            feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.labels: labels}
        )
        print('discrim_loss:', self.last_loss)

class SHAIRLDiscriminator:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        n_tasks,
        n_timesteps,
        basis_size,
        out_activation=None,
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.elu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        discount=0.99,
        learning_rate=1e-3
    ):
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.n_tasks = n_tasks
        self.n_timesteps = n_timesteps
        self.last_task_losses = np.tile(-np.log(0.5), n_tasks)
        self.noise_params = np.tile(0., n_tasks)
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')
            self.next_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='next_obs')
            self.tasks_timesteps = tf.placeholder(tf.int32, shape=[None, 2], name='tasks_timesteps')
            self.next_tasks_timesteps = tf.stack((self.tasks_timesteps[:, 0], self.tasks_timesteps[:, 1] + 1), axis=1)

            # reward basis network. shared accross tasks and timesteps
            self.basis_network = MLP('basis', ob_dim, basis_size, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.basis = self.basis_network.layers['out']
            self.next_basis_network = MLP('basis', ob_dim, basis_size, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.next_obs, reuse=True)
            self.next_basis = self.next_basis_network.layers['out']
            # ground truth
            # points = tf.constant([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=tf.float32)
            # self.basis = tf.pad(tf.norm(tf.reshape(tf.tile(self.obs, [1, 4]), (-1, 4, ob_dim)) - points, axis=2), [[0,0], [0,1]], constant_values=1.)
            # self.next_basis = tf.pad(tf.norm(tf.reshape(tf.tile(self.next_obs, [1, 4]), (-1, 4, ob_dim)) - points, axis=2), [[0,0], [0,1]], constant_values=1.)

            # w(task, timestep)
            self.all_reward_weights = tf.get_variable('reward_weights', [n_tasks, n_timesteps+1, basis_size], initializer=weight_init())
            self.all_value_weights = tf.get_variable('value_weights', [n_tasks, n_timesteps+1, basis_size], initializer=weight_init())

            # rewards
            self.reward_weights = tf.gather_nd(self.all_reward_weights, self.tasks_timesteps)
            self.value_weights = tf.gather_nd(self.all_value_weights, self.tasks_timesteps)
            self.next_value_weights = tf.gather_nd(self.all_value_weights, self.next_tasks_timesteps)
            self.rewards = tf.expand_dims(tf.reduce_sum(self.basis * self.reward_weights, axis=1), axis=1)
            self.values = tf.expand_dims(tf.reduce_sum(self.basis * self.value_weights, axis=1), axis=1)
            self.next_values = tf.expand_dims(tf.reduce_sum(self.next_basis * self.next_value_weights, axis=1), axis=1)

            # estimate p(a | s, expert) as follows:
            self.expert_action_log_probs = self.rewards + discount*self.next_values - self.values

            self.policy_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='policy_action_log_probs')
            self.expert_log_probs = self.expert_action_log_probs - tf.reduce_logsumexp(tf.stack((self.expert_action_log_probs, self.policy_action_log_probs)), axis=0)

            # training
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels') # 1 if from expert else 0
            # self.expert_loss = -tf.log(self.expert_probs)
            # self.policy_loss = -tf.log(1-self.expert_probs)
            self.loss = -tf.reduce_mean(self.labels*self.expert_log_probs + (1-self.labels)*tf.log(1-tf.exp(self.expert_log_probs)+1e-8))
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def expert_log_prob(self, obs, next_obs, policy_action_log_probs, task, global_session):
        tasks = np.expand_dims(np.tile(task, obs.shape[0]), axis=1)
        obs, next_obs, timesteps = obs[:, :self.ob_dim], next_obs[:, :self.ob_dim], obs[:, self.ob_dim:]
        tasks_timesteps = np.concatenate((tasks, timesteps), axis=1)
        expert_log_probs = global_session.run(
            self.expert_log_probs,
            feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_log_probs: policy_action_log_probs, self.tasks_timesteps: tasks_timesteps}
        )
        return expert_log_probs

    def reward(self, obs, task, global_session):
        tasks = np.expand_dims(np.tile(task, obs.shape[0]), axis=1)
        obs, timesteps = obs[:, :self.ob_dim], obs[:, self.ob_dim:]
        tasks_timesteps = np.concatenate((tasks, timesteps), axis=1)
        rewards = global_session.run(
            self.rewards,
            feed_dict={self.obs: obs, self.tasks_timesteps: tasks_timesteps}
        )
        return rewards

    def train(self,
        expert_obs, expert_next_obs, expert_actions,
        policy_obs, policy_next_obs, policy_actions,
        policies, global_session,
        n_iters=100, batch_size=32,
    ):
        # expert_obs, etc. = [n_tasks * np.array(size=(#_demos, n_timesteps, ob_dim+1))]
        # the extra ob_dim is time

        # add noise if the discriminator is too good to increase the signal for the generator
        self.noise_params[self.last_task_losses < -np.log(0.9)] += 0.01
        self.noise_params[self.last_task_losses >= -np.log(0.9)] *= 0.9
        # weird indexing here since last ob_dim is time
        for task in range(self.n_tasks):
            task_expert_obs, task_expert_next_obs, task_expert_actions, task_policy_obs, task_policy_next_obs, task_policy_actions, noise = expert_obs[task][:, :, :self.ob_dim], expert_next_obs[task][:, :, :self.ob_dim], expert_actions[task][:, :, :self.ob_dim], policy_obs[task][:, :, :self.ob_dim], policy_next_obs[task][:, :, :self.ob_dim], policy_actions[task][:, :, :self.ob_dim], self.noise_params[task]
            task_expert_obs += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_expert_obs, (-1, self.ob_dim)), axis=0), size=task_expert_obs.shape)
            task_expert_next_obs += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_expert_next_obs, (-1, self.ob_dim)), axis=0), size=task_expert_next_obs.shape)
            task_expert_actions += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_expert_actions, (-1, self.action_dim)), axis=0), size=task_expert_actions.shape)
            task_policy_obs += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_policy_obs, (-1, self.ob_dim)), axis=0), size=task_policy_obs.shape)
            task_policy_next_obs += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_policy_next_obs, (-1, self.ob_dim)), axis=0), size=task_policy_next_obs.shape)
            task_policy_actions += np.random.normal(loc=0, scale=noise*np.std(np.reshape(task_policy_actions, (-1, self.action_dim)), axis=0), size=task_policy_actions.shape)

        policy_action_log_probs = []
        for task in range(self.n_tasks):
            task_policy_obs, task_policy_actions, policy = np.reshape(policy_obs[task], (-1, self.ob_dim+1)), np.reshape(policy_actions[task], (-1, self.action_dim)), policies[task]
            policy_action_log_probs.append(
                np.reshape(
                    global_session.run(
                        policy.action_log_probs,
                        feed_dict={policy.obs: task_policy_obs, policy.actions: task_policy_actions}
                    ),
                    (-1, self.n_timesteps, 1)
                )
            )

        expert_action_log_probs_under_policy = []
        for task in range(self.n_tasks):
            task_expert_obs, task_expert_actions, policy = np.reshape(expert_obs[task], (-1, self.ob_dim+1)), np.reshape(expert_actions[task], (-1, self.action_dim)), policies[task]
            expert_action_log_probs_under_policy.append(
                np.reshape(
                    global_session.run(
                        policy.action_log_probs,
                        feed_dict={policy.obs: task_expert_obs, policy.actions: task_expert_actions}
                    ),
                    (-1, self.n_timesteps, 1)
                )
            )

        mb_labels = np.concatenate((np.ones((batch_size*self.n_tasks, 1)), np.zeros((batch_size*self.n_tasks, 1))))
        for iter_ in range(n_iters):
            mb_expert_obs, mb_expert_next_obs, mb_expert_action_log_probs_under_policy, mb_expert_tasks_timesteps = sample_basis_minibatch(expert_obs, expert_next_obs, expert_action_log_probs_under_policy, batch_size)
            mb_policy_obs, mb_policy_next_obs, mb_policy_action_log_probs, mb_policy_tasks_timesteps = sample_basis_minibatch(policy_obs, policy_next_obs, policy_action_log_probs, batch_size)
            mb_obs, mb_next_obs, mb_policy_action_log_probs, mb_tasks_timesteps = np.concatenate((mb_expert_obs, mb_policy_obs)), np.concatenate((mb_expert_next_obs, mb_policy_next_obs)), np.concatenate((mb_expert_action_log_probs_under_policy, mb_policy_action_log_probs)), np.concatenate((mb_expert_tasks_timesteps, mb_policy_tasks_timesteps))
            global_session.run(
                self.train_op,
                feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.tasks_timesteps: mb_tasks_timesteps, self.labels: mb_labels}
            )

        for task in range(self.n_tasks):
            mb_task_obs, mb_task_next_obs, mb_task_policy_action_log_probs, mb_task_tasks_timesteps, mb_task_labels = np.concatenate((mb_obs[task*batch_size:(task+1)*batch_size], mb_obs[(task+self.n_tasks)*batch_size:(task+self.n_tasks+1)*batch_size])), np.concatenate((mb_next_obs[task*batch_size:(task+1)*batch_size], mb_next_obs[(task+self.n_tasks)*batch_size:(task+self.n_tasks+1)*batch_size])), np.concatenate((mb_policy_action_log_probs[task*batch_size:(task+1)*batch_size], mb_policy_action_log_probs[(task+self.n_tasks)*batch_size:(task+self.n_tasks+1)*batch_size])), np.concatenate((mb_tasks_timesteps[task*batch_size:(task+1)*batch_size], mb_tasks_timesteps[(task+self.n_tasks)*batch_size:(task+self.n_tasks+1)*batch_size])), np.concatenate((mb_labels[task*batch_size:(task+1)*batch_size], mb_labels[(task+self.n_tasks)*batch_size:(task+self.n_tasks+1)*batch_size]))
            self.last_task_losses[task] = global_session.run(
                self.loss,
                feed_dict={self.obs: mb_task_obs, self.next_obs: mb_task_next_obs, self.policy_action_log_probs: mb_task_policy_action_log_probs, self.tasks_timesteps: mb_task_tasks_timesteps, self.labels: mb_task_labels}
            )
        print('discrim_loss:', np.mean(self.last_task_losses))
