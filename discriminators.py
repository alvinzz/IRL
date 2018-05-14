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
        learning_rate=1e-4
    ):
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

        loss = global_session.run(
            self.loss,
            feed_dict={self.obs: mb_obs, self.next_obs: mb_next_obs, self.policy_action_log_probs: mb_policy_action_log_probs, self.labels: labels}
        )
        print('discrim_loss:', loss)

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
        learning_rate=1e-4,
    ):
        self.ob_dim = ob_dim
        self.action_dim = action_dim
        self.n_tasks = n_tasks
        self.n_timesteps = n_timesteps
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
            # # ground truth
            # toy1
            # points = tf.constant([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=tf.float32)
            # self.basis = tf.pad(tf.norm(tf.reshape(tf.tile(self.obs, [1, 4]), (-1, 4, ob_dim)) - points, axis=2), [[0,0], [0,1]], constant_values=1.)
            # self.next_basis = tf.pad(tf.norm(tf.reshape(tf.tile(self.next_obs, [1, 4]), (-1, 4, ob_dim)) - points, axis=2), [[0,0], [0,1]], constant_values=1.)

            # w(task, timestep)
            self.all_reward_weights = tf.get_variable('reward_weights', [n_tasks, n_timesteps+1, basis_size], initializer=weight_init())
            self.all_value_weights = tf.get_variable('value_weights', [n_tasks, n_timesteps+1, basis_size], initializer=weight_init())
            # toy2
            # self.all_reward_weights = tf.constant(
            #     np.stack((
            #         np.concatenate((np.tile([-1,0,-3], [50,1]), np.tile([-1,0,-3], [51,1]))),
            #         np.concatenate((np.tile([-1,0,-3], [50,1]), np.tile([0,-1,-3], [51,1]))),
            #         np.concatenate((np.tile([0,-1,-3], [50,1]), np.tile([-1,0,-3], [51,1]))),
            #         np.concatenate((np.tile([0,-1,-3], [50,1]), np.tile([0,-1,-3], [51,1]))),
            #     )),
            # dtype=tf.float32)
            # self.all_value_weights = 12*self.all_reward_weights

            # rewards and values
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
            # expert_loss = -log(expert_probs), policy_loss = -log(1-expert_probs)
            self.loss = -tf.reduce_mean(self.labels*self.expert_log_probs + (1-self.labels)*tf.log(1-tf.exp(self.expert_log_probs)+1e-8))
            self.w_optimizer = tf.train.AdamOptimizer(learning_rate=100*learning_rate)
            self.w_grads = self.w_optimizer.compute_gradients(self.loss, [self.all_reward_weights, self.all_value_weights])
            self.w_train_op = self.w_optimizer.apply_gradients(self.w_grads)
            self.f_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.f_grads = self.f_optimizer.compute_gradients(self.loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope()+'/basis'))
            self.f_train_op = self.f_optimizer.apply_gradients(self.f_grads)

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
        n_iters=10, batch_size=32,
    ):
        # expert_obs, etc. = [n_tasks * np.array(size=(#_demos * n_timesteps, ob_dim+1))]
        # the extra ob_dim is time
        policy_action_log_probs = []
        for task in range(self.n_tasks):
            task_policy_obs, task_policy_actions, policy = policy_obs[task], policy_actions[task], policies[task]
            policy_action_log_probs.append(
                global_session.run(
                    policy.action_log_probs,
                    feed_dict={policy.obs: task_policy_obs, policy.actions: task_policy_actions}
                ),
            )
        policy_action_log_probs = np.expand_dims(policy_action_log_probs, axis=2)

        expert_action_log_probs_under_policy = []
        for task in range(self.n_tasks):
            task_expert_obs, task_expert_actions, policy = expert_obs[task], expert_actions[task], policies[task]
            expert_action_log_probs_under_policy.append(
                global_session.run(
                    policy.action_log_probs,
                    feed_dict={policy.obs: task_expert_obs, policy.actions: task_expert_actions}
                ),
            )
        expert_action_log_probs_under_policy = np.expand_dims(expert_action_log_probs_under_policy, axis=2)

        # coord descent
        labels = np.concatenate((np.ones((expert_action_log_probs_under_policy.size, 1)), np.zeros((policy_action_log_probs.size, 1))))
        all_expert_obs, all_policy_obs, all_expert_next_obs, all_policy_next_obs, all_expert_action_log_probs_under_policy, all_policy_action_log_probs = np.reshape(expert_obs, (-1, self.ob_dim+1)), np.reshape(policy_obs, (-1, self.ob_dim+1)), np.reshape(expert_next_obs, (-1, self.ob_dim+1)), np.reshape(policy_next_obs, (-1, self.ob_dim+1)), np.reshape(expert_action_log_probs_under_policy, (-1, 1)), np.reshape(policy_action_log_probs, (-1, 1))
        all_tasks = np.expand_dims(np.concatenate((np.array([np.tile(task, expert_obs[task].shape[0]) for task in range(self.n_tasks)]).flatten(), np.array([np.tile(task, policy_obs[task].shape[0]) for task in range(self.n_tasks)]).flatten())), axis=1)
        task_labels = [np.concatenate((np.ones((expert_action_log_probs_under_policy[task].size, 1)), np.zeros((policy_action_log_probs[task].size, 1)))) for task in range(self.n_tasks)]
        for iter_ in range(n_iters):
            # optimize w
            last_w_loss = [1e9 for task in range(self.n_tasks)]
            cur_w_loss = [1e8 for task in range(self.n_tasks)]
            for task in range(self.n_tasks):
                obs, next_obs, action_log_probs = np.concatenate((expert_obs[task], policy_obs[task])), np.concatenate((expert_next_obs[task], policy_next_obs[task])), np.concatenate((expert_action_log_probs_under_policy[task], policy_action_log_probs[task]))
                obs, timesteps, next_obs = obs[:, :self.ob_dim], obs[:, self.ob_dim:], next_obs[:, :self.ob_dim]
                tasks_timesteps = np.concatenate((np.tile(task, timesteps.shape), timesteps), axis=1)
                i = 0
                while cur_w_loss[task] < last_w_loss[task] and i < 100:
                    loss, _ = global_session.run(
                        [self.loss, self.w_train_op],
                        feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_log_probs: action_log_probs, self.tasks_timesteps: tasks_timesteps, self.labels: task_labels[task]}
                    )
                    cur_w_loss[task], last_w_loss[task] = loss, cur_w_loss[task]
                    i += 1
                print('Task', task, 'w_loss:', loss)

            # optimize f
            last_f_loss = 1e9
            cur_f_loss = 1e8
            obs, next_obs, action_log_probs = np.concatenate((all_expert_obs, all_policy_obs)), np.concatenate((all_expert_next_obs, all_policy_next_obs)), np.concatenate((all_expert_action_log_probs_under_policy, all_policy_action_log_probs))
            obs, timesteps, next_obs = obs[:, :self.ob_dim], obs[:, self.ob_dim:], next_obs[:, :self.ob_dim]
            tasks_timesteps = np.concatenate((all_tasks, timesteps), axis=1)
            i = 0
            while cur_f_loss < last_f_loss and i < 100:
                loss, _ = global_session.run(
                    [self.loss, self.f_train_op],
                    feed_dict={self.obs: obs, self.next_obs: next_obs, self.policy_action_log_probs: action_log_probs, self.tasks_timesteps: tasks_timesteps, self.labels: labels}
                )
                cur_f_loss, last_f_loss = loss, cur_f_loss
                i += 1
            print('f_loss:', loss)

class StandardDiscriminator:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.elu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        discount=0.99,
        learning_rate=1e-4
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')
            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.input = tf.concat((self.obs, self.actions), axis=1)
            # expert probability network
            self.prob_network = MLP('prob', ob_dim+action_dim, 2, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.input)
            self.unscaled_probs = self.prob_network.layers['out']
            self.expert_log_probs = tf.log(tf.nn.softmax(self.unscaled_probs)[:, 1:])

            # training
            self.labels = tf.placeholder(tf.int32, shape=[None], name='labels') # 1 if from expert else 0
            # self.expert_loss = -tf.log(self.expert_probs)
            # self.policy_loss = -tf.log(1-self.expert_probs)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unscaled_probs, labels=self.labels))
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def expert_log_prob(self, obs, actions, global_session):
        expert_log_probs = global_session.run(
            self.expert_log_probs,
            feed_dict={self.obs: obs, self.actions: actions}
        )
        return expert_log_probs

    def train(self,
        expert_obs, expert_actions,
        policy_obs, policy_actions,
        policy, global_session,
        n_iters=100, batch_size=32
    ):
        labels = np.ones(100*batch_size)
        mb_obs, mb_actions, _ = sample_minibatch(expert_obs, expert_actions, np.zeros_like(expert_actions), 100*batch_size)
        expert_loss = global_session.run(
            self.loss,
            feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.labels: labels}
        )
        labels = np.zeros(100*batch_size)
        mb_obs, mb_actions, _ = sample_minibatch(policy_obs, policy_actions, np.zeros_like(policy_actions), 100*batch_size)
        policy_loss = global_session.run(
            self.loss,
            feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.labels: labels}
        )
        for iter_ in range(n_iters):
            r = np.random.rand()
            if r >= 0.5 and expert_loss >= 0.01:
                labels = np.ones(batch_size)
                mb_obs, mb_actions, _ = sample_minibatch(expert_obs, expert_actions, np.zeros_like(expert_actions), batch_size)
                expert_loss, _ = global_session.run(
                    [self.loss, self.train_op],
                    feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.labels: labels}
                )
            elif r < 0.5 and policy_loss >= 0.01:
                labels = np.zeros(batch_size)
                mb_obs, mb_actions, _ = sample_minibatch(policy_obs, policy_actions, np.zeros_like(policy_actions), batch_size)
                policy_loss, _ = global_session.run(
                    [self.loss, self.train_op],
                    feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.labels: labels}
                )

        print('discrim loss on expert:', expert_loss)
        print('discrim loss on policy:', policy_loss)
