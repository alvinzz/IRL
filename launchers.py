import gym
from algos import *
from rewards import *
from rollouts import *
import tensorflow as tf
import numpy as np
import pickle
from envs import *
register_custom_envs()
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def train_expert(
    n_iters, save_dir, name,
    env_name, make_reward_fn=make_ent_env_reward_fn, irl_model_algo=AIRL, irl_model_name=None,
    timesteps_per_rollout=10000, ep_max_len=500, demo_timesteps=1e5,
    rl_algo=RL, use_checkpoint=False,
):
    """
    Trains an expert policy.
    Args:
        n_iters: Number of training iterations.
        save_dir: Local path to directory to save model.
        name: Name of model to save.
        env_name: OpenAI Gym name. Register new envs in envs/__init__.py.
        make_reward_fn: Function that takes in a model as argument and returns
            a reward function for the policy.
            For example, make_irl_reward_fn (in rewards.py) uses (an entropy bonus)
                plus (the discriminator's log-probability of the trajectory
                coming from the expert) as the reward.
            The default, make_ent_env_reward_fn, uses (the environment reward)
                plus (an entropy bonus) as the reward. Should pass in None as
                the model argument.
        irl_model_algo: The class of model passed into make_reward_fn.
            Must be able to call
            model.discriminator.expert_log_prob(obs, next_obs, action_log_probs, task, model.sess).
        irl_model_name: The name of the model passed into make_reward_fn.
        timesteps_per_rollout: Number of timesteps to collect per rollout.
        ep_max_len: Maximum episode length.
        demo_timesteps: Number of timesteps to save as expert demonstration after
            the policy has finished training.
        rl_algo: The class of the reinforcement learning algorithm to use.
            Modify algos.py to change policy type, optimizer, architecture, etc.
        use_checkpoint: Boolean, whether or not to attempt to restore a previous
            expert policy from which to start training.
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    expert_model = rl_algo(name, env_fn, checkpoint=checkpoint)
    if irl_model_name:
        irl_graph = tf.Graph()
        with irl_graph.as_default():
            reward_fn_model = irl_model_algo(irl_model_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(save_dir, irl_model_name))
    else:
        reward_fn_model = None

    print('\nTraining expert...')
    expert_model.train(n_iters, make_reward_fn(reward_fn_model), timesteps_per_rollout, ep_max_len)

    print('\nCollecting expert trajectories, evaluating on original task...')
    expert_obs, expert_next_obs, expert_actions, _, _, _, _, _ = collect_and_process_rollouts(env_fn, expert_model.policy, make_env_reward_fn(None), expert_model.sess, demo_timesteps, ep_max_len)
    pickle.dump({'expert_obs': expert_obs, 'expert_next_obs': expert_next_obs, 'expert_actions': expert_actions}, open('{}/{}.pkl'.format(save_dir, name), 'wb'))

    expert_model.saver.save(expert_model.sess, '{}/{}_model'.format(save_dir, name))
    return expert_model

def train_irl(
    n_iters, save_dir, name, expert_name,
    env_name, make_reward_fn=make_irl_reward_fn,
    timesteps_per_rollout=10000, ep_max_len=500,
    irl_algo=AIRL, use_checkpoint=False,
):
    """
    Trains an IRL model.
    Args:
        n_iters: Number of training iterations.
        save_dir: Local path to directory to save model.
        name: Name of model to save.
        expert_name: Name of expert trajectories file (will attempt to load
            dictionary containing 'expert_obs, expert_next_obs, expert_actions'
            from (save_dir + '/' + expert_name + '_model.pkl')).
        env_name: OpenAI Gym env name. Register new envs in envs/__init__.py.
        make_reward_fn: Function that takes in a model as argument and returns a
            reward function for the policy. make_irl_reward_fn (in rewards.py)
            uses (an entropy bonus) plus (the discriminator's log-probability
            of the trajectory coming from the expert) as the reward.
        irl_algo: The class of the IRL algorithm to use. Passed into make_reward_fn.
            Modify algos.py to change algo hyperparameters.
        timesteps_per_rollout: Number of timesteps to collect per rollout.
        ep_max_len: Maximum episode length.
        use_checkpoint: Boolean, whether or not to attempt to restore a previous
            expert policy from which to start training.
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    data = pickle.load(open('{}/{}.pkl'.format(save_dir, expert_name), 'rb'))
    expert_obs, expert_next_obs, expert_actions = data['expert_obs'], data['expert_next_obs'], data['expert_actions']

    print('\nTraining IRL...')
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    irl_model = irl_algo(name, env_fn, expert_obs, expert_next_obs, expert_actions, checkpoint=checkpoint)
    irl_model.train(n_iters, make_reward_fn(irl_model), timesteps_per_rollout, ep_max_len)

    # evaluate and save model
    print('\nEvaluating policy on original task...')
    collect_and_process_rollouts(env_fn, irl_model.policy, make_env_reward_fn(None), irl_model.sess, 20*ep_max_len, ep_max_len)

    irl_model.saver.save(irl_model.sess, '{}/{}_model'.format(save_dir, name))
    return irl_model

def train_shairl(
    n_iters, save_dir, name, expert_names,
    env_names, make_reward_fns=make_shairl_reward_fns,
    timesteps_per_rollout=200, ep_len=40,
    irl_algo=SHAIRL, basis_size=3, use_checkpoint=False,
):
    """
    Trains an IRL model with linear per-task-per-timestep weights for a shared
        nonlinear basis.
    Args:
        n_iters: Number of training iterations.
        save_dir: Local path to directory to save model.
        name: Name of model to save.
        expert_names: List of names for expert trajectories files (will attempt
            to load dictionary containing 'expert_obs, expert_next_obs, expert_actions'
            from (save_dir + '/' + expert_name + '_model.pkl')).
        env_names: List of OpenAI Gym env names. Register new envs in
            envs/__init__.py. Should be same length as the expert_names list,
            with corresponding entries in corresponding positions.
        make_reward_fns: Function that takes in a model as argument and returns
            a list of reward functions, one for each of the (env, expert) pairs
            specified through the expert_names and env_names arguments.
        irl_algo: The class of the multitask IRL algorithm to use.
            Passed into make_reward_fns. Modify algos.py to change hyperparameters.
        timesteps_per_rollout: Number of timesteps to collect per rollout.
        ep_len: A fixed episode length.
        basis_size: The size of the shared nonlinear basis.
        use_checkpoint: Boolean, whether or not to attempt to restore a previous
            expert policy from which to start training.
    """
    tf.reset_default_graph()
    env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    expert_obs, expert_next_obs, expert_actions = [], [], []
    for expert_name in expert_names:
        data = pickle.load(open('{}/{}.pkl'.format(save_dir, expert_name), 'rb'))
        expert_obs.append(data['expert_obs'])
        expert_next_obs.append(data['expert_next_obs'])
        expert_actions.append(data['expert_actions'])

    print('\nTraining SHAIRL...')
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    irl_model = irl_algo(name, basis_size, env_fns, ep_len, expert_obs, expert_next_obs, expert_actions, checkpoint=checkpoint)
    irl_model.train(n_iters, make_reward_fns(irl_model), timesteps_per_rollout, ep_len)

    # evaluate and save model
    print('\nEvaluating policy on original tasks...')
    for task in range(len(env_fns)):
        print('Task', task)
        collect_and_process_rollouts(env_fns[task], irl_model.policies[task], make_env_reward_fn(None), irl_model.sess, 20*ep_len, ep_len)

    irl_model.saver.save(irl_model.sess, '{}/{}_model'.format(save_dir, name))
    return irl_model

def train_shairl_expert(
    n_iters, save_dir, name,
    env_names, basis_size, task,
    make_reward_fn=make_shairl_learned_reward_fn, irl_model_algo=SHAIRL, irl_model_name=None,
    timesteps_per_rollout=1000, ep_len=100, demo_timesteps=1e4,
    rl_algo=RL, use_checkpoint=False,
):
    """
    Trains a forward RL model for a task using the learned reward from a SHAIRL model.
    Args:
        n_iters: Number of training iterations.
        save_dir: Local path to directory to save model.
        name: Name of model to save.
        env_names: List of OpenAI Gym env names. Register new envs in
            envs/__init__.py. Used only to initialize the SHAIRL model, needs
            only to have the correct length and proper entry for the desired task.
        task: Index of the task to train the forward RL on.
        make_reward_fns: Function that takes in a model and task as arguments
            and returns a reward function.
        irl_algo: The class of the multitask IRL algorithm to use.
            Passed into make_reward_fns. Modify algos.py to change hyperparameters.
        timesteps_per_rollout: Number of timesteps to collect per rollout.
        ep_len: A fixed episode length.
        basis_size: The size of the shared nonlinear basis.
        demo_timesteps: Number of timesteps to save as expert demonstration after
            the policy has finished training.
        rl_algo: The class of the reinforcement learning algorithm to use.
            Modify algos.py to change policy type, optimizer, architecture, etc.
        use_checkpoint: Boolean, whether or not to attempt to restore a previous
            expert policy from which to start training.
    """
    tf.reset_default_graph()
    env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    env_fn = env_fns[task]
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    expert_model = rl_algo(name, env_fn, checkpoint=checkpoint)
    irl_graph = tf.Graph()
    with irl_graph.as_default():
        reward_fn_model = irl_model_algo(irl_model_name, basis_size, env_fns, ep_len, None, None, None, checkpoint='{}/{}_model'.format(save_dir, irl_model_name))

    print('\nTraining expert...')
    expert_model.train(n_iters, make_reward_fn(reward_fn_model, task), timesteps_per_rollout, ep_len)

    print('\nEvaluating on original task...')
    collect_and_process_rollouts(env_fn, expert_model.policy, make_env_reward_fn(None), expert_model.sess, demo_timesteps, ep_len)

    expert_model.saver.save(expert_model.sess, '{}/{}_model'.format(save_dir, name))
    return expert_model

def visualize_expert(env_name, expert_dir, expert_name, rl_algo=RL, ep_max_len=100, n_runs=1):
    """
    View a learned forward RL policy.
    Args:
        env_name: OpenAI Gym env name.
        expert_dir: Directory of expert model.
        expert_name: Name of expert model. Attempts to load
            expert_dir + '/' + expert_name + '_model' using TensorFlow checkpoint.
        rl_algo: Class of expert model.
        ep_max_len: Number of timesteps to show.
        n_runs: Number of runs to show (for stochastic polcies/envs).
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    expert_model = rl_algo(expert_name, env_fn, checkpoint='{}/{}_model'.format(expert_dir, expert_name))
    env = gym.make(env_name)
    tot_reward = 0
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            last_obs = obs
            env.render()
            time.sleep(0.02)
            action = expert_model.policy.act([obs], expert_model.sess)[0]
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            t += 1
        time.sleep(1)
    print('avg ep reward:', tot_reward / n_runs)

def visualize_irl_policy(env_name, irl_dir, irl_name, irl_algo=AIRL, ep_max_len=100, n_runs=1):
    """
    View a learned IRL policy.
    Args:
        env_name: OpenAI Gym env name.
        irl_dir: Directory of IRL model.
        irl_name: Name of IRL model. Attempts to load
            irl_dir + '/' + irl_name + '_model' using TensorFlow checkpoint.
        irl_algo: Class of IRL model.
        ep_max_len: Number of timesteps to show.
        n_runs: Number of runs to show (for stochastic polcies/envs).
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))
    env = gym.make(env_name)
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            env.render()
            time.sleep(0.02)
            action = irl_model.policy.act([obs], irl_model.sess)[0]
            obs, reward, done, info = env.step(action)
            t += 1
        time.sleep(1)

def visualize_shairl_policy(env_names, tasks, irl_dir, irl_name, irl_algo=SHAIRL, basis_size=3, ep_len=100, n_runs=1):
    """
    View a learned SHAIRL policy.
    Args:
        env_names: List of OpenAI Gym env names. Used only to initialize the
            SHAIRL model, needs only to have the correct length and proper
            entries for the desired tasks.
        tasks: List of indices of tasks to be shown.
        irl_dir: Directory of SHAIRL model.
        irl_name: Name of SHAIRL model. Attempts to load
            irl_dir + '/' + irl_name + '_model' using TensorFlow checkpoint.
        basis_size: Size of SHAIRL basis.
        ep_len: Episode length for the SHAIRL model.
        n_runs: Number of runs to show (for stochastic polcies/envs).
    """
    tf.reset_default_graph()
    env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    irl_model = irl_algo(irl_name, basis_size, env_fns, ep_len, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))
    for task in tasks:
        env = gym.make(env_names[task])
        for n in range(n_runs):
            obs = env.reset()
            done = False
            t = 0
            while not done and t < ep_len:
                env.render()
                time.sleep(0.02)
                action = irl_model.policies[task].act([obs], irl_model.sess)[0]
                obs, reward, done, info = env.step(action)
                t += 1
            time.sleep(1)

# works only for 2D envs
def visualize_irl_reward(env_name, irl_dir, irl_name, irl_algo=AIRL):
    """
    View the learned reward of an IRL model. Works only for 2D envs, and will
    need to be modified for different (x,y) bounds and observation space dims.
    Args:
        env_name: OpenAI Gym env name.
        irl_dir: Directory of IRL model.
        irl_name: Name of IRL model. Attempts to load
            irl_dir + '/' + irl_name + '_model' using TensorFlow checkpoint.
        irl_algo: Class of IRL model.
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))

    rewards = np.zeros((20, 20))
    for i, x in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
        for j, y in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
            rewards[i, j] = irl_model.discriminator.reward(np.array([[x, y, 0, 0]]), irl_model.sess)

    print('scale:', np.min(rewards), '(black) to', np.max(rewards), '(white)')
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    plt.imshow(rewards.T, cmap='gray', origin='lower')
    plt.show()

def visualize_shairl_reward(env_names, tasks, irl_dir, irl_name, irl_algo=SHAIRL, basis_size=3, ep_len=100, frame_skip=1):
    """
    View the learned reward of SHAIRL. Works only for 2D envs, and will
    need to be modified for different (x,y) bounds and observation space dims.
    Args:
        env_names: List of OpenAI Gym env names. Used only to initialize the
            SHAIRL model, needs only to have the correct length and proper
            entries for the desired tasks.
        tasks: List of indices of tasks to be shown.
        irl_dir: Directory of SHAIRL model.
        irl_name: Name of SHAIRL model. Attempts to load
            irl_dir + '/' + irl_name + '_model' using TensorFlow checkpoint.
        basis_size: Size of SHAIRL basis.
        ep_len: Episode length for the SHAIRL model.
        frame_skip: Speeds up animation by showing only some timestep rewards
            (one in every frame_skip).
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    irl_model = irl_algo(irl_name, basis_size, env_fns, ep_len, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))
    print('Showing 1 in every {} timesteps (out of {})'.format(frame_skip, ep_len))
    for task in tasks:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        def animate():
            for timestep in range(0, ep_len, frame_skip):
                rewards = np.zeros((20, 20))
                for i, x in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
                    for j, y in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
                        rewards[i, j] = irl_model.discriminator.reward(np.array([[x, y, 0, timestep]]), task, irl_model.sess)

                print('time:', timestep, 'scale:', np.min(rewards), '(black) to', np.max(rewards), '(white)')
                rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
                im = plt.imshow(rewards.T, cmap='gray', origin='lower')
                fig.canvas.draw()
        win = fig.canvas.manager.window
        fig.canvas.manager.window.after(0, animate)
        plt.show()

def visualize_shairl_basis(env_names, irl_dir, irl_name, irl_algo=SHAIRL, basis_size=3, ep_len=100):
    """
    View the learned basis of SHAIRL. Works only for 2D envs, and will
    need to be modified for different (x,y) bounds and observation space dims.
    Args:
        env_names: List of OpenAI Gym env names. Used only to initialize the
            SHAIRL model, needs only to have the correct length and proper
            entries for the desired tasks.
        irl_dir: Directory of SHAIRL model.
        irl_name: Name of SHAIRL model. Attempts to load
            irl_dir + '/' + irl_name + '_model' using TensorFlow checkpoint.
        basis_size: Size of SHAIRL basis.
        ep_len: Episode length for the SHAIRL model.
    """
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    irl_model = irl_algo(irl_name, basis_size, env_fns, ep_len, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))

    basis = np.zeros((20, 20, basis_size))
    for i, x in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
        for j, y in zip(np.arange(20), np.linspace(-1.5, 1.5, 20)):
            basis[i, j] = irl_model.sess.run(
                irl_model.discriminator.basis,
                feed_dict={irl_model.discriminator.obs: np.array([[x, y, 0]])}
            )

    for i in range(basis_size):
        print('scale:', np.min(basis[:, :, i]), '(black) to', np.max(basis[:, :, i]), '(white)')
        basis[:, :, i] = (basis[:, :, i] - np.min(basis[:, :, i])) / (np.max(basis[:, :, i]) - np.min(basis[:, :, i]))
        plt.imshow(basis[:, :, i].T, cmap='gray', origin='lower')
        plt.show()

if __name__ == '__main__':
    # create list of expert and environment names for SHAIRL
    expert_names = []
    env_names = []
    for i in range(0,1):
        for j in range(1,2):
            expert_names.append('expert-{}{}'.format(i, j))
            env_names.append('PointMass-v{}{}'.format(i, j))

    # RUN UTILS.PY FIRST TO GENERATE EXPERT DEMOS!

    # train expert policies
    # for i in range(0,1):
    #     for j in range(1,2):
    #         print('Training', i, j)
    #         train_expert(n_iters=1000, save_dir='data/pointmass', name='expert-{}{}'.format(i, j), env_name='PointMass-v{}{}'.format(i, j), use_checkpoint=False, timesteps_per_rollout=200, ep_max_len=40, demo_timesteps=200)
    #         visualize_expert(env_name='PointMass-v{}{}'.format(i, j), expert_dir='data/pointmass', expert_name='expert-{}{}'.format(i, j))

    # train AIRL
    # train_irl(
    #     n_iters=100, save_dir='data/pointmass', name='airl_01_toy', expert_name='expert-01',
    #     env_name='PointMass-v01', make_reward_fn=make_irl_reward_fn,
    #     timesteps_per_rollout=200, ep_max_len=40,
    #     irl_algo=AIRL, use_checkpoint=True,
    # )
    # visualize_irl_reward(env_name='PointMass-v01', irl_dir='data/pointmass', irl_name='airl_01_toy_orig', irl_algo=AIRL)
    # visualize_irl_policy(ep_max_len=40, env_name='PointMass-v01', irl_dir='data/pointmass', irl_name='airl_01_toy_orig', irl_algo=AIRL)

    # train SHAIRL
    train_shairl(basis_size=1, ep_len=40, n_iters=100, save_dir='data/pointmass', name='shairl_01_toy', expert_names=expert_names, env_names=env_names, use_checkpoint=True)
    visualize_shairl_basis(basis_size=1, ep_len=40, env_names=env_names, irl_dir='data/pointmass', irl_name='shairl_01_toy')
    # visualize_shairl_reward(basis_size=1, ep_len=40, env_names=env_names, tasks=[0], irl_dir='data/pointmass', irl_name='shairl_01_toy', frame_skip=1)
    visualize_shairl_policy(basis_size=1, ep_len=40, env_names=env_names, tasks=[0], irl_dir='data/pointmass', irl_name='shairl_01_toy', n_runs=1)
