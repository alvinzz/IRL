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
    timesteps_per_rollout=1000, ep_len=100,
    irl_algo=SHAIRL, basis_size=3, use_checkpoint=False,
):
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

def train_intention(
    n_iters, n_intentions, save_dir, name, expert_name,
    env_name, make_reward_fn=make_intention_reward_fn,
    timesteps_per_rollout=10000, ep_max_len=1000,
    irl_algo=IntentionGAN, use_checkpoint=False,
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    data = pickle.load(open('{}/{}.pkl'.format(save_dir, expert_name), 'rb'))
    expert_obs, expert_actions = data['expert_obs'], data['expert_actions']

    print('\nTraining IRL...')
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    irl_model = irl_algo(name, env_fn, n_intentions, expert_obs, expert_actions, checkpoint=checkpoint)
    irl_model.train(n_iters, make_reward_fn(irl_model), timesteps_per_rollout, ep_max_len)

    # save model
    irl_model.saver.save(irl_model.sess, '{}/{}_model'.format(save_dir, name))
    return irl_model

def train_choice_intention(
    n_iters, n_intentions, save_dir, name, expert_name,
    env_name, make_chooser_reward_fn=make_intention_chooser_reward_fn, make_reward_fn=make_intention_reward_fn,
    timesteps_per_rollout=10000, ep_max_len=2500,
    irl_algo=IntentionChoiceGAN, use_checkpoint=False,
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    data = pickle.load(open('{}/{}.pkl'.format(save_dir, expert_name), 'rb'))
    expert_obs, expert_next_obs, expert_actions = data['expert_obs'], data['expert_next_obs'], data['expert_actions']

    print('\nTraining IRL...')
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    irl_model = irl_algo(name, env_fn, n_intentions, expert_obs, expert_next_obs, expert_actions, checkpoint=checkpoint)
    irl_model.train(n_iters, make_chooser_reward_fn(irl_model), make_reward_fn(irl_model), timesteps_per_rollout, ep_max_len)

    # save model
    irl_model.saver.save(irl_model.sess, '{}/{}_model'.format(save_dir, name))
    return irl_model

def visualize_expert(env_name, expert_dir, expert_name, rl_algo=RL, ep_max_len=100, n_runs=1):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    expert_model = rl_algo(expert_name, env_fn, checkpoint='{}/{}_model'.format(expert_dir, expert_name))

    env = gym.make(env_name)
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
            t += 1
        time.sleep(1)

def visualize_irl_policy(env_name, irl_dir, irl_name, irl_algo=AIRL, ep_max_len=100, n_runs=1):
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

def visualize_intention_policy(env_name, n_intentions, save_dir, irl_name, intentions, irl_algo=IntentionGAN, ep_max_len=100, n_runs=3):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, n_intentions, None, None, checkpoint='{}/{}_model'.format(save_dir, irl_name))
    env = gym.make(env_name)
    for intention in intentions:
        one_hot_intention = np.zeros(n_intentions)
        one_hot_intention[intention] = 1
        for n in range(n_runs):
            obs = env.reset()
            done = False
            t = 0
            while not done and t < ep_max_len:
                env.render()
                time.sleep(0.02)
                action = irl_model.policy.act([np.concatenate((obs, one_hot_intention))], irl_model.sess)[0]
                obs, reward, done, info = env.step(action)
                t += 1
            time.sleep(1)

# works only for 2D envs
def visualize_irl_reward(env_name, irl_dir, irl_name, irl_algo=AIRL):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, None, None, None, checkpoint='{}/{}_model'.format(irl_dir, irl_name))

    rewards = np.zeros((100, 100))
    for i, x in zip(np.arange(100), np.linspace(-1.5, 1.5, 20)):
        for j, y in zip(np.arange(100), np.linspace(-1.5, 1.5, 20)):
            rewards[i, j] = irl_model.discriminator.reward(np.array([[x, y, 0]]), irl_model.sess)

    print('scale:', np.min(rewards), '(black) to', np.max(rewards), '(white)')
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    plt.imshow(rewards, cmap='gray', origin='lower')
    plt.show()

def visualize_shairl_reward(env_names, tasks, irl_dir, irl_name, irl_algo=SHAIRL, basis_size=3, ep_len=100, frame_skip=1):
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

def test_turtle(env_name, n_intentions, save_dir, irl_name, irl_algo=IntentionChoiceGAN, n_runs=4):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    irl_model = irl_algo(irl_name, env_fn, n_intentions, None, None, None, checkpoint='{}/{}_model'.format(save_dir, irl_name))
    env = gym.make(env_name)

    # env.box = np.array([0.7, 0.4])
    # env.target = np.array([0.4, 0.7])
    # env.target = np.array([0.7, 0.1]) #bad
    # env.box = np.array([0.4, 0.6])
    # env.state = np.array([0,0,0])
    # env.box_angle = np.pi/5 #np.pi/5
    # env.start_box_angle = np.arctan2(env.box[1] - env.state[1], env.box[0] - env.state[0])
    # rewards = np.zeros((20, 20))
    # for i, x in zip(np.arange(20), np.linspace(0, 1, 20)):
    #     for j, y in zip(np.arange(20), np.linspace(0, 1, 20)):
    #         env.state = np.array([x, y, 0])
    #         env.box = np.array([x-0.2, y])
    #         # rewards[i, j] = irl_model.discriminator.reward(np.array([env._get_obs()]), irl_model.sess)
    #         rewards[i, j] = irl_model.intention_policy.test_act([env._get_obs()], irl_model.sess)[0]
    #
    # print('scale:', np.min(rewards), '(black) to', np.max(rewards), '(white)')
    # rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    # im = plt.imshow(rewards.T, cmap='gray', origin='lower')
    # plt.show()

    data = []
    for iter_ in range(n_runs):
        env.box = np.array([0.7, 0.4]) #good
        env.target = np.array([0.4, 0.7])
        # env.target = np.array([0.7, 0.1]) #bad
        # env.box = np.array([0.4, 0.6])
        env.state = np.array([0.5,0.5,0])
        env.box_angle = -np.pi/5 #np.pi/5
        env.start_box_angle = np.arctan2(env.box[1] - env.state[1], env.box[0] - env.state[0])
        # ob = env.reset()
        # start = [env.box, env.target, env.state, env.box_angle]
        # print('starting new trial')
        ob = env._get_obs()
        t = 0
        done = False
        print(iter_)
        while t < 50:
            env.render()
            if t%10 == 0:
                im = env.viewer.get_array()
                plt.imsave('../Pictures/good{}_{}.png'.format(t,iter_), im)
            # # sampled intention
            # intention_probs = irl_model.intention_policy.test_probs([ob], irl_model.sess)[0]
            # action = np.zeros(2)
            # for intention in range(n_intentions):
            #     one_hot_intention = np.zeros(n_intentions)
            #     one_hot_intention[intention] = 1
            #     action += intention_probs[intention] \
            #         * irl_model.policy.test_act([np.concatenate((ob, one_hot_intention))], irl_model.sess)[0]

            # deterministic intention
            # intention = irl_model.intention_policy.test_act([ob], irl_model.sess)[0]
            # # data.append([t, intention])
            # data.append([np.linalg.norm(env.state[:2] - env.box), intention])
            intention = iter_
            one_hot_intention = np.zeros(n_intentions)
            one_hot_intention[intention] = 1
            action = irl_model.policy.test_act([np.concatenate((ob, one_hot_intention))], irl_model.sess)[0]

            ob, reward, done, info = env.step(action)
            t += 1
            # if np.linalg.norm(env.box - env.target) < 0.03:
            #     break
            # if np.linalg.norm(env.state[:2] - env.target) < 0.08:
            #     im = env.viewer.get_array()
            #     plt.imsave('../Pictures/good{}.png'.format(t), im)
            #     print(start)
            #     break
        im = env.viewer.get_array()
        plt.imsave('../Pictures/good{}_{}.png'.format(t,iter_), im)
        # time.sleep(1)
    # data = np.array(data)
    #
    # # intentions = data[:, 1]
    # # unique, counts = np.unique(intentions, return_counts=True)
    # # plt.title('Intention Usage')
    # # plt.xlabel('Intention')
    # # plt.ylabel('Usage')
    # # plt.bar(unique, counts/np.sum(counts))
    # # plt.xticks(unique, unique)
    # # plt.show()
    #
    # counts = np.zeros((5, 4))
    # min_, max_ = np.min(data[:, 0]), np.max(data[:, 0])
    # range_ = max_ - min_
    # for row in data:
    #     bin_ = 5*(row[0]-min_) // range_
    #     if bin_ == 5:
    #         bin_ = 4
    #     counts[int(bin_)][int(row[1])] += 1
    # for i in range(5):
    #     counts[i] /= np.sum(counts[i])
    # plt.title('Intention Mixture as Function of Turtlebot Distance to Box')
    # plt.xlabel('Distance')
    # plt.ylabel('Mixture')
    # labels = ['Intention ' + str(i) for i in range(4)]
    # plt.stackplot(np.arange(5), counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3], labels=labels)
    # plt.legend(loc=2)
    # plt.xticks(np.arange(5), ['[{0:.2f}, {1:.2f})'.format(range_/5*i+min_, range_/5*(i+1)+min_) for i in range(5)])
    # plt.show()

if __name__ == '__main__':
    # for _ in range(10000):
    #     train_choice_intention(
    #         n_iters=10, n_intentions=4, save_dir='data/turtle', name='intention_choice2', expert_name='intention_expert2',
    #         env_name='Turtle-v0', use_checkpoint=True,
    #     )
    # 2: [64, 64, 64], [64], [64, 64, 64], [64, 64]
    # 5: [64, 64, 64],  [64, 64, 64],  [64, 64, 64], [64]
    test_turtle(env_name='Turtle-v0', n_intentions=4, save_dir='data/turtle', irl_name='intention_choice5')

    # for _ in range(20000):
        # train_intention(n_iters=100, n_intentions=4, save_dir='data/turtle', name='intention2', expert_name='intention_expert', env_name='Turtle-v0', use_checkpoint=True)
    # visualize_intention_policy(env_name='Turtle-v0', n_intentions=4, save_dir='data/turtle', irl_name='intention2', intentions=[0,1,2,3], n_runs=1, ep_max_len=1000)
    # test_turtle(env_name='Turtle-v0', n_intentions=4, save_dir='data/turtle', irl_name='intention2', intentions=[2,0,1,1], n_runs=3)

    # expert_names = []
    # env_names = []
    # for i in range(2):
    #     for j in range(2):
    #         expert_names.append('expert-{}{}'.format(i, j))
    #         env_names.append('PointMass-v{}{}'.format(i, j))

    #TODO: simple example, coord descent, replay buffer, reward only

    # for i in range(4):
    #     for j in range(4):
    #         print('Training', i, j)
    #         train_expert(n_iters=200, save_dir='data/pointmass', name='expert-{}{}'.format(i, j), env_name='PointMass-v{}{}'.format(i, j), use_checkpoint=False, timesteps_per_rollout=1000, ep_max_len=250, demo_timesteps=1e4)
    #         visualize_expert(env_name='PointMass-v{}{}'.format(i, j), expert_dir='data/pointmass', expert_name='expert-{}{}'.format(i, j))

    # train_shairl(n_iters=1, save_dir='data/pointmass', name='shairl_22', expert_names=expert_names, env_names=env_names, use_checkpoint=False)
    # for _ in range(20000):
        # train_shairl(n_iters=1, save_dir='data/pointmass', name='shairl_22_toy', expert_names=expert_names, env_names=env_names, use_checkpoint=True)
    # visualize_shairl_basis(env_names=env_names, irl_dir='data/pointmass', irl_name='shairl_22')
    # visualize_shairl_reward(env_names=env_names, tasks=[0,1,2,3], irl_dir='data/pointmass', irl_name='shairl_22', frame_skip=10)
    # visualize_shairl_policy(env_names=env_names, tasks=[0,1,2,3], irl_dir='data/pointmass', irl_name='shairl_22', n_runs=3)

    # for task in np.random.choice(np.arange(16), size=4, replace=False):
    # for task in [5, 12, 15, 7]:
        # print('Task:', task)
        # train_shairl_expert(n_iters=1000, save_dir='data/pointmass', name='44_{}_learned_expert'.format(task), env_names=env_names, basis_size=3, task=task, use_checkpoint=False, irl_model_name='shairl_44')
        # visualize_expert(env_names[task], 'data/pointmass', '44_{}_learned_expert'.format(task))
    # tf.reset_default_graph()
    # env_fns = [lambda: gym.make(env_name) for env_name in env_names]
    # irl_model = SHAIRL('shairl_toy', 5, env_fns, 100, None, None, None, checkpoint='data/pointmass/shairl_toy_model')
    # print(irl_model.sess.run(irl_model.discriminator.all_reward_weights))
    # print(irl_model.sess.run(irl_model.discriminator.all_value_weights))
