import numpy as np

def minibatch(obs, next_obs, action_probs, batch_size):
    random_indices = np.random.randint(0, obs.shape[0], size=batch_size)
    return obs[random_indices], next_obs[random_indices], action_probs[random_indices]
