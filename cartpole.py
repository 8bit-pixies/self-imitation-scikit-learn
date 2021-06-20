"""
We first have to pull out trajectories and select top-k
and then apply behavioral cloning.

This is using "self-imitation learning"

Sample output:

                                             Rollout Return                                         
     ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
199.0┤                                                                                      ▖▘▘▝▝▝▝│
178.0┤                                                                                ▗▗ ▝ ▘       │
     │                                                                               ▖  ▝          │
157.0┤                                                                          ▗▝ ▖▘              │
     │                                                                      ▖▖▝▝                   │
     │                                                                  ▗ ▖▖                       │
137.0┤                                                            ▖ ▖▗▗▝                           │
116.0┤                                                        ▗▝▝  ▘                               │
     │                                            ▖ ▗▗▗▝ ▘▘▘▘▝                                     │
 95.0┤                                     ▖▗▗▗▝ ▘ ▘                                               │
     │                             ▖▗▗▗ ▘▘▘                                                        │
 74.0┤▖▘▘▘▝▝▗▝ ▖▖▖▖▗▗▗ ▖▖▘▖▝▝▝▝ ▘▘▘                                                                │
     └┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬┘
      0                     20.0                    41                    62.0                    82

82: average return 198.99


"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import gym
import numpy as np
import random

import plotext as plt
import numpy as np


def rollout(env, policy):
    obs = env.reset()

    done = False
    total_reward = 0
    while not done:
        act = policy.get_action(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward
    return total_reward


def get_trajectories(
    env, policy=None, eps=0.05, data=[], num_trajectories=1000, max_cycle=1000, k=10
):
    data = data
    if len(data) > k:
        # keep 10 from the previous observation.
        reward_total = [np.sum(x["reward"]) for x in data]
        arg_reward = np.argsort(reward_total)[-k:]
        data = [x for idx, x in enumerate(data) if idx in arg_reward]
        num_trajectories = num_trajectories - k

    for _ in range(num_trajectories):
        obs_list = []
        action_list = []
        done_list = []
        reward_list = []
        next_obs_list = []
        # env = gym.make('CartPole-v0')
        obs = env.reset()
        done = False

        for _ in range(max_cycle):
            if policy is None or np.random.uniform() < eps:
                action = env.action_space.sample()
            else:
                action = policy.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)
            obs = next_obs
            if done:
                break
        data.append(
            {
                "obs": obs_list,
                "action": action_list,
                "done": done_list,
                "reward": reward_list,
                "next_obs": next_obs_list,
            }
        )
    return data


class BehaviorCloning(object):
    def __init__(self, policy=MLPClassifier(), k=10, gamma=0.999):
        self.policy = policy
        self.k = k
        self.alpha = 0.2

    def fit(self, traj):
        # filter the trajectories, and train only on top.
        reward_total = [np.sum(x["reward"]) for x in traj]
        arg_reward = np.argsort(reward_total)[-self.k :]

        # also take some trajectories at random
        arg_random = [x for x in np.arange(len(traj)) if x not in arg_reward]
        arg_random = random.sample(arg_random, self.k // 2)
        arg_random = []

        traj = [
            x for idx, x in enumerate(traj) if idx in arg_reward or idx in arg_random
        ]
        X = [x["obs"] for x in traj]
        Y = [x["action"] for x in traj]
        weight = [np.sum(x["reward"]) for x in traj]
        weight = StandardScaler().fit_transform(np.array(weight).reshape(-1, 1))
        weight = weight.flatten()
        weight += np.abs(np.min(weight))

        # weight = weight * 0 + 1
        weight = np.exp(weight)
        weight = weight.tolist()

        # resample X, Y according to weight
        X_resample = None
        Y_resample = None
        for x, y, w in zip(X, Y, weight):
            x_resample = np.vstack([x for _ in range(int(w))])
            y_resample = np.hstack([y for _ in range(int(w))])
            if X_resample is None:
                X_resample = x_resample
                Y_resample = y_resample
            else:
                X_resample = np.vstack([X_resample, x_resample])
                Y_resample = np.hstack([Y_resample, y_resample])

        # now shuffle
        X, Y = shuffle(X_resample, Y_resample, random_state=0)

        try:
            self.policy.partial_fit(
                x,
                y,
            )
        except:
            self.policy.fit(x, y)
        return self

    def get_action(self, obs):
        return self.policy.predict(obs.reshape(1, -1))[0]


iters = 1000
plt.clp()
plt.plotsize(100, 30)
plt.title("Streaming Data")
plt.colorless()
eval_perf = []
eps = 0.99

for i in range(iters):
    env = gym.make("CartPole-v0")
    bc = BehaviorCloning()
    if i == 0:
        data = get_trajectories(env, eps=eps ** i)
    else:
        data = get_trajectories(env, bc, data=[], eps=eps ** i)
    bc.fit(data)

    # try evaluating and training more
    rollout_return = []
    for _ in range(100):
        total_reward = rollout(env, bc)
        rollout_return.append(total_reward)

    rollout_return = np.mean(rollout_return)
    eval_perf.append(rollout_return)
    plt.clp()
    plt.cls()
    plt.plotsize(100, 30)
    plt.title("Rollout Return")
    plt.scatter(np.arange(len(eval_perf)), np.array(eval_perf), marker="small")
    plt.show()
    print(f"{i}: average return {rollout_return}")
