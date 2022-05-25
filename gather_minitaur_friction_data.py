# Jeremiah Coholich 8/17/2020

import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import time

import pybullet as p
import time
import json

mu_lb = 0.001
mu_ub = 1.0 
num_mu = 100
num_steps_per_mu = 1000
data = {}
render = False
if True:

    sys.path.append('a2c_ppo_acktr')

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()

    args.det = not args.non_det

    env = make_vec_envs(
        'MinitaurBulletEnv-v0',
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False,
        render=render)


    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = \
                torch.load(os.path.join('trained_models/trained_minitaur_normal', 'MinitaurBulletEnv-v0' + ".pt"))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(1,
                                          actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)


    if render_func is not None:
        render_func('human')

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

mus = list(np.linspace(mu_lb, mu_ub, num_mu))
minitaur = env.venv.venv.envs[0].env.env.minitaur
for mu in mus:
    minitaur.SetFootFriction(mu)
    num_steps = 0
    mu_observations = torch.empty((num_steps_per_mu, 28))

    while num_steps < num_steps_per_mu: # I'm not interseted in recording the initial observation, since it is not influenced by dyanmics
        obs = env.reset()
        while True:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

            # Obser reward and next obs
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, _ = env.step(action)


            masks.fill_(0.0 if done else 1.0)

            mu_observations[num_steps] = obs
            num_steps += 1

            if done or num_steps >= num_steps_per_mu:
                break

        

            # if args.env_name.find('Bullet') > -1:
            #     if torsoId > -1:
            #         distance = 5
            #         yaw = 0
            #         humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            #         p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            # if render_func is not None:
            #     render_func('human')



            # time.sleep(1./240)

    data[mu] = mu_observations.tolist()
    print('finished mu of %0.3f' %mu)

with open('data/minitaur_friction_data.txt', 'w') as file:
    json.dump(data, file)


