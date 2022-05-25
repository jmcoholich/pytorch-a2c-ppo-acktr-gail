import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

# from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
# from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import gym
import time

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

env = gym.make('gym_aliengo:aliengo-v0', render=True)

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i


action = env.action_space.low
for k in range(3):
    assert (action[k::3] == action[k]).all()

i = 0
while True:
    # Obser reward and next obs
    action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, _, _ = env.step(action)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if i%1000 == 0:
        positions = obs[12:24] 
        for i in range(3):
            error = positions[i::3] - action[i::3]
            print(error)
        print()

    time.sleep(1./240)
    i +=1
