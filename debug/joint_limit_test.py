# python save_video.py --load-dir trained_models/ppo --env-name "MinitaurBulletEnv-v0"

import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import cv2
import gym
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
parser.add_argument(
    '--num-vids',
    type=int,
    default=1,
    help='number of videos to record and save')
args = parser.parse_args()


args.det = not args.non_det

# env = make_vec_envs(
#     args.env_name,
#     args.seed + 1000,
#     1,
#     None,
#     None,
#     device='cpu',
#     allow_early_resets=False,
#     render=False)

env = gym.make('gym_aliengo:aliengo-v0')




obs = env.reset()


if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i


action = env.action_space.low
print('low:', action)

# breakpoint()
frames = 4 * 60
for i in range(args.num_vids):
    img_array = []
    for frame_num in range(frames):

        # for k in range(3):
        #     assert (action[k::3] == action[k]).all()


        obs, reward, _, _ = env.step(action)


        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if frame_num%2 == 0:
            img = env.render(mode='rgb_array')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_array.append(img)


    height, width, layers = img.shape
    size = (width, height)

    if not os.path.exists('videos/' + args.env_name):
        os.makedirs('videos/' + args.env_name)
    out = cv2.VideoWriter('videos/' + args.env_name + '/vid_'+ str(i) + 'jointlimtest_low.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for img in img_array:
        out.write(img)
    out.release()

action = env.action_space.high
print('high:', action)
# breakpoint()
frames = 4 * 60
for i in range(args.num_vids):
    img_array = []
    for frame_num in range(frames):

        for k in range(3):
            assert (action[k::3] == action[k]).all()


        obs, reward, _, _ = env.step(action)


        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if frame_num%2 == 0:
            img = env.render(mode='rgb_array')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_array.append(img)


    height, width, layers = img.shape
    size = (width, height)

    if not os.path.exists('videos/' + args.env_name):
        os.makedirs('videos/' + args.env_name)
    out = cv2.VideoWriter('videos/' + args.env_name + '/vid_'+ str(i) + 'jointlimtest_high.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for img in img_array:
        out.write(img)
    out.release()

# if __name__ == '__main__':


