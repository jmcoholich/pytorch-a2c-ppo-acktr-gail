# python save_video.py --load-dir trained_models/ppo --env-name "gym_aliengo:MinitaurBulletEnv_Friction-v0"
'''
python save_video.py --load-dir trained_models/ppo --env-name "gym_aliengo:aliengo-v0"
python save_video.py --load-dir trained_models/None5/ppo --env-name "gym_aliengo:aliengo-v0"
python save_video.py --load-dir trained_models/Experiment1_more_logging3/ppo --env-name "gym_aliengo:aliengo-v0"
python save_video.py --load-dir trained_models/None58641/ppo --env-name "gym_aliengo:aliengo-v0"
'''
import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import cv2
import matplotlib.pyplot as plt


# FPS = 24 # of the video, NOT of the simulation (which is kept constant at 240 Hz)
# VIDEO_SPEED = 0.1 # how fast the video plays vs real time

# assert 240.0 * VIDEO_SPEED % FPS == 0
# assert 240.0 * VIDEO_SPEED / FPS >= 1


def add_frame(render_func, img_array):
    img = render_func('rgb_array')
    height, _, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.putText(np.float32(img), ('%f'% VIDEO_SPEED).rstrip('0') + 'x Speed'  , (1, height - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
    img = cv2.putText(np.float32(img), '%d FPS' % FPS , (1, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
    img = np.uint8(img)
    img_array.append(img)
    return img_array


def write_video(img_array, env_name, fps, args):
    height, width, layers = img_array[0].shape
    size = (width, height)
    if not os.path.exists('videos/' + env_name):
        os.makedirs('videos/' + env_name)
    
    filename = os.path.join('videos', env_name, os.path.split(os.path.split(args.load_dir)[0])[1] + '.avi' )
    print(filename)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for img in img_array:
        out.write(img)
    out.release()
    print('Video saved')


def create_plots(commanded_actions, mapped_joint_positions, env_name, args):
    # indices are in order of [shoulder, hip, knee] for FR, FL, RR, RL
    parts = ['shoulder', 'hip', 'knee']
    positions = ['FR', 'FL', 'RR', 'RL']
    labels = [positions[i//3]+ '_' + parts[i%3] for i in range(12)]
    n = len(commanded_actions)//12
    commanded_actions = commanded_actions.reshape((n, 12))
    mapped_joint_positions = mapped_joint_positions.reshape((n, 12))
    # reorder so that the subplot locations map to robot leg positions
    reorder = [4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9 ]
    labels = [labels[reorder[i]-1] for i in range(12)]
    commanded_actions[:,np.arange(12)] = commanded_actions[:, np.array(reorder)-1]
    mapped_joint_positions[:,np.arange(12)] = mapped_joint_positions[:, np.array(reorder)-1]

    fig = plt.figure(figsize=(20.0, 11.25))
    for i in range(12):
        ax = fig.add_subplot(6, 2, i + 1)
        ax.plot(commanded_actions[:,i], label='commanded joint position')
        ax.plot(mapped_joint_positions[:,i], label='actual joint position')
        ax.hlines([-1,1], 0, n, 'k', linestyles='dotted')
        loc = 'left' if (i+1) % 2 == 1 else 'right'
        ax.set_title(labels[i], loc=loc)
        ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=2)
    fig.suptitle('Commanded vs Actual Joint Positions -- Mapped to Action Space [-1.0, 1.0]')
    filename = os.path.join('videos', env_name, os.path.split(os.path.split(args.load_dir)[0])[1] + '.png' )
    fig.savefig(filename)
    print('Plots Saved')


def main():
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
        default=True,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--num-vids',
        type=int,
        default=1,
        help='number of videos to record and save')
    args = parser.parse_args()


    args.det = not args.non_det
    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False,
        render=False)

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    # Start run ############################################################################################################

    assert 240.0%env.venv.venv.envs[0].n_hold_frames == 0
    global FPS
    global VIDEO_SPEED
    FPS = int(240.0/env.venv.venv.envs[0].n_hold_frames)
    VIDEO_SPEED = 1.0
    obs = env.reset()
    img_array = []
    counter = 0
    img_array = add_frame(render_func, img_array)
    commanded_actions = np.array([])
    mapped_joint_positions = np.array([])

    episode_rew = 0
    while True: # Sample an episode
        with torch.no_grad():
            value, action, _, recurrent_hidden_states, _ = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # try period of 0.8 seconds = 150 bpm = 1.25 full cycle frequency
        # a sin wave in 0.8 seconds
    
        # t = counter * env.venv.venv.envs[0].n_hold_frames/240.
        # period = 0.8
        # action  = np.ones(12)
        # action[[0,3,6,9]] = 0
        # action[[1,10]] = np.sin((np.pi * 2 / period * t)) * 0.5 + 0.25 # thighs
        # action[[2,11]] = 1 #np.sin((np.pi * 2 / period * t)) * 0.5 + 0.5
        # action[[4,7]]  = np.sin((np.pi * 2 / period * t) + np.pi) * 0.5 + 0.25 # thighs
        # action[[5,8]]  = 1 # np.sin((np.pi * 2 / period * t) + np.pi) * 0.5 + 0.5
        # action = torch.from_numpy(action).unsqueeze(0) 
        obs, reward, done, info = env.step(action)
        episode_rew += reward
        commanded_actions = np.append(commanded_actions, action[0].numpy(),axis=0)
        joint_positions = env.venv.venv.envs[0].joint_positions
        mapped_joint_positions = np.append(mapped_joint_positions, 
                                env.venv.venv.envs[0]._positions_to_actions(joint_positions),axis=0)

        # if counter % (240 * VIDEO_SPEED/ FPS) == 0 :
        img_array = add_frame(render_func, img_array)
        counter += 1
        # done = False
        # if t >= period:
        #     break
        if done:
            break

    print('Total Episode Reward: %0.2f' %episode_rew)
    create_plots(commanded_actions, mapped_joint_positions, args.env_name, args)
    write_video(img_array, args.env_name, FPS, args)
    return episode_rew

if __name__ == '__main__':
    main()




