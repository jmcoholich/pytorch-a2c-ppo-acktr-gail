"""
python cherry_pick_vid.py --load-dir trained_models/None912626/ppo --env-name "gym_al│·······
iengo:aliengo-v0"
"""


import save_video
episode_rew = 0
max_reward = -1e10
summed_rew = 0
counter = 0
min_reward = 1e10
while episode_rew < 800:
    episode_rew = save_video.main()
    counter += 1
    summed_rew += episode_rew
    if episode_rew > max_reward:
        max_reward = episode_rew
    if episode_rew < min_reward:
        min_reward = episode_rew
    print('\nMax Reward: %.2f'%max_reward)
    print('Min Reward: %.2f'%min_reward)
    print('Avg Reward: %.2f\n'%(summed_rew/counter))