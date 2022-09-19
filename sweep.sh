

wandb enabled

# default
# for i in {1..5}
# do
#     python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_default
# done

# no observation norm
# for i in {1..5}
# do
#     python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_obs_norm --no_obs_norm
# done

# no grad norm clipping
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_grad_clip --max-grad-norm 999999
done

wandb disabled