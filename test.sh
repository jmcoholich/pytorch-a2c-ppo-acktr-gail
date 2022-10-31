#!/usr/bin/env bash
set -e

wandb enabled


# default
# for i in {1..2}
# do
#     python main.py --env-name "BipedalWalker-v3" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 500000  --use-proper-time-limits --seed $i --wandb-run-name walker_default
# done

for i in {1..2}
do
    python main.py --env-name "BipedalWalker-v3" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 8 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000  --use-proper-time-limits --seed $i --wandb-run-name walker_default3
done

for i in {1..2}
do
    python main.py --env-name "BipedalWalker-v3" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 16 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000  --use-proper-time-limits --seed $i --wandb-run-name walker_default4
done

# pip install box2d-py
# https://github.com/automl/auto-sklearn/issues/314


