#!/usr/bin/env bash
set -e

wandb enabled

# no reward norm
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_no_obs_norm --no-rew-norm
done

# no observation norm
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_no_obs_norm --no-obs-norm
done

# entropy decay
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_default --use-linear-entropy-decay
done

# default
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_default
done





# no grad clip
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_no_grad_clip --max-grad-norm 999999999
done

# no advantage standardization
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_no_adv_stand --no-standardize-advantage
done


# no GAE
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_no_GAE
done

# linear LR decay
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_linear_lr_decay --use-linear-lr-decay
done

# clipped value loss
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_clipped_value_loss --use-clipped-value-loss
done

# no bootstrapping incomplete episodes
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000 --seed $i --wandb-run-name CartPole_no_bootstrap_incomplete
done



# # hyperparam sweep
for i in {1..5}
do
for EPOCHS in 10 20
do
for LR in 5e-4 5e-3 1e-3
do
for ENTROP in 0.0 0.01 0.02
do
for VALUE in 0.25 0.5
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr $LR --entropy-coef $ENTROP --value-loss-coef $VALUE --ppo-epoch $EPOCHS --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb-run-name CartPole_sweep_${EPOCHS}_${LR}_${ENTROP}_${VALUE}
done
done
done
done
done

wandb disabled