

wandb enabled

# default
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_default
done

# no observation norm
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_obs_norm --no-obs-norm
done


# no grad clip
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_grad_clip--max-grad-norm 999999999
done

# no advantage standardization
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_adv_stand --no-standardize-advantage
done


# no GAE
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_no_GAE
done

# linear LR decay
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_linear_lr_decay --use_linear_lr_decay
done

# clipped value loss
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 1.0 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_clipped_value_loss --use_clipped_value_loss
done

# no bootstrapping incomplete episodes
for i in {1..5}
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr 1e-3 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000 --seed $i --wandb_run_name CartPole_no_bootstrap_incomplete
done



# # hyperparam sweep
for i in {1..5}
for EPOCHS in 10 20
for LR in 5e-4 5e-3 1e-3
for ENTROP in 0.0 0.01 0.02
for VALUE in 0.25 0.5
do
do
do
do
do
    python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 300 --num-processes 2 --lr $LR --entropy-coef $ENTROP --value-loss-coef $VALUE --ppo-epoch $EPOCHS --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 50000  --use-proper-time-limits --seed $i --wandb_run_name CartPole_sweep_$EPOCHS_$LR_$ENTROP_$VALUE
done
done
done
done
done

wandb disabled