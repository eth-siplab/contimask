#!/usr/bin/env bash

# TO COMPARE THE RARE FEATURE VALUE EXPERIMENTS ACROSS DIFFERENTLY PARAMETERIZED MASKS
# H=32
python3 rare_feature_value.py --hidden_dim 32 --L 34 --pert FMA --mask MFF --optim G  --device 0 &
python3 rare_feature_value.py --hidden_dim 32 --L 32 --pert FMA --mask MFF --optim G  --device 1 &
python3 rare_feature_value.py --hidden_dim 32 --L 24 --pert FMA --mask MFF --optim G  --device 2 &
wait

# H=16
python3 rare_feature_value.py --hidden_dim 16 --L 24 --pert FMA --mask MFF --optim G  --device 0 &
python3 rare_feature_value.py --hidden_dim 16 --L 12 --pert FMA --mask MFF --optim G  --device 1 &
wait

# H=8
python3 rare_feature_value.py --hidden_dim 8 --L 24 --pert FMA --mask MFF --optim G --device 0 &
python3 rare_feature_value.py --hidden_dim 8 --L 12 --pert FMA --mask MFF --optim G --device 1 &
wait