#!/bin/bash
export CUDA_VISIBLE_DEVICES=2


envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
  "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
## crowdsourced labels (CS) + linear (MLP)
#domain="mujoco"
#modality="state"
#structure="mlp"
#fake_label=false
#ensemble_size=3
#n_epochs=800
#num_query=2000
#len_query=200
#data_dir="../crowdsource_human_labels"
#seed=999
#exp_name="CS-MLP"
#for env in "${envs[@]}"
#do
#    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
#        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
#        seed=$seed exp_name=$exp_name > ./logs/${exp_name}-${env}-${seed}-${n_epochs}-${ensemble_size}-.log 2>&1 &
#done

# TFM
#envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
#  "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
## crowdsourced labels (CS) + TFM
#domain="mujoco"
#modality="state"
#structure="transformer1"
#fake_label=false
#ensemble_size=3
#n_epochs=800
#num_query=2000
#len_query=200
#data_dir="../crowdsource_human_labels"
#seed=999
#exp_name="CS-TFM"
#
#for env in "${envs[@]}"
#do
#    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
#        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
#        seed=$seed exp_name=$exp_name > ./logs/${exp_name}-${env}-${seed}-${n_epochs}-${ensemble_size}-.log 2>&1 &
#done

# DPR
envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
  "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
# crowdsourced labels (CS) + DPR
domain="mujoco"
modality="state"
structure="DPR"
fake_label=false
ensemble_size=3
n_epochs=500
num_query=2000
len_query=200
data_dir="../crowdsource_human_labels"
seed=999
disc_lr=0.0006
exp_name="CS-DPR"
n_steps=10
for env in "${envs[@]}"; do
  nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
    ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir n_steps=$n_steps \
    seed=$seed exp_name=$exp_name disc_lr=$disc_lr >./logs/${exp_name}-${env}.log 2>&1 &
done

#envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
       #  "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
## crowdsourced labels (CS) + CDPR
#domain="mujoco"
#modality="state"
#structure="C-DPR"
#fake_label=false
#ensemble_size=3
#n_epochs=500
#num_query=2000
#len_query=200
#data_dir="../crowdsource_human_labels"
#seed=999
#max_seq_len=50
#disc_lr=0.0003
#exp_name="CS-CDPR"
#lable_dim=10
#discrim_depth=4
#num_units=256
#n_steps=10
#device="cuda:0"
#disc_momentum=0.9
#
#for env in "${envs[@]}"
#do
#    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
#        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
#        lable_dim$lable_dim discrim_depth$discrim_depth num_units=$num_units n_steps=$n_steps device=$device disc_momentum=$disc_momentum \
#        seed=$seed exp_name=$exp_name max_seq_len=$max_seq_len  disc_lr=$disc_lr > ./logs/${exp_name}-${env}-${seed}-${n_epochs}-${ensemble_size}-${disc_lr}.log 2>&1 &
#done
