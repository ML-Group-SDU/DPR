#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
####### MLP
#envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")
#
## crowdsourced labels (CS) + linear (MLP)
#domain="adroit"
#modality="state"
#structure="mlp"
#fake_label=false
#ensemble_size=3
#n_epochs=800
#num_query=2000
#len_query=50
#data_dir="../crowdsource_human_labels"
#seed=999
#exp_name="CS-MLP"
#
#for env in "${envs[@]}"
#do
#    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
#        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
#        seed=$seed exp_name=$exp_name > ./logs/${exp_name}-${env}-${seed}-${n_epochs}-${ensemble_size}-.log 2>&1 &
#done

######## TFM
#envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")
#envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1")
## crowdsourced labels (CS) + transformer (TFM)
#domain="adroit"
#modality="state"
#structure="transformer1"
#fake_label=false
#ensemble_size=3
#n_epochs=800
#num_query=2000
#len_query=50
#data_dir="../crowdsource_human_labels"
#seed=999
#exp_name="CS-TFM"
#max_seq_len=50
#
#for env in "${envs[@]}"
#do
#    nohup python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
#        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
#        seed=$seed exp_name=$exp_name max_seq_len=$max_seq_len > ./logs/${exp_name}-${env}-${seed}-${n_epochs}-${ensemble_size}-.log 2>&1 &
#done


###### DPR
envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")

# crowdsourced labels (CS) + transformer (TFM)
domain="adroit"
modality="state"
structure="DPR"
fake_label=false
ensemble_size=1
n_epochs=500
num_query=2000
len_query=50
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-DIFF"
max_seq_len=50

for env in "${envs[@]}"
do
    nohup python train_reward_model_diff.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
        seed=$seed exp_name=$exp_name max_seq_len=$max_seq_len >/dev/null 2>&1 &
done

#envs=("pen-human-v1" "pen-cloned-v1" "door-human-v1" "door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")
#envs=("door-cloned-v1" "hammer-human-v1" "hammer-cloned-v1")
#domain="adroit"
#modality="state"
#structure="C-DPR"
#fake_label=false
#ensemble_size=3
#n_epochs=500
#num_query=2000
#len_query=50
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
