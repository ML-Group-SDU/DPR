#!/bin/bash

#################### CS-MLP ######################

# CS-MLP-Hopper-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Hopper-medium-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hopper-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Hopper-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=hopper
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-replay-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hopper-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Hopper-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Walker2d-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Walker2d-medium-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Walker2d-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Walker2d-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:2"
    env_type=walker2d
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-replay-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Walker2d-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Walker2d-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:3"
    env_type=walker2d
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-expert-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Halfcheetah-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Halfcheetah-medium-v2
    mkdir -p ./logs/$name
    device="cuda:4"
    env_type=halfcheetah
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Halfcheetah-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Halfcheetah-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:4"
    env_type=halfcheetah
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-replay-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Halfcheetah-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-MLP-TD3BC-Halfcheetah-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:5"
    env_type=halfcheetah
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-expert-v2/CS-MLP/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################### CS-TFM ######################

# CS-TFM-Hopper-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Hopper-medium-v2-51
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hopper-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Hopper-medium-replay-v2-51
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=hopper
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-replay-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hopper-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Hopper-medium-expert-v2-51
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Walker2d-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Walker2d-medium-v2-51
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Walker2d-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Walker2d-medium-replay-v-51
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=walker2d
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-replay-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Walker2d-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Walker2d-medium-expert-v2-51
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-expert-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Halfcheetah-medium-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Halfcheetah-medium-v2-51
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Halfcheetah-medium-replay-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Halfcheetah-medium-replay-v2-51
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=halfcheetah
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-replay-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Halfcheetah-medium-expert-v2
for ((seed=0; seed<5; seed+=1))
do
    name=CS-TFM-TD3BC-Halfcheetah-medium-expert-v2-51
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-expert-v2/CS-TFM/epoch_800_query_2000_len_200_seed_999/models/reward_model_51.pt"
    reward_model_type=transformer
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################### CS-DPR ######################

# CS-DPR-Hopper-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Hopper-medium-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Hopper-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Hopper-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=hopper
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-replay-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Hopper-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Hopper-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Walker2d-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Walker2d-medium-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Walker2d-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Walker2d-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=walker2d
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-replay-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model1.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Walker2d-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Walker2d-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-expert-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Halfcheetah-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Halfcheetah-medium-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Halfcheetah-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Halfcheetah-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=halfcheetah
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-replay-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-DPR-Halfcheetah-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-TD3BC-Halfcheetah-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-expert-v2/CS-DPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################### CS-CDPR ######################

# CS-CDPR-Hopper-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Hopper-medium-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Hopper-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Hopper-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=hopper
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-replay-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Hopper-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Hopper-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hopper
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/hopper-medium-expert-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Walker2d-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Walker2d-medium-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Walker2d-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Walker2d-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=walker2d
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-replay-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Walker2d-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Walker2d-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=walker2d
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/walker2d-medium-expert-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Halfcheetah-medium-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Halfcheetah-medium-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Halfcheetah-medium-replay-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Halfcheetah-medium-replay-v2
    mkdir -p ./logs/$name
    device="cuda:1"
    env_type=halfcheetah
    dataset=medium_replay_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-replay-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-CDPR-Halfcheetah-medium-expert-v2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-TD3BC-Halfcheetah-medium-expert-v2
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=halfcheetah
    dataset=medium_expert_v2
    reward_model_path="./rlhf/reward_model_logs/halfcheetah-medium-expert-v2/CS-CDPR/epoch_500_query_2000_len_200_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/td3_bc/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/td3_bc_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done
