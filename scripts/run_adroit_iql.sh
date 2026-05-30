#!/bin/bash

################### CS-MLP ######################
export CUDA_VISIBLE_DEVICES=5
# CS-MLP-Pen-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

export CUDA_VISIBLE_DEVICES=4
# CS-MLP-Pen-cloned-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-MLP-Door-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-MLP-Door-cloned-v1
export CUDA_VISIBLE_DEVICES=2
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hammer-human-v1
export CUDA_VISIBLE_DEVICES=1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-MLP-Hammer-cloned-v1
export CUDA_VISIBLE_DEVICES=3
for ((seed=0; seed<3; seed+=1))
do
    name=CS-MLP-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-MLP/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=mlp
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################### CS-TFM ######################
export CUDA_VISIBLE_DEVICES=0
# CS-TFM-Pen-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Pen-cloned-v1
export CUDA_VISIBLE_DEVICES=1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

export CUDA_VISIBLE_DEVICES=2
# CS-TFM-Door-human-v1
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Door-cloned-v1
export CUDA_VISIBLE_DEVICES=3
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hammer-human-v1
export CUDA_VISIBLE_DEVICES=4
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

# CS-TFM-Hammer-cloned-v1
export CUDA_VISIBLE_DEVICES=5
for ((seed=0; seed<3; seed+=1))
do
    name=CS-TFM-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-TFM/epoch_800_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=transformer1
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &

    echo "$name $seed training start!"
done

################### CS-DPR ######################

# CS-DPR-IQL-Pen-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<5; seed+=1))
do
    name=CS-DPR-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-DPR-IQL-Pen-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<5; seed+=1))
do
    name=CS-DPR-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done


# CS-DPR-IQL-Door-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-DPR-IQL-Door-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-DPR-IQL-Hammer-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-DPR-IQL-Hammer-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-DPR-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-DPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

################### CS-CDPR ######################

# CS-CDPR-IQL-Pen-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<5; seed+=1))
do
    name=CS-CDPR-IQL-Pen-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/pen-human-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-CDPR-IQL-Pen-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<5; seed+=1))
do
    name=CS-CDPR-IQL-Pen-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=pen
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/pen-cloned-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done


# CS-CDPR-IQL-Door-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-IQL-Door-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/door-human-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-CDPR-IQL-Door-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-IQL-Door-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=door
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/door-cloned-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-CDPR-IQL-Hammer-human-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-IQL-Hammer-human-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=human_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-human-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done

# CS-CDPR-IQL-Hammer-cloned-v1
export CUDA_VISIBLE_DEVICES=0
for ((seed=0; seed<3; seed+=1))
do
    name=CS-CDPR-IQL-Hammer-cloned-v1
    mkdir -p ./logs/$name
    device="cuda:0"
    env_type=hammer
    dataset=cloned_v1
    reward_model_path="./rlhf/reward_model_logs/hammer-cloned-v1/CS-CDPR/epoch_500_query_2000_len_50_seed_999/models/reward_model.pt"
    reward_model_type=C-DPR
    config_path=./configs/offline/iql/$env_type/$dataset.yaml
    nohup python -u algorithms/offline/iql_p.py --device $device --seed $seed \
    --reward_model_path $reward_model_path --config_path $config_path \
    --reward_model_type $reward_model_type --seed $seed --name $name \
    >./logs/$name/$seed.log 2>&1 &
    echo "$name $seed training start!"
done