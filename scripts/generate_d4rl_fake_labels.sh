#!/bin/bash

 domain="mujoco"
 num_query=2000
 len_query=200
 seed=42900

 # mujoco
 envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
       "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
 for env in "${envs[@]}"
 do
   python fast_track/generate_d4rl_fake_labels.py --domain $domain --env_name "$env" --num_query $num_query --len_query $len_query --seed $seed
   wait
 done
