#!/bin/bash

python test_agents.py --agent_type 2 --methodid 1 --limit_action True --trial_num 0 --config_num 1 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 1000
#python test_agents.py --agent_type 2 --methodid 1 --limit_action True --trial_num 1 --config_num 1 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 10000
#python test_agents.py --agent_type 2 --methodid 1 --limit_action True --trial_num 2 --config_num 1 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 10000
#
#python test_agents.py --agent_type 2 --methodid 1 --limit_action False --trial_num 0 --config_num 2 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 10000
#python test_agents.py --agent_type 2 --methodid 1 --limit_action False --trial_num 1 --config_num 2 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 10000
#python test_agents.py --agent_type 2 --methodid 1 --limit_action False --trial_num 2 --config_num 2 -H 32 --cache-size 5000 --history-len 1 --mem-size 50000 --prepopulate 500 --priority 0 --return-est pengs-median --update-freq 1 --num_timesteps 10000
