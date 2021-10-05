# How to run the baseline experiments
## Pre-novelty training
Train both policy gradient and PPO on the prenovelty crafting task using:
  - `python experiment.py --novelty_name prenovelty --train_episodes 10000 --algorithm PPO --n_trials 5`
  - `python experiment.py --novelty_name prenovelty --train_episodes 10000 --algorithm policy_gradient --n_trials 5`

Each of these commands will create a fresh `<EXPERIMENT_ID>` folder that all results are saved into.
For both baselines find the highest performing prenovelty model in 
`/RAPid-Learn/data/<EXPERIMENT_ID>/prenovelty/<TRIAL_ID>` (depending on date and hash, it will be called something like: 
`2267a0875d2e49f5b1f910d9b0f7df0b-2021-09-30_22:10:50-PPO-10000episodes-rewardshapingon/prenovelty/trial-0/prenovelty_model.zip` or `.npz` for policy gradient). 
Make a copy of this model, put it in the `/RAPid-Learn/data/<EXPERIMENT_ID>/prenovelty` folder. This makes it available 
to use as a transfer model for the novelty experiments. Insert for `<EXPERIMENT_ID>` respectively below.

## Post-novelty learning and evaluation
### PPO experiments
These experiments use the PPO implementation from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repo.
  - firecraftingtablehard: `python experiment.py --novelty_name firecraftingtablehard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertreehard: `python experiment.py --novelty_name rubbertreehard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axefirecteasy: `python experiment.py --novelty_name axefirecteasy --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakhard: `python experiment.py --novelty_name axetobreakhard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - scrapeplank: `python experiment.py --novelty_name scrapeplank --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`    

### PG experiments
These experiments use the custom policy gradient algorithm that is also used for the subgoal learners within the main RAPid-Learn architecture.
  - firecraftingtablehard: `python experiment.py --novelty_name firecraftingtablehard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertreehard: `python experiment.py --novelty_name rubbertreehard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axefirecteasy: `python experiment.py --novelty_name axefirecteasy --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakhard: `python experiment.py --novelty_name axetobreakhard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - scrapeplank: `python experiment.py --novelty_name scrapeplank --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`    
