# How to run the baseline experiments
## Pre-novelty training
Train both policy gradient and PPO on the prenovelty crafting task using:
  - `python experiment.py --novelty_name prenovelty --train_episodes 10000 --algorithm PPO --n_trials 5`
  - `python experiment.py --novelty_name prenovelty --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --exploration_mode ucb --learner guided_policy`

For both baselines find the trained model in `/RAPid-Learn/data/` (depending on date and hash, it will be called something like: 
`2021-09-06_13:37:28-baseline-PPO-50000episodes-rewardshapingon-68074bfd5da343678bbacc75a1dbdb70/prenovelty_model.zip` or .npz for policy gradient). 
Copy the full path to this model, strip the `.zip` extension and insert for `<BASE_MODEL_PG>` and `<BASE_MODEL_PPO>` respectively below:

## Post-novelty learning and evaluation
### PPO experiments
These experiments use the PPO implementation from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repo.
  - firecraftingtableeasy: `python experiment.py --novelty_name firecraftingtableeasy --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - firecraftingtablehard: `python experiment.py --novelty_name firecraftingtablehard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertree: `python experiment.py --novelty_name rubbertree --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertreehard: `python experiment.py --novelty_name rubbertreehard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axefirecteasy: `python experiment.py --novelty_name axefirecteasy --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakeasy: `python experiment.py --novelty_name axetobreakeasy --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakhard: `python experiment.py --novelty_name axetobreakhard --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - scrapeplank: `python experiment.py --novelty_name scrapeplank --train_episodes 10000 --algorithm PPO --n_trials 5 --experiment_id <EXPERIMENT_ID>`    

### PG experiments
These experiments use the custom policy gradient algorithm that is also used for the subgoal learners within the main RAPid-Learn architecture.
  - firecraftingtableeasy: `python experiment.py --novelty_name firecraftingtableeasy --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - firecraftingtablehard: `python experiment.py --novelty_name firecraftingtablehard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertree: `python experiment.py --novelty_name rubbertree --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - rubbertreehard: `python experiment.py --novelty_name rubbertreehard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axefirecteasy: `python experiment.py --novelty_name axefirecteasy --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakeasy: `python experiment.py --novelty_name axetobreakeasy --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - axetobreakhard: `python experiment.py --novelty_name axetobreakhard --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`  
  - scrapeplank: `python experiment.py --novelty_name scrapeplank --train_episodes 10000 --algorithm policy_gradient --n_trials 5 --experiment_id <EXPERIMENT_ID>`    
