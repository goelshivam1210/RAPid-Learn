# How to run the baseline experiments
## Pre-novelty training
Train both policy gradient and PPO on the prenovelty crafting task using:
  - `python experiment.py --experiment baseline --train_episodes 50000 --reward_shaping --algorithm PPO --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment policy_gradient --train_episodes 50000 --reward_shaping --trials_pre_novelty 10 --trials_post_learning 20`

For both baselines find the trained model in `/RAPid-Learn/data/` (depending on date and hash, it will be called something like: 
`2021-09-06_13:37:28-baseline-PPO-50000episodes-rewardshapingon-68074bfd5da343678bbacc75a1dbdb70/prenovelty_model.zip`). 
Copy the full path to this model, strip the `.zip` extension and insert for `<BASE_MODEL_PG>` and `<BASE_MODEL_PPO>` respectively below:

## Post-novelty learning and evaluation
### PPO experiments
These experiments use the PPO implementation from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) repo.
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL_PPO> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name firecraftingtablehard --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL_PPO> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name rubbertree --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL_PPO> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axefirecteasy --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL_PPO> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axetobreakhard --trials_pre_novelty 10 --trials_post_learning 20`
### PG experiments
These experiments use the custom policy gradient algorithm that is also used for the subgoal learners within the main RAPid-Learn architecture.
- firecraftingtablehard: `python experiment.py --experiment policy_gradient --load_model <BASE_MODEL_PG> --train_episodes 50000 --reward_shaping --novelty_name firecraftingtablehard --trials_pre_novelty 10 --trials_post_learning 20`  
- rubbertree: `python experiment.py --experiment policy_gradient --load_model <BASE_MODEL_PG> --train_episodes 50000 --reward_shaping --novelty_name rubbertree --trials_pre_novelty 10 --trials_post_learning 20`  
- axefirecteasy: `python experiment.py --experiment policy_gradient --load_model <BASE_MODEL_PG> --train_episodes 50000 --reward_shaping --novelty_name axefirecteasy --trials_pre_novelty 10 --trials_post_learning 20`
- axetobreakeasy: `python experiment.py --experiment policy_gradient --load_model <BASE_MODEL_PG> --train_episodes 50000 --reward_shaping --novelty_name axetobreakhard --trials_pre_novelty 10 --trials_post_learning 20`

## Transfer learning experiment
This experiment requires a second step, as we're testing how quickly the hard version of AXE_TO_BREAK can be learnt, 
when starting from a model competent at the easy version.
Again, use the base model as a starting point to learn the novelty:
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL>  --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axetobreakeasy`

As before, find the path to this model, strip the `.zip` extension and insert for `<AXETOBREAKEASY_MODEL>` below:
But this time, use this 
  - `python experiment.py --experiment baseline --load_model <AXETOBREAKEASY_MODEL> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axetobreakhard`
