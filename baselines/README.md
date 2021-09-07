# How to run baseline experiments
Commands to replicate the baseline experiments:
Train a model on the basetask using:
  - `python experiment.py --experiment baseline --train_episodes 50000 --reward_shaping --algorithm PPO`

Find the base model in the `/RAPid-Learn/data/` folder (depending on date and hash, it will be called something like: 
`2021-09-06_13:37:28-baseline-PPO-50000episodes-rewardshapingon-68074bfd5da343678bbacc75a1dbdb70/model_3250000_steps.zip`). 
Copy the full path to this model, strip the `.zip` extension and insert for `<BASE_MODEL>` below:

## Novelty injection experiments
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name firecraftingtablehard --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name rubbertree --trials_pre_novelty 10 --trials_post_learning 20`
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axefirecteasy --trials_pre_novelty 10 --trials_post_learning 20`

## Transfer learning experiment
This experiment requires a second step, as we're testing how quickly the hard version of AXE_TO_BREAK can be learnt, 
when starting from a model competent at the easy version.
Again, use the base model as a starting point to learn the novelty:
  - `python experiment.py --experiment baseline --load_model <BASE_MODEL>  --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axetobreakeasy`

As before, find the path to this model, strip the `.zip` extension and insert for `<AXETOBREAKEASY_MODEL>` below:
But this time, use this 
  - `python experiment.py --experiment baseline --load_model <AXETOBREAKEASY_MODEL> --train_episodes 50000 --reward_shaping --algorithm PPO --novelty_name axetobreakhard`


