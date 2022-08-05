## RAPid-Learn : Learn to Recover and Plan Again for Novelties in Open World Environments

This repository contains the code for the RAPidLearn. A framework which integrates planning and learning to solve 
complex task in non stationary environments for open world learnings. The following code uses the gym-novel-gridworlds environments. The planner used in this case is the Metric FF_v_2.1. For any questions please contact shivam.goel@tufts.edu

Technical Appendix to **RAPid-Learn: A Framework for Learning to Recover for Handling Novelties in Open-World Environments.** in TechnicalAppendix.pdf


### Setup the planner

Download [MetricFF](https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.1.tgz).

`cd Metric-FF-v2.1` <BR>
`make . `

### Setup the virtual environment

Use the requirements file to install the dependencies 

`pip install -r requirements.txt`

[Optional] Create a conda environment

`conda create --name <YOUR_ENV_NAME> --python=3.7` <BR>
`conda activate <YOUR_ENV_NAME>` <BR>
`pip install -r requirements.txt` <BR>

### Setup the gym_novelgridworld

Install [gym_novel_gridworlds](https://github.com/gtatiya/gym-novel-gridworlds). <BR>

Switch to branch `adaptive_agents`

### Run the code

To run the RAPidLearn -->


Run <BR>
``` python experiment.py```

You can also use arguments. <BR>
``` python experiment.py --TP <trials pre novelty> --TN <trials_post_novelty> --N <novelty_name> --L <learner_name> -E <exploration_mode> -R <render>```

`novelty_name`
 1. `'axetobreakeasy'`
 2. `'axetobreakhard'`
 3. `'firecraftingtableeasy'`
 4. `'firecraftingtablehard'`
 5. `'rubbertree'`
 6. `'axefirecteasy'`
 7. `'rubbertreehard'`

`learner_name` <BR>
  1. `'epsilon-greedy'`
  2. `'smart-exploration'`   
 
 `exploration_mode` <BR>
  1. `'uniform'`
  2. `'ucb'`   

For questions please contact shivam.goel@tufts.edu
