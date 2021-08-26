## RAPid-Learn : Learn to Recover and Plan Again for Novelties in Open World Environments

This repository contains the code for the RAPidLearn. A framework which integrates planning and learning to solve 
complex task in non stationary environments for open world learnings. The following code uses the gym-novel-gridworlds environments. The planner used in this case is the Metric FF_v_2.1. For any questions please contact shivam.goel@tufts.edu

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

Switch to branch `RAPid-Learn` 

### Run the code

<!-- Brain.py consists of the main code. 

To run the RAPidLearn -->
 
 <!-- In order to insert novelty do the following:
 Change the variable `self.novelty_name` value (line #100 in `brain.py`) to any of the following
 1. `'axetobreak'`
 2. `'firecraftingtable'`
 3. `'rubbertree'` -->

In order to run experiments run the following script <BR>
``` python experiment.py --TP <trials pre novelty> --N <novelty_name> --L <learner_name>```

`novelty_name`
 1. `'axetobreakeasy'`
 2. `'axetobreakhard'`
 3. `'firecraftingtableeasy'`
 4. `'firecraftingtablehard'`
 5. `'rubbertree'`

`learner_name` <BR>
  1. `'epsilon-greedy'`
  2. `'smart-exploration'`   

Run <BR>
``` python brain.py```
