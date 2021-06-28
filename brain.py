'''
# author: Shivam Goel
# email: shivam.goel@tufts.edu

# This file is the Brain of SPELer.
# It is the main file that needs to be run to run the framework

Important References




'''

# load an environment.
# connect with the planning agent. 
# generate the plan and peform it on the game engine. 
# for some trials. It can be called warmup.
##

# insert a new environment in the middle (though a random number).
# plan and see
# call SPELer to investigate and the agent enters the exploration mode. 
# learns to reach a state based upon the reward function generated according to the failed step in a plan.
# #

# instantiate the planning instance. callback here
# instantiate a learner instance. callback here
##



# call a rl learner which communicates with the planner agent to 
# find a plannable state. A threshld can be setup here and a hyperparameter 
#   can be learned from here. an agent can also solve two things at once
# learn a policy to learn to solve the RL problem

# instantiate a operator_learner instance to convert the policy to a set of pre condition and post effect 
# update the PDDL representation of the agent.

# call the planner to plan and solve.
# 

###

# inseet at least 5 new environments.

### 

# repeat the process above and add the functionality to control the number of iterations here.

##
# make a new filoe called experiment.py which runs this script accordingly.

import os
import csv
import math
import argparse

import gym
import numpy as np
import gym_novel_gridworlds

from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap



class Brain:
    
    def __init__(self) -> None:
        pass

    def init_brain():
        pass


    def reset_brain():
        pass

    def run_brain():
        # this is the driving function of this class.
        # call the environment and run the envuronment for x number of trials.
        # 

        pass


    def call_learner():
        # This function instantiates a RL learner to start finding interesting states to send 
        # to the planner
        # 

        # 
        pass

    def call_planner():
        # this function wehn called calls the planner
        pass

    def instantiate_env(env_id, novelty_family, inject):

        ###
        # This function instantiate a new instance of the environment for the agent to interact with.
        # All the novelty is injected here.
        # it takes the env_ID and the novelty arguemtns list as input.
        # it returns an instance of the environment.
        ###

        # env2 = gym.make(env_id)

        # env2.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
        # env2 = LimitActions(env2, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
        # env2 = LidarInFront(env2) # generate the observation space using LIDAR sensors
        # # print(env.unbreakable_items)
        # env2.reward_done = 1000
        # env2.reward_intermediate = 50
        # if inject:
        #     env2 = inject_novelty(env2, novelty_family[0], novelty_family[1], novelty_family[2], novelty_family[3])
        # check_env(env2, warn=True) # check the environment    
        # return env2


        pass

    def call_operator_learner():
        pass
    
    def inject_novelty():
        pass
    
    def run_trials():
        pass



    def evaluate_policy():
        pass
# if __name__ == main():





