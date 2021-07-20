'''
# author: Shivam Goel
# email: shivam.goel@tufts.edu

# This file is the Brain of RAPidL.
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

import operator_generalization
import os
import re
import sys
import csv
import math
import time
import copy
import argparse
import subprocess

import gym
import numpy as np
import gym_novel_gridworlds

from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
from generate_pddl import *
from learner import *
# from pddl_generator import *
from operator_generalization import *

# import sys, string, os, arcgisscripting
# os.system("C:/Documents and Settings/flow_model/flow.exe")
# dictionary that maps the PDDL actions to the game engine permissable actions.
map = {'moveforward':'Forward',
        'turnleft':'Left',
        'turnright':'Right',
        'breakblock': 'Break',
        'placetreetap':'Place_tree_tap',
        'extractrubber':'Extract_rubber',
        'craftplanks':'Craft_plank',
        'craftsticks':'Craft_stick',
        'crafttreetap': 'Craft_tree_tap',
        'craftpogostick': 'Craft_pogo_stick',
        'select': 'Select'}

class Brain:
    
    def __init__(self):
        self.learner = None
        pass

    def init_brain(self):
        pass


    def reset_brain(self):
        pass

    def run_brain(self, env_id=None):
        '''        
            This is the driving function of this class.
            Call the environment and run the environment for x number of trials.
        '''

        # env = gym.make('NovelGridworld-Pogostick-v1')
        # done = False
        # while not done:
        #     env.render()
        #     action = env.action_space.sample()
        #     obs, reward, done, info = env.step(action)

        # env.close()
        # sys.exit(1)
        env_id = 'NovelGridworld-Pogostick-v1' # hardcoded for now. will add the argparser later.
        env = self.instantiate_env(env_id, None, False) # make a new instance of the environment.
        obs = env.reset() 
        self.generate_pddls(env)
        plan, game_action_set = self.call_planner("domain", "problem", env) # get a plan
        # print("game action set aftert the planner = {}".format(game_action_set))
        # time.sleep(10000)
        result, failed_action = self.execute_plan(env, game_action_set, obs)
        
        if not result:
            learned_policy = self.call_learner(failed_action=failed_action, env=env)
            learned_operator =  self.call_operator_learner(failed_action, learned_policy)
            # This will update the PDDLs with the new operator for planning.
            self.generate_pddls(env, learned_operator, update_flag = True)
            plan, game_action_set = self.call_planner("domain", "problem", env) # get a plan

        
        if result:
            print("succesfully completed the task without any hassle!")
        # if not result: # if the results in plan failure.
            # call the learner with the failed operator

            # call the policy_learner to convert the policy to an operator.
        # TODO
        # store the learned policy in two buffers. Replay buffer and the 
        # buffer which stores the transferrable policies.




        # run the planner's OP to the domain as the step
        pass


    def call_learner(self, failed_action, env=None):
        # This function instantiates a RL learner to start finding interesting states to send 
        # to the planner
        #

        if env is None:
            env_id = 'NovelGridworld-Pogostick-v1'  # hardcoded for now. will add the argparser later.
            env = self.instantiate_env(env_id, None, False)  # make a new instance of the environment.
            obs = env.reset()

        #TODO: If we want the learner to perform training on the failed action for N episodes,
        #        then we would have to reset to the failed step between letting the RL agent take over.
        #      Took out planning stuff so it wouldn't know how to reach a latter plan step to start
        #        learning a policy from
        if self.learner is None:
            self.learner = Learner(failed_action, env)
        else:
            self.learner.learn_state()
        return self.learner


        # 
        pass

    def call_planner(self, domain, problem, env):
        '''
            Given a domain and a problem file
            This function return the ffmetric Planner output.
            In the action format
        '''
    
        run_script = "Metric-FF-v2.1/./ff -o "+self.pddl_dir+os.sep+domain+".pddl -f "+self.pddl_dir+os.sep+problem+".pddl -s 0"
        output = subprocess.getoutput(run_script)
        plan, game_action_set = self._output_to_plan(output, env)
        return plan, game_action_set

    def _output_to_plan(self, output, env):
        '''
        Helper function to perform regex on the output from the planner.
        ### I/P: Takes in the ffmetric output and
        ### O/P: converts it to a action sequence list.
        '''

        ff_plan = re.findall(r"\d+?: (.+)", output.lower()) # matches the string to find the plan bit from the ffmetric output.
        action_set = []
        for i in range (len(ff_plan)):
            action_set.append(ff_plan[i].split(" ")[0])
        # print (ff_plan[0].split())

        if "unsolvable" in output:
            print ("Plan not found with FF! Error: {}".format(
                output))
        if ff_plan[-1] == "reach-goal":
            ff_plan = ff_plan[:-1]
        

        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)
        # print ("game action set = {}".format(game_action_set))
        for i in range(len(game_action_set)):
            game_action_set[i] = map[game_action_set[i]]
        for i in range(len(game_action_set)):
            game_action_set[i] = env.actions_id[game_action_set[i]]
        
        return action_set, game_action_set

    def instantiate_env(self, env_id, novelty_family, inject):
        '''
         This function instantiate a new instance of the environment for the agent to interact with.
         All the novelty is injected here.
         ### I/P: it takes the env_ID and the novelty arguemtns list as input.
         ### O/P: it returns an instance of the environment.
        '''
        env = gym.make(env_id)
        # env.render()
        # print ("env.actions_id = {}".format(env.actions_id))
        env.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
        # env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
        # env = LidarInFront(env) # generate the observation space using LIDAR sensors
        # print(env.unbreakable_items)
        env.reward_done = 1000
        env.reward_intermediate = 50
        if inject:
            env = inject_novelty(env, novelty_family[0], novelty_family[1], novelty_family[2], novelty_family[3]) 
        return env

    def call_operator_learner(self, failed_action, learned_policy):
        '''
        This function calls the operator learner class.
        It converts the learned policy to the operator and updates the domain file of the PDDL representation.
        '''

        og = operator_generalization(failed_action, learned_policy)
        # learned_operator = og.policy_to_action()
        return og.policy_to_action()
        pass
    
    def inject_novelty():
        pass
    
    def run_trials():
        pass

    def generate_pddls(self, env, learned_operator = None, update_flag = False):
        self.pddl_dir = "PDDL"
        os.makedirs(self.pddl_dir, exist_ok = True)
        generate_prob_pddl(self.pddl_dir, env)
        if learned_operator is not None:
            generate_domain_pddl(self.pddl_dir, env, learned_operator)
    '''
    This function generates the PDDLs from the current environment instance
    ### I/P environment object
    ### O/P returns pddl names if PDDL generated successfully, else returns false.
    '''

    ### Yash's functions will be integrated here.




    def execute_plan(self, env, plan, obs):
        '''
        This function executes the plan on the domain step by step
        ### I/P: environment instance and sequence of actions step by step
        ### O/P: SUCCESS/ FAIL with the failed action
        '''
        print ("env.actions_id = {}".format(env.actions_id))
        # obs = env.reset()
        # env.render()
        rew_eps = 0
        count = 0
        env.render()
        for i in range (len(plan)):
            print ("Taking action {} = {}".format(i,list(env.actions_id.keys())[list(env.actions_id.values()).index(plan[i])]))
            env.render()
            obs, reward, done, info = env.step(plan[i])
            time.sleep(0.5)
            rew_eps += reward
            count += 1
            # if args['render']:
            # if i == 1:
            if done:
                if env.inventory_items_quantity[env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                    # count = step
                    # break
                    return True, None
            # if info.equals('FAIL'):
            #     return False, plan[i]
        return False, None

    def evaluate_policy():
        pass
if __name__ == '__main__':
    brain1 = Brain()
    #testing RL connection
    learned_policy = brain1.call_learner(failed_action='break minecraft:log')
    # brain1.run_brain()






