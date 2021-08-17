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

import numpy as np
from scipy.spatial import distance
import gym
import gym_novel_gridworlds


from utils import AStarOperator, AStarPlanner

from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
from generate_pddl import *
from learner_v2 import *
from operator_generalization import *

action_map = {'moveforward':'Forward',
        'turnleft':'Left',
        'turnright':'Right',
        'break': 'Break',
        'place':'Place_tree_tap',
        'extractrubber':'Extract_rubber',
        'craftplank': 'Craft_plank',
        'craftstick':'Craft_stick',
        'crafttree_tap': 'Craft_tree_tap',
        'craftpogo_stick': 'Craft_pogo_stick',
        'select': 'Select'}

class Brain:
    
    def __init__(self):
        # self.novelty_name = 'rubber_tree'
        self.learner = None
        self.failed_action_set = {}
        self.novelty_name = None
        self.completed_trials = 0
        self.learned_policies_dict = {} # store failed action:learner_instance object

    def run_brain(self, env_id=None, novelty_name = None):
        '''        
            This is the driving function of this class.
            Call the environment and run the environment for x number of trials.
        '''
        env_id = 'NovelGridworld-Pogostick-v1' # hardcoded for now. will add the argparser later.
        env = self.instantiate_env(env_id) # make a new instance of the environment.
        # self.novelty_name = novelty_name # to be used in future.
        self.novelty_name = 'rubbertree'
        env = self.inject_novelty(novelty_name = self.novelty_name)
        obs = env.reset() 
        self.generate_pddls(env)
        self.domain_file_name = "domain"
        # print ("Actions ID from the env = {} ".format(env.actions_id))
        plan, game_action_set = self.call_planner(self.domain_file_name, "problem", env) # get a plan
        # print("game action set aftert the planner = {}".format(game_action_set))
        result, failed_action = self.execute_plan(env, game_action_set, obs)
        print ("result = {}  Failed Action =  {}".format(result, failed_action))
        
        if not result and failed_action is not None: # cases when the plan failed for the first time and the agent needs to learn a new action using RL
            print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
            self.learned = self.call_learner(failed_action=failed_action, env=env)
            
            if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
                print ("Agent succesfully learned a new action in the form of policy. Now resetting to test.")
                self.run_brain()
            
        if not result and failed_action is None: # The agent used the learned policy and yet was unable to solve
            print ("Failed to execute policy successfully. Now resetting for another trial.")
            self.run_brain()
 
        if result:
            print("succesfully completed the task without any hassle!")
            # print("Needed to transfer: ", self.completed_trails)

        pass


    def call_learner(self, failed_action, env=None, transfer = False):
        # This function instantiates a RL learner to start finding interesting states to send 
        # to the planner

        if env is None:
            env_id = 'NovelGridworld-Pogostick-v1'  # hardcoded for now. will add the argparser later.
            env = self.instantiate_env(env_id, None, False)  # make a new instance of the environment.
            obs = env.reset()

        if failed_action not in self.learned_policies_dict:
            self.learner = Learner(failed_action, env)
            self.learned_policies_dict[failed_action] = self.learner # save the learner instance object to the learned poliocies dict.
            learned = self.learner.learn_policy(self.novelty_name) # learn to reach the goal state, if reached save the learned policy using novelty_name
            return learned
        else:
            print("\n Learner Using the learned policy")
            played = self.learned_policies_dict[failed_action].play_learned_policy(env, novelty_name=self.novelty_name, operator_name=failed_action) # returns whether the policy was successfully played or not
            return played

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
        print ("ffplan = {}".format(ff_plan))
        action_set = []
        for i in range (len(ff_plan)):
            if ff_plan[i].split(" ")[0] == "approach":
                action_set.append(ff_plan[i])
            elif ff_plan[i].split(" ")[0] == "select":
                to_append = ff_plan[i].split(" ")
                sep = "_"
                to_append = sep.join(to_append).capitalize()
                action_set.append(to_append)

            else:
                action_set.append(ff_plan[i].split(" ")[0])

        if "unsolvable" in output:
            print ("Plan not found with FF! Error: {}".format(
                output))
        if ff_plan[-1] == "reach-goal":
            ff_plan = ff_plan[:-1]
        
        # print ("game action set  = {}".format(action_set))
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)
        # print ("game action set = {}".format(game_action_set))
        for i in range(len(game_action_set)):
            if game_action_set[i].split(" ")[0] != "approach" and game_action_set[i].split("_")[0] != "Select":
                game_action_set[i] = action_map[game_action_set[i]]
        # print ("game action set = {}".format(game_action_set))
        for i in range(len(game_action_set)):
            if game_action_set[i] in action_map:
                game_action_set[i] = env.actions_id[game_action_set[i]]
        # print (game_action_set)
        return action_set, game_action_set

    def instantiate_env(self, env_id, novelty_family=None, inject=False):
        '''
         This function instantiate a new instance of the environment for the agent to interact with.
         All the novelty is injected here.
         ### I/P: it takes the env_ID and the novelty arguemtns list as input.
         ### O/P: it returns an instance of the environment.
        '''
        env = gym.make(env_id)
        env.render()
        env.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
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
    
    def run_motion_planner(self, env, action):
        # Instantiation of the AStar Planner with the 
        # ox = 
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        sx = env.agent_location[1]
        sy = env.agent_location[0]
        so = env.agent_facing_str
        print ("agent is at {}, {} and facing {}".format(sy, sx, so))
        binary_map = copy.deepcopy(env.map)
        binary_map[binary_map > 0] = 1
        grid_size = 1.0
        robot_radius = 0.9
        # obstacle positions
        ox, oy = [], []
        for r in range(len(binary_map[0])):
            for c in range(len(binary_map[1])):
                if binary_map[r][c] == 1:
                    ox.append(c)
                    oy.append(r)
        astar_planner = AStarPlanner(ox, oy, grid_size, robot_radius)
        astar_operator = AStarOperator(name = None, goal_type=None, effect_set=None)

        loc2 = action.split(" ")[-1] # last value of the approach action gives the location to go to
        print ("location to go to = {}".format(loc2))
        gx, gy = sx, sy

        if loc2 in env.items:
            locs = np.asarray((np.where(env.map == env.items_id[loc2]))).T
            gx, gy = locs[0][1], locs[0][0]
        relcoord = np.random.randint(4)
        gx_ = gx
        gy_ = gy
        if relcoord == 0:
            gx_ = gx + 1
            ro = 'WEST'
        elif relcoord == 1:
            gx_ = gx - 1
            ro = 'EAST'
        elif relcoord == 2:
            gy_ = gy + 1
            ro = 'NORTH'
        elif relcoord == 3:
            gy_ = gy - 1
            ro = 'SOUTH'

        rxs, rys = astar_planner.planning(sx, sy, gx_, gy_)
        # print("Goal location: {} {}".format(gx_, gy_) )
        # print ("rxs and rys generated from the plan = {} {}".format(rxs, rys))
        sx, sy, plan = astar_operator.generateActionsFromPlan(sx, sy, so, rxs, rys, ro)
        return plan
        ## rx and ry have all the x, y points to use and         
        # a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    
    def inject_novelty(self, novelty_name):

        env_id = 'NovelGridworld-Pogostick-v1'
        env = gym.make(env_id)

        novelty_arg1 = self.novelty_name
        novelty_arg2 = ''
        difficulty = 'medium'

        env = inject_novelty(env, novelty_name, difficulty, novelty_arg1, novelty_arg2)
        return env
    
    def run_trials():
        pass

    def generate_pddls(self, env):
        self.pddl_dir = "PDDL"
        os.makedirs(self.pddl_dir, exist_ok = True)
        generate_prob_pddl(self.pddl_dir, env)
    '''
    This function generates the PDDLs from the current environment instance
    ### I/P environment object
    ### O/P returns pddl names if PDDL generated successfully, else returns false.
    '''
    def execute_plan(self, env, plan, obs):
        '''
        This function executes the plan on the domain step by step
        ### I/P: environment instance and sequence of actions step by step
        ### O/P: SUCCESS/ FAIL with the failed action
        '''
        rew_eps = 0
        count = 0
        env.render()
        matching = [s for s in plan if "approach" in s]
        print ("matching = {}".format(matching))
        i = 0
        while (i < len(plan)):
            print("Executing plan_step: ", plan[i])
            sub_plan = []
            if plan[i] in self.failed_action_set and 'approach' not in plan[i]:#switch to rl
                obs, reward, done, info = env.step(env.actions_id[plan[i]])
                self.executed_learned_policy = True # weird Hack.
                print ("Info = {}".format(info))
                if info['result']==False:
                    print("\n Using the learned policy")
                    self.executed_learned_policy = self.call_learner(failed_action = plan[i], env = env)
                if not self.executed_learned_policy:
                    return False, None
                else:
                    i+=1
            elif plan[i] in self.failed_action_set and 'approach' in plan[i]:
                sub_plan = self.run_motion_planner(env, plan[i])
                print ("subplan = ",sub_plan)
                self.executed_learned_policy = True
                # now execute the sub-plan
                for j in range (len(sub_plan)):
                    env.render()
                    obs, reward, done, info = env.step(env.actions_id[sub_plan[j]])
                    if info['result']==False:
                        print("\n Using the learned policy")
                        self.executed_learned_policy = self.call_learner(failed_action = plan[i], env = env)
                        if self.executed_learned_policy:
                            break
                if not self.executed_learned_policy:
                    return False, None
                else:
                    i+=1
            elif "approach" in plan[i] and plan[i] not in self.failed_action_set:
                # call the motion planner here to generate the lower level actions
                sub_plan = self.run_motion_planner(env, plan[i])
                print ("subplan = ",sub_plan)
                i+=1
                # now execute the sub-plan
                for j in range (len(sub_plan)):
                    env.render()
                    obs, reward, done, info = env.step(env.actions_id[sub_plan[j]])
                    # print ("Info = {}".format(info))
                    if info['result']==False:
                        self.failed_action_set[plan[i]] = None
                        return False, plan[i]
                    env.render()
                    rew_eps += reward
                    count += 1
                    if done:
                        if env.inventory_items_quantity[env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                            return True, None
            # go back to the planner's normal plan
            elif "approach" not in plan[i] and plan[i] not in self.failed_action_set:
                print ("Executing {} action from main plan in the environment".format(env.actions_id[plan[i]]))
                env.render()
                obs, reward, done, info = env.step(env.actions_id[plan[i]])
                print ("Info = {}".format(info))
                if info['result']==False:
                    self.failed_action_set[plan[i]] = None
                    return False, plan[i]
                rew_eps += reward
                count += 1
                if done:
                    if env.inventory_items_quantity[env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                        return True, None
                i+=1
        return False, None

    def evaluate_policy():
        pass
if __name__ == '__main__':
    brain1 = Brain()
    brain1.run_brain()






