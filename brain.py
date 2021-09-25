'''
# author: Shivam Goel
# email: shivam.goel@tufts.edu

# This file is the Brain of RAPidL.
# It is the main file which talks to the planner, learner and the game instance.

Important References

'''
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

# from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
# from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
from generate_pddl import *
from learner_v2 import Learner
# from operator_generalization import *

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
    
    def __init__(self, novelty_name=None, render=False):
        # self.novelty_name = 'rubber_tree'
        self.learner = None
        self.failed_action_set = {}
        self.novelty_name = novelty_name
        self.completed_trials = 0
        self.learned_policies_dict = {} # store failed action:learner_instance object
        self.render = render
        self.actions_bump_up = {}
        self.plan = []
        # self.actions_to_remove = [] # these actions are the ones that the planner doesnt need. they are updated by the learner.

    def run_brain(self, env_id=None, novelty_name=None, num_trails_pre_novelty=None, num_trials_post_learning=None, inject_multiple = False):
        '''        
            This is the driving function of this class.
            Call the environment and run the environment for x number of trials.
        '''
        #### 
        # THIS IS DEPRECATED
        ###
        env_id = 'NovelGridworld-Pogostick-v1' # hardcoded for now. will add the argparser later.
        env = self.instantiate_env(env_id) # make a new instance of the environment.
        # self.novelty_name = novelty_name # to be used in future.
        # get environment instances before injecting novelty
        env_pre_items_quantity = copy.deepcopy(env.items_quantity)
        env_pre_actions = copy.deepcopy(env.actions_id)
        if novelty_name is None:
            self.novelty_name = 'axetobreakhard' #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree
        else:
            self.novelty_name = novelty_name

        env = self.inject_novelty(novelty_name = self.novelty_name)
        env.reset()
        self.new_item_in_world = None
        # get environment instances after novelty injection
        env_post_items_quantity = copy.deepcopy(env.items_quantity)
        env_post_actions = copy.deepcopy(env.actions_id)

        if len(env_post_items_quantity.keys() - env_pre_items_quantity.keys()) > 0:
            # we need to bump up the movement actions probabilities and set flag for new item to True
            self.new_item_in_world = env_post_items_quantity.keys() - env_pre_items_quantity.keys() # This is a dictionary
            self.actions_bump_up.update({'Forward':env_post_actions['Forward']}) # add all the movement actions
            self.actions_bump_up.update({'Left':env_post_actions['Left']})
            self.actions_bump_up.update({'Right':env_post_actions['Right']})
        # print("actions to bump up: ", self.actions_bump_up)
        # print("New item: ", self.new_item_in_world)

        for action in env_post_actions.keys() - env_pre_actions.keys(): # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})             
        # print("Observation space: ", env.observation_space.shape[0])

        obs = env.reset() 
        self.generate_pddls(env)
        self.domain_file_name = "domain"
        # print ("Actions ID from the env = {} ".format(env.actions_id))
        plan, game_action_set = self.call_planner(self.domain_file_name, "problem", env) # get a plan
        # print("game action set aftert the planner = {}".format(game_action_set))
        result, failed_action = self.execute_plan(env, game_action_set, obs)
        # print ("result = {}  Failed Action =  {}".format(result, failed_action))
        
        if not result and failed_action is not None: # cases when the plan failed for the first time and the agent needs to learn a new action using RL
            # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
            self.learned = self.call_learner(failed_action=failed_action, env=env)
            
            if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
                # print ("Agent succesfully learned a new action in the form of policy. Now resetting to test.")
                self.run_brain()
            
        if not result and failed_action is None: # The agent used the learned policy and yet was unable to solve
            # print ("Failed to execute policy successfully. Now resetting for another trial.")
            self.run_brain()
 
        if result:
            pass
            # print("succesfully completed the task without any hassle!")
            # print("Needed to transfer: ", self.completed_trails)

    def call_learner(self, failed_action, plan, actions_bump_up = None, new_item_in_the_world = None, env=None, transfer = None, guided_action = False, guided_policy = False, exploration_mode = 'uniform'):
        # This function instantiates a RL learner to start finding interesting states to send 
        # to the planner

        if env is None:
            env_id = 'NovelGridworld-Pogostick-v1'  # hardcoded for now. will add the argparser later.
            env = self.instantiate_env(env_id, None, False)  # make a new instance of the environment.
            obs = env.reset()
        if actions_bump_up is not None:
            self.actions_bump_up = actions_bump_up
        else:
            self.actions_bump_up = {}
        if new_item_in_the_world is not None:
            self.new_item_in_world = new_item_in_the_world
        else:
            self.new_item_in_world = None
            
        self.current_failed_action = failed_action
        if failed_action not in self.learned_policies_dict:
            self.actions_bump_up.update({failed_action:env.actions_id[failed_action]}) # add failed actions in the list
            self.learner = Learner(failed_action = failed_action, env = env, plan = plan, actions_bump_up= self.actions_bump_up,new_item_in_the_world= self.new_item_in_world, guided_policy=guided_policy, guided_action=guided_action, exploration_mode = exploration_mode)
            self.learned_policies_dict[failed_action] = self.learner # save the learner instance object to the learned poliocies dict.
            learned, data, data_eval = self.learner.learn_policy(self.novelty_name, self.learned_policies_dict, self.failed_action_set, transfer=transfer) # learn to reach the goal state, if reached save the learned policy using novelty_name
            return learned, data, data_eval
        else:
            played, is_reward_met = self.learned_policies_dict[failed_action].play_learned_policy(env, novelty_name=self.novelty_name, operator_name=failed_action) # returns whether the policy was successfully played or not
            if is_reward_met:
                for action_to_remove in self.learned_policies_dict[failed_action].reward_funcs.actions_that_generate_objects_required: # now we remove the actions for those objects
                    try:
                        while True:
                            self.game_action_set.remove(action_to_remove)
                    except ValueError:
                        pass                                        
            return played

    def call_planner(self, domain, problem, env):
        '''
            Given a domain and a problem file
            This function return the ffmetric Planner output.
            In the action format
        '''
    
        run_script = "Metric-FF-v2.1/./ff -o "+self.pddl_dir+os.sep+domain+".pddl -f "+self.pddl_dir+os.sep+problem+".pddl -s 0"
        output = subprocess.getoutput(run_script)
        plan, self.game_action_set = self._output_to_plan(output, env)
        return plan, self.game_action_set

    def _output_to_plan(self, output, env):
        '''
        Helper function to perform regex on the output from the planner.
        ### I/P: Takes in the ffmetric output and
        ### O/P: converts it to a action sequence list.
        '''

        ff_plan = re.findall(r"\d+?: (.+)", output.lower()) # matches the string to find the plan bit from the ffmetric output.
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
        
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)

        for i in range(len(game_action_set)):
            if game_action_set[i].split(" ")[0] != "approach" and game_action_set[i].split("_")[0] != "Select":
                game_action_set[i] = action_map[game_action_set[i]]
        for i in range(len(game_action_set)):
            if game_action_set[i] in action_map:
                game_action_set[i] = env.actions_id[game_action_set[i]]
        return action_set, game_action_set

    def instantiate_env(self, env_id, novelty_family=None, inject=False):
        '''
         This function instantiate a new instance of the environment for the agent to interact with.
         All the novelty is injected here.
         ### I/P: it takes the env_ID and the novelty arguemtns list as input.
         ### O/P: it returns an instance of the environment.
        '''
        env = gym.make(env_id)
        if self.render:
            env.render()
        env.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
        # env.reward_done = 1000
        # env.reward_intermediate = 50
        if inject:
            env = inject_novelty(env, novelty_family[0], novelty_family[1], novelty_family[2], novelty_family[3]) 
        return env
    
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
        # print ("agent is at {}, {} and facing {}".format(sy, sx, so))
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
        # print ("location to go to = {}".format(loc2))
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
        sx, sy, plan = astar_operator.generateActionsFromPlan(sx, sy, so, rxs, rys, ro)
        return plan
        ## rx and ry have all the x, y points to use and         
    
    def inject_novelty(self, novelty_name, env=None):

        if env is None:
            env_id = 'NovelGridworld-Pogostick-v1'
            env = gym.make(env_id)
        else:
            env = env

        env = inject_novelty(env, novelty_name)
        return env

    def generate_pddls(self, env, new_item = None):
        self.pddl_dir = "PDDL"
        os.makedirs(self.pddl_dir, exist_ok = True)
        generate_prob_pddl(self.pddl_dir, env)
        if new_item is not None:
            print ("new item adding to the domain file = ", new_item)
            generate_domain_pddl(self.pddl_dir, env, new_item)
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
        self.plan = plan
        if self.render:
            env.render()

        i = 0
        while (i < len(self.plan)):
            # print("Executing plan_step: ", plan[i])
            sub_plan = []
            if self.plan[i] in self.failed_action_set and 'approach' not in self.plan[i]:#switch to rl
                obs, reward, done, info = env.step(env.actions_id[self.plan[i]])
                self.executed_learned_policy = True # weird Hack.
                # print ("Info = {}".format(info))
                if info['result']==False:
                    # print("\n Using the learned policy")
                    self.executed_learned_policy = self.call_learner(failed_action = self.plan[i], env = env, plan=self.plan)
                    
                if not self.executed_learned_policy:
                    return False, None, env.step_count
                else:
                    i+=1
                    if (i >= len(self.plan)):
                        return True, None, env.step_count
                    else:
                        pass
            elif self.plan[i] in self.failed_action_set and 'approach' in self.plan[i]:
                sub_plan = self.run_motion_planner(env, self.plan[i])
                # print ("subplan = ",sub_plan)
                self.executed_learned_policy = True
                # now execute the sub-plan
                for j in range (len(sub_plan)):
                    if self.render:
                        env.render()
                    obs, reward, done, info = env.step(env.actions_id[sub_plan[j]])
                    if info['result']==False:
                        print("\n Using the learned policy")
                        self.executed_learned_policy = self.call_learner(failed_action = self.plan[i], env = env, plan=self.plan)
                        if self.executed_learned_policy:
                            break
                if not self.executed_learned_policy:
                    return False, None, env.step_count
                else:
                    i+=1
            elif "approach" in self.plan[i] and self.plan[i] not in self.failed_action_set:
                # call the motion planner here to generate the lower level actions
                sub_plan = self.run_motion_planner(env, self.plan[i])
                # print ("subplan = ",sub_plan)
                i+=1
                # now execute the sub-plan
                for j in range (len(sub_plan)):
                    if self.render:
                        env.render()
                    obs, reward, done, info = env.step(env.actions_id[sub_plan[j]])
                    # print ("Info = {}".format(info))
                    if info['result']==False:
                        self.failed_action_set[self.plan[i]] = None
                        return False, self.plan[i], env.step_count
                    if self.render:
                        env.render()
                    rew_eps += reward
                    count += 1
                    if done:
                        if env.inventory_items_quantity[env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                            return True, None, env.step_count
            # go back to the planner's normal plan
            elif "approach" not in self.plan[i] and self.plan[i] not in self.failed_action_set:
                # print ("Executing {} action from main plan in the environment".format(env.actions_id[plan[i]]))
                if self.render:
                    env.render()
                obs, reward, done, info = env.step(env.actions_id[self.plan[i]])
                # print ("Info = {}".format(info))
                if info['result']==False:
                    self.failed_action_set[self.plan[i]] = None
                    return False, self.plan[i], env.step_count
                rew_eps += reward
                count += 1
                if done:
                    if env.inventory_items_quantity[env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                        return True, None, env.step_count
                i+=1
        
        return False, None, env.step_count


if __name__ == '__main__':
    brain1 = Brain()
    brain1.run_brain(novelty_name= 'axetobreakeasy') # Inject axetobreakeasy -> Let it learn and perform 1 episode
    brain1.run_brain(novelty_name = 'axetobreakhard')
    # Now inject axetobreakhard in the same instance of the class and then use the same policy






