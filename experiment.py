
# this file runs the experiments
# Instantiate the environment.
# run 20 trials pre novelty.

# inject novelty after 20 trials
# save the learning curves into a CSV
# run 20 trials after learning.
# inject another novelty, if needed.
# repeat.

# update the CSV while saving task success in all the trials.
import time
import math
import sys
import os
import glob
import csv
import argparse
import copy
import uuid


import gym
from brain import Brain

ENV_ID = 'NovelGridworld-Pogostick-v1' # always remains the same.

class Experiment:
    def __init__(self, args):
        self.hashid = uuid.uuid4().hex
        # Make the env
        # from arg take pre novety trials
        # In for loop
            # Dynamically generate problem pddl
            # Get a plan
            # Execute the plan
        # In for loop
            # Inject novelty (multiple if any)
                # Dynamically generate problem pddl
                # Get a plan
                # Execute a plan until failure
                # Go to learner
                # Check if policy for the failed operator already exists
                    # If yes, try to use the policy to accomplish the failed operator
                    # If success, move ahead to the plan and complete the task
                    # If failure, re-learn the imroved (difficult) novelty and save the policy
                # Create a new learner object
                # Learn and save the policy 
        env = gym.make(ENV_ID)
        brain1 = Brain(render = args['render'])
        # run the pre novelty trials 
        for pre_novelty_trial in range(args['trials_pre_novelty']):
            obs = env.reset()
            env.render() 
            brain1.generate_pddls(env)
            plan, game_action_set = brain1.call_planner("domain", "problem", env) # get a plan            
            result, failed_action = brain1.execute_plan(env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if result == True:
                self.save_results([1])
            else:
                self.save_results([0])

        env_pre_items_quantity = copy.deepcopy(env.items_quantity)
        env_pre_actions = copy.deepcopy(env.actions_id)
        # inject novelty
        self.novelty_name = args['novelty_name']
        env = brain1.inject_novelty(novelty_name = self.novelty_name)
        for novelty in args['multiple_novelty']:
            env = brain1.inject_novelty(novelty_name = novelty)
                
        self.new_item_in_world = None
        self.actions_bump_up = {}
        # get environment instances after novelty injection
        env_post_items_quantity = copy.deepcopy(env.items_quantity)
        env_post_actions = copy.deepcopy(env.actions_id)

        if len(env_post_items_quantity.keys() - env_pre_items_quantity.keys()) > 0:
            # we need to bump up the movement actions probabilities and set flag for new item to True
            self.new_item_in_world = env_post_items_quantity.keys() - env_pre_items_quantity.keys() # This is a dictionary
            self.actions_bump_up.update({'Forward':env_post_actions['Forward']}) # add all the movement actions
            self.actions_bump_up.update({'Left':env_post_actions['Left']})
            self.actions_bump_up.update({'Right':env_post_actions['Right']})
            
        for action in env_post_actions.keys() - env_pre_actions.keys(): # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})             

        # now we run post novelty trials
        for post_novelty_trial in range(args['trials_post_learning']):
            obs = env.reset() 
            brain1.generate_pddls(env)
            plan, game_action_set = brain1.call_planner("domain", "problem", env) # get a plan
            # print("game action set aftert the planner = {}".format(game_action_set))
            result, failed_action = brain1.execute_plan(env, game_action_set, obs)
            print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if not result and failed_action is not None: # cases when the plan failed for the first time and the agent needs to learn a new action using RL
                # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
                self.learned = brain1.call_learner(failed_action=failed_action, actions_bump_up= self.actions_bump_up,new_item_in_the_world=self.new_item_in_world, env=env)
                if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
                    # print ("Agent succesfully learned a new action in the form of policy. Now resetting to test.")
                    continue
            if not result and failed_action is None: # The agent used the learned policy and yet was unable to solve
                print ("Trial - {}, Done - {}".format(post_novelty_trial, 0))
                self.save_results([0])
                continue
            if result:
                self.save_results([1])
                print ("Trial - {}, Done - {}".format(post_novelty_trial, 1))
                # print("succesfully completed the task without any hassle!")

    def save_results (self, data):
        os.makedirs("data" + os.sep + args['novelty_name']+args['learner']+self.hashid, exist_ok=True)
        # if tag == 'pre_novelty_trials':
        db_file_name = "data" + os.sep+str(args['novelty_name'])+args['learner']+self.hashid+ os.sep+"results.csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # print("wjerlew")
    ap.add_argument("-N", "--novelty_name", default='axetobreakeasy', help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree", type=str)
    ap.add_argument("-M", "--multiple_novelty", default = {},  help="Multiple Novelties dictionary input", type = dict)
    ap.add_argument("-TP", "--trials_pre_novelty", default= 1, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN","--trials_post_learning", default = 10, help="Number of trials post recovering from novelty", type = int)
    ap.add_argument("-P", "--print_every", default= 200, help="Number of epsiodes you want to print the results", type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, smart-exploration", type=str)
    ap.add_argument("-R", "--render", default=False, type=bool)

    args = vars(ap.parse_args())
    print (args['render'])
    experiment1 = Experiment(args)