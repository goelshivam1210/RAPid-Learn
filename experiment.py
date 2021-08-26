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
EPS_TO_EVAL = 5

class Experiment:
    def __init__(self, args):
        self.hashid = uuid.uuid4().hex
        if args['learner'] == 'smart-exploration':
            self.guided_action = True
            self.guided_policy = True
        else:
            self.guided_action = False
            self.guided_policy = False

        header_train = ['Trial_no', 'Epsilon','Rho','Timesteps','Reward','Done']
        header_test = ['Episode_no','Trial_no','Timesteps','Reward','Done']
        self.generate_results(header_train,"train")
        self.generate_results(header_test,"test")


        env = gym.make(ENV_ID)
        env.reset()
        brain1 = Brain(novelty_name = args['novelty_name'], render = args['render'])
        # run the pre novelty trials 
        for pre_novelty_trial in range(args['trials_pre_novelty']):
            obs = env.reset()
            env.render() 
            brain1.generate_pddls(env)
            plan, game_action_set = brain1.call_planner("domain", "problem", env) # get a plan            
            result, failed_action, step_count = brain1.execute_plan(env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if result == True:
                self.save_results([pre_novelty_trial, 0, 0, step_count,1000-step_count, 1],"train")
                self.save_results([0, pre_novelty_trial, step_count,1000-step_count, 1],"test")
            else:
                self.save_results([pre_novelty_trial, 0, 0, step_count,-step_count, 0],"train")
                self.save_results([0, pre_novelty_trial, step_count,1000-step_count, 1],"test")

        env_pre_items_quantity = copy.deepcopy(env.items_quantity)
        env_pre_actions = copy.deepcopy(env.actions_id)
        # inject novelty
        self.novelty_name = args['novelty_name']
        env = brain1.inject_novelty(novelty_name = self.novelty_name, env = env)
        env.reset() # this is the key
        self.new_item_in_world = None
        self.actions_bump_up = {}
        # get environment instances after novelty injection
        env_post_items_quantity = copy.deepcopy(env.items_quantity)
        # print("Post items quant: ", env.items_quantity)
        env_post_actions = copy.deepcopy(env.actions_id)

        if len(env_post_items_quantity.keys() - env_pre_items_quantity.keys()) > 0:
            # we need to bump up the movement actions probabilities and set flag for new item to True
            self.new_item_in_world = env_post_items_quantity.keys() - env_pre_items_quantity.keys() # This is a dictionary
            self.actions_bump_up.update({'Forward':env_post_actions['Forward']}) # add all the movement actions
            self.actions_bump_up.update({'Left':env_post_actions['Left']})
            self.actions_bump_up.update({'Right':env_post_actions['Right']})
            
        for action in env_post_actions.keys() - env_pre_actions.keys(): # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})       
        # print ("new_item_in_the_world = ", self.new_item_in_world)
        brain1.generate_pddls(env, self.new_item_in_world) # update the domain file      

        # now we run post novelty trials
        for post_novelty_trial in range(args['trials_post_learning']):
            obs = env.reset() 
            # brain1.generate_pddls(env, self.new_item_in_world)
            plan, game_action_set = brain1.call_planner("domain", "problem", env) # get a plan
            # print("game action set aftert the planner = {}".format(game_action_set))
            result, failed_action, step_count = brain1.execute_plan(env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if not result and failed_action is not None: # cases when the plan failed for the first time and the agent needs to learn a new action using RL
                # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
                self.save_results([1, 0, 0, step_count, 0-step_count, 0],"train")
                self.learned, data, data_eval = brain1.call_learner(failed_action=failed_action, actions_bump_up=self.actions_bump_up, new_item_in_the_world=self.new_item_in_world, env=env,transfer = args['transfer'], guided_action= self.guided_action, guided_policy=self.guided_policy)
                if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
                    # data_3 = data[3][-1]
                    for i in range(len(data[0])):
                        self.save_results([2+i, data[3][i], data[4][i], data[2][i], data[0][i], data[1][i]], "train")
                    # data_3_eval = data[3][-1]
                    for j in range(len(data_eval[0])):
                        self.save_results([data_eval[3][i],i%EPS_TO_EVAL, data_eval[2][i], data_eval[0][i], data_eval[1][i]], "test")
                    continue

            if not result and failed_action is None: # The agent used the learned policy and yet was unable to solve
                self.save_results([post_novelty_trial, 0,0, step_count, 0-step_count, 0], "train")
                self.save_results([data[3][i]+1,post_novelty_trial, step_count, 0-step_count, 0], "test")
                continue
            if result:
                self.save_results([post_novelty_trial, 0,0, step_count, 1000-step_count, 1],"train")
                self.save_results([data_eval[3][j]+1, post_novelty_trial, step_count, 1000-step_count, 1],"test")

    def save_results (self, data, tag):
        os.makedirs("data" + os.sep + args['novelty_name'] + "_" + args['learner'] + "_" + self.hashid, exist_ok=True)
        db_file_name = "data" + os.sep + args['novelty_name'] + "_" + args['learner'] + "_" + self.hashid + os.sep + str(tag) + "results.csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)
 
    def generate_results(self, headers, tag):
        os.makedirs("data" + os.sep + args['novelty_name'] + "_" + args['learner'] + "_" + self.hashid, exist_ok=True)
        db_file_name = "data" + os.sep + args['novelty_name'] + "_" + args['learner'] + "_" + self.hashid + os.sep + str(tag) + "results.csv"
        # db_file_name = "data" + os.sep+str(args['novelty_name'])+args['learner']+self.hashid+ os.sep+str(tag)+"results.csv"
        with open(db_file_name, 'a') as f: # append to the file created
            writer = csv.writer(f)
            writer.writerow(headers)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # print("wjerlew")
    ap.add_argument("-N", "--novelty_name", default='axetobreakhard', help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy", type=str)
    ap.add_argument("-TP", "--trials_pre_novelty", default= 1, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN","--trials_post_learning", default = 5, help="Number of trials post recovering from novelty", type = int)
    ap.add_argument("-P", "--print_every", default= 200, help="Number of epsiodes you want to print the results", type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, smart-exploration", type=str)
    ap.add_argument("-T", "--transfer", default=False, type=bool)
    ap.add_argument("-R", "--render", default=False, type=bool)
    args = vars(ap.parse_args())
    experiment1 = Experiment(args)