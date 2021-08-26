import time
import math
import sys
import os
import glob
import csv
import argparse
import copy
import uuid
from abc import abstractmethod, ABC

import gym

from brain import Brain

ENV_ID = 'NovelGridworld-Pogostick-v1'  # always remains the same.
EPS_TO_EVAL = 5


class Experiment(ABC):
    DATA_DIR = 'data'

    def __init__(self, args, header_train, header_test, extra_run_ids):
        self.hashid = uuid.uuid4().hex
        self.results_dir = Experiment.DATA_DIR + os.sep + extra_run_ids + self.hashid
        os.makedirs(self.results_dir, exist_ok=True)

        self.write_row_to_results(header_train, "train")
        self.write_row_to_results(header_test, "test")

        self.env = gym.make(ENV_ID)

    @abstractmethod
    def run(self):
        pass

    def write_row_to_results(self, data, tag):
        db_file_name = self.results_dir + os.sep + str(tag) + "results.csv"
        with open(db_file_name, 'a') as f:  # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)


class RapidExperiment(Experiment):
    HEADER_TRAIN = ['Trial_no', 'Epsilon', 'Rho', 'Timesteps', 'Reward', 'Done']
    HEADER_TEST = ['Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done']

    def __init__(self, args):
        super(RapidExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                              args['novelty_name'] + args['learner'])
        if args['learner']:
            self.guided_action = True
            self.guided_policy = True
        else:
            self.guided_action = False
            self.guided_policy = False

    def run(self):
        brain1 = Brain(novelty_name=args['novelty_name'], render=args['render'])
        # run the pre novelty t`rials
        for pre_novelty_trial in range(args['trials_pre_novelty']):
            obs = self.env.reset()
            self.env.render()
            brain1.generate_pddls(self.env)
            plan, game_action_set = brain1.call_planner("domain", "problem", self.env)  # get a plan
            result, failed_action, step_count = brain1.execute_plan(self.env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if result == True:
                self.write_row_to_results([pre_novelty_trial, 0, 0, step_count, 1000 - step_count, 1], "train")
                self.write_row_to_results([0, pre_novelty_trial, step_count, 1000 - step_count, 1], "test")
            else:
                self.write_row_to_results([pre_novelty_trial, 0, 0, step_count, -step_count, 0], "train")
                self.write_row_to_results([0, pre_novelty_trial, step_count, 1000 - step_count, 1], "test")

        env_pre_items_quantity = copy.deepcopy(self.env.items_quantity)
        env_pre_actions = copy.deepcopy(self.env.actions_id)
        # inject novelty
        self.novelty_name = args['novelty_name']
        self.env = brain1.inject_novelty(novelty_name=self.novelty_name)

        self.new_item_in_world = None
        self.actions_bump_up = {}
        # get environment instances after novelty injection
        env_post_items_quantity = copy.deepcopy(self.env.items_quantity)
        env_post_actions = copy.deepcopy(self.env.actions_id)

        if len(env_post_items_quantity.keys() - env_pre_items_quantity.keys()) > 0:
            # we need to bump up the movement actions probabilities and set flag for new item to True
            self.new_item_in_world = env_post_items_quantity.keys() - env_pre_items_quantity.keys()  # This is a dictionary
            self.actions_bump_up.update({'Forward': env_post_actions['Forward']})  # add all the movement actions
            self.actions_bump_up.update({'Left': env_post_actions['Left']})
            self.actions_bump_up.update({'Right': env_post_actions['Right']})

        for action in env_post_actions.keys() - env_pre_actions.keys():  # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})

            # now we run post novelty trials
        for post_novelty_trial in range(args['trials_post_learning']):
            obs = self.env.reset()
            brain1.generate_pddls(self.env)
            plan, game_action_set = brain1.call_planner("domain", "problem", self.env)  # get a plan
            # print("game action set aftert the planner = {}".format(game_action_set))
            result, failed_action, step_count = brain1.execute_plan(self.env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if not result and failed_action is not None:  # cases when the plan failed for the first time and the agent needs to learn a new action using RL
                # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
                self.write_row_to_results([1, 0, 0, step_count, 0 - step_count, 0], "train")
                self.learned, data, data_eval = brain1.call_learner(failed_action=failed_action,
                                                                    actions_bump_up=self.actions_bump_up,
                                                                    new_item_in_the_world=self.new_item_in_world,
                                                                    env=self.env, transfer=args['transfer'],
                                                                    guided_action=self.guided_action,
                                                                    guided_policy=self.guided_policy)
                if self.learned:  # when the agent successfully learns a new action, it should now test it to re-run the environment.
                    for i in range(len(data[0])):
                        self.write_row_to_results([2 + i, data[3][i], data[4][i], data[2][i], data[0][i], data[1][i]], "train")
                    for i in range(len(data_eval[0])):
                        self.write_row_to_results(
                            [data_eval[3][i], i % EPS_TO_EVAL, data_eval[2][i], data_eval[0][i], data_eval[1][i]],
                            "test")
                    continue

            if not result and failed_action is None:  # The agent used the learned policy and yet was unable to solve
                # print ("Trial - {}, Done - {}".format(post_novelty_trial, 0))
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 0 - step_count, 0], "train")
                self.write_row_to_results([data[3][i] + 1, post_novelty_trial, step_count, 0 - step_count, 0], "test")
                continue
            if result:
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 1000 - step_count, 1], "train")
                self.write_row_to_results([data_eval[3][i] + 1, post_novelty_trial, step_count, 1000 - step_count, 1], "test")
                # print ("Trial - {}, Done - {}".format(post_novelty_trial, 1))
                # print("succesfully completed the task, without any hassle!")


class BaselineExperiment(Experiment):
    HEADER_TRAIN = ['Trial_no', 'Timesteps', 'Reward', 'Done']
    HEADER_TEST = ['Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done']

    def __init__(self, args):
        super(BaselineExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST)

    def run(self):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--novelty_name", default='axetobreakeasy',
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("-TP", "--trials_pre_novelty", default=1, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN", "--trials_post_learning", default=5, help="Number of trials post recovering from novelty",
                    type=int)
    ap.add_argument("-P", "--print_every", default=200, help="Number of epsiodes you want to print the results",
                    type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, smart-exploration", type=str)
    ap.add_argument("-T", "--transfer", default=False, type=bool)
    ap.add_argument("-R", "--render", default=False, type=bool)
    args = vars(ap.parse_args())
    experiment1 = RapidExperiment(args)
    experiment1.run()
