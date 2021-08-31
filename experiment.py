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
import yaml
import gym

from brain import Brain
from baselines.wrappers import *
from params import EPS_TO_EVAL

ENV_ID = 'NovelGridworld-Pogostick-v1'  # always remains the same.

class Experiment(ABC):
    DATA_DIR = 'data'

    def __init__(self, args, header_train, header_test, extra_run_ids=''):
        self.hashid = uuid.uuid4().hex
        self.results_dir = Experiment.DATA_DIR + os.sep + extra_run_ids + self.hashid
        self.trials_pre_novelty = args['trials_pre_novelty']
        self.trials_post_learning = args['trials_post_learning']
        self.novelty_name = args['novelty_name']
        self.render = args['render']

        os.makedirs(self.results_dir, exist_ok=True)

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
        if args['learner'] == 'smart-exploration':
            self.guided_action = True
            self.guided_policy = True
        else:
            self.guided_action = False
            self.guided_policy = False

        self.write_row_to_results(self.HEADER_TRAIN, "train")
        self.write_row_to_results(self.HEADER_TEST, "test")

    def run(self):
        self.env.reset()
        brain1 = Brain(novelty_name=self.novelty_name, render=self.render)
        # run the pre novelty trials
        for pre_novelty_trial in range(self.trials_pre_novelty):
            obs = self.env.reset()
            if self.render:
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
        self.env = brain1.inject_novelty(novelty_name=self.novelty_name, env=self.env)
        self.env.reset()  # this is the key
        self.new_item_in_world = None
        self.actions_bump_up = {}
        # get environment instances after novelty injection
        env_post_items_quantity = copy.deepcopy(self.env.items_quantity)
        # print("Post items quant: ", env.items_quantity)
        env_post_actions = copy.deepcopy(self.env.actions_id)

        if len(env_post_items_quantity.keys() - env_pre_items_quantity.keys()) > 0:
            # we need to bump up the movement actions probabilities and set flag for new item to True
            self.new_item_in_world = env_post_items_quantity.keys() - env_pre_items_quantity.keys()  # This is a set
            self.actions_bump_up.update({'Forward': env_post_actions['Forward']})  # add all the movement actions
            self.actions_bump_up.update({'Left': env_post_actions['Left']})
            self.actions_bump_up.update({'Right': env_post_actions['Right']})

        for action in env_post_actions.keys() - env_pre_actions.keys():  # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})
            # print ("new_item_in_the_world = ", self.new_item_in_world)
        brain1.generate_pddls(self.env, self.new_item_in_world)  # update the domain file

        # now we run post novelty trials
        for post_novelty_trial in range(self.trials_post_learning):
            obs = self.env.reset()
            # brain1.generate_pddls(env, self.new_item_in_world)
            plan, game_action_set = brain1.call_planner("domain", "problem", self.env)  # get a plan
            # print("game action set aftert the planner = {}".format(game_action_set))
            result, failed_action, step_count = brain1.execute_plan(self.env, game_action_set, obs)
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if not result and failed_action is not None:  # cases when the plan failed for the first time and the agent needs to learn a new action using RL
                # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
                flag_to_check_new_item_in_inv = False
                if self.new_item_in_world is not None:
                    for new_item in self.new_item_in_world:
                        if self.env.inventory_items_quantity[new_item] > 0:
                            flag_to_check_new_item_in_inv = True
                            self.trials_post_learning+=1
                            print("item obtained in inv")
                            # time.sleep(3)
                            break
                if flag_to_check_new_item_in_inv == True:
                    print("trials post learning: ", self.trials_post_learning)
                    continue
                # if self.env.inventory_items_quanity[i] for i in self.new_item_in_world
                self.write_row_to_results([1, 0, 0, step_count, 0 - step_count, 0], "train")
                # print ("In brain self.guided_action = {}  self.guided_policy = {}".format(self.guided_action, self.guided_policy))

                self.learned, data, data_eval = brain1.call_learner(failed_action=failed_action,
                                                                    actions_bump_up=self.actions_bump_up,
                                                                    new_item_in_the_world=self.new_item_in_world,
                                                                    env=self.env, transfer=args['transfer'],
                                                                    plan = game_action_set,
                                                                    guided_action=self.guided_action,
                                                                    guided_policy=self.guided_policy)
                if self.learned:  # when the agent successfully learns a new action, it should now test it to re-run the environment.
                    # data_3 = data[3][-1]
                    for i in range(len(data[0])):
                        self.write_row_to_results([2 + i, data[3][i], data[4][i], data[2][i], data[0][i], data[1][i]], "train")
                    # data_3_eval = data[3][-1]
                    for j in range(len(data_eval[0])):
                        self.write_row_to_results(
                            [data_eval[3][j], j % EPS_TO_EVAL, data_eval[2][j], data_eval[0][j], data_eval[1][j]],
                            "test")
                    continue 

            if not result and failed_action is None:  # The agent used the learned policy and yet was unable to solve
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 0 - step_count, 0], "train")
                self.write_row_to_results([data_eval[3][j] + 1, post_novelty_trial, step_count, 0 - step_count, 0], "test")
                continue
            if result:
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 1000 - step_count, 1], "train")
                self.write_row_to_results([data_eval[3][j] + 1, post_novelty_trial, step_count, 1000 - step_count, 1], "test")


class BaselineExperiment(Experiment):
    HEADER_TRAIN = ['episode', 'timesteps', 'reward', 'success']
    HEADER_TEST = ['trial', 'episode', 'timesteps', 'reward', 'success']
    MAX_TIMESTEPS_PER_EPISODE = 500
    TRAIN_EPISODES = 10

    def __init__(self, args):
        super(BaselineExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST, "baseline-")

        # Import these here so you can run RAPID experiments without having to have stable baselines installed
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3 import PPO
        set_random_seed(42, using_cuda=True)

        self.env = gym.make('NovelGridworld-Pogostick-v1')

        # Environment wrappers
        # (The order of these seems to be important)
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        # self.env = RecordEpisodeStatsWrapper(self.env)
        self.env = TrainTestModeWrapper(self.env)
        self.env = Monitor(self.env, self.results_dir + os.sep + "monitor.csv", allow_early_resets=True,
                           info_keywords=('success', 'mode'))
        self.env = RewardShaping(self.env)
        check_env(self.env, warn=True)

        self.env.reward_done = 1000
        self.env.reward_intermediate = 50

        self.model = PPO("MlpPolicy", self.env, verbose=1)
        # self.logger = configure(self.results_dir, ["stdout", "csv"])
        # self.model.set_logger(self.logger)

    def run(self):
        self.train()
        self.evaluate()

    def train(self):
        self.env.metadata['mode'] = 'train'
        self.model.learn(total_timesteps=self.TRAIN_EPISODES * self.MAX_TIMESTEPS_PER_EPISODE)

    def evaluate(self):
        self.env.metadata['mode'] = 'test'

        from stable_baselines3.common.evaluation import evaluate_policy
        obs = self.env.reset()
        done = False
        evaluate_policy(self.model, self.env, self.trials_pre_novelty)

        self.env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--novelty_name", default='axetobreakeasy',
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("-TP", "--trials_pre_novelty", default=3, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN", "--trials_post_learning", default=5, help="Number of trials post recovering from novelty",
                    type=int)
    ap.add_argument("-P", "--print_every", default=200, help="Number of epsiodes you want to print the results",
                    type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, smart-exploration", type=str)
    ap.add_argument("-T", "--transfer", default=None, type=str)
    ap.add_argument("-R", "--render", default=False, type=bool)
    args = vars(ap.parse_args())
    experiment1 = BaselineExperiment(args)
    experiment1.run()
