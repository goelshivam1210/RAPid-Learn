import time
from datetime import datetime
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
import logging

from stable_baselines3.common.callbacks import CheckpointCallback

from brain import Brain
from baselines.wrappers import *
from params import EPS_TO_EVAL

ENV_ID = 'NovelGridworld-Pogostick-v1'  # always remains the same.

class Experiment:
    DATA_DIR = 'data'

    def __init__(self, args, header_train, header_test, extra_run_ids=''):
        self.hashid = uuid.uuid4().hex
        self.experiment_id = extra_run_ids + "-" + self.hashid
        self.results_dir = self._get_results_dir()
        self.trials_pre_novelty = args['trials_pre_novelty']
        self.trials_post_learning = args['trials_post_learning']
        self.novelty_name = args['novelty_name']
        self.render = args['render']

        self.env = gym.make(ENV_ID)

    def _get_results_dir(self):
        return Experiment.DATA_DIR + os.sep + self.experiment_id

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
    HEADER_TEST = ['Mode', 'Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done'] # Mode = 0: Planning, 1 = Learning, 2= Recovery

    def __init__(self, args):
        super(RapidExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                              "_" + args['novelty_name'] + "_"+ args['learner'])
        os.makedirs(self.results_dir, exist_ok=True)

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
                self.write_row_to_results([0, 0, pre_novelty_trial, step_count, 1000 - step_count, 1], "test")
            else:
                self.write_row_to_results([pre_novelty_trial, 0, 0, step_count, -step_count, 0], "train")
                self.write_row_to_results([0, 0, pre_novelty_trial, step_count,  - step_count, 0], "test")

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
                            print("item accidentally obtained in inv.. resetting..")
                            # time.sleep(3)
                            break
                if flag_to_check_new_item_in_inv == True:
                    brain1.failed_action_set = {}
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
                    # self.write_row_to_results([1, data_eval[3], data_eval[2], data_eval[0], data_eval[1]], "test")
                    for j in range(len(data_eval[0])):
                        self.write_row_to_results(
                            [1, data_eval[3][j], EPS_TO_EVAL, data_eval[2][j], data_eval[0][j], data_eval[1][j]],
                            "test")
                    continue 

            if not result and failed_action is None:  # The agent used the learned policy and yet was unable to solve
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 0 - step_count, 0], "train")
                self.write_row_to_results([2, data_eval[3][j] + 1, post_novelty_trial, step_count, 0 - step_count, 0], "test")
                continue
            if result:
                self.write_row_to_results([post_novelty_trial, 0, 0, step_count, 1000 - step_count, 1], "train")
                self.write_row_to_results([2, data_eval[3][j] + 1, post_novelty_trial, step_count, 1000 - step_count, 1], "test")


class BaselineExperiment(Experiment):
    HEADER_TRAIN = ['episode', 'timesteps', 'reward', 'success']
    HEADER_TEST = ['trial', 'episode', 'timesteps', 'reward', 'success']
    MAX_TIMESTEPS_PER_EPISODE = 500
    SAVED_MODEL_NAME = 'model'

    def __init__(self, args):
        self.TRAIN_EPISODES = args["train_episodes"]
        self.load_model = args["load_model"]
        self.reward_shaping = args["reward_shaping"]
        self.algorithm = args["algorithm"]

        super(BaselineExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                                 f"{to_datestring(time.time())}-baseline-{self.algorithm}-{self.TRAIN_EPISODES}episodes-"
                                                 f"{'rewardshapingon' if self.reward_shaping else 'rewardshapingoff'}")

        # Import these here so you can run RAPID experiments without having to have stable baselines installed
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3 import PPO
        set_random_seed(42, using_cuda=True)

        self.env = StatePlaceholderWrapper(self.env, n_placeholders_inventory=2, n_placeholders_lidar=2)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=3)

        if self.load_model:
            print(f"Attempting to load pretrained model {self.load_model}.")
            self.experiment_id = self.load_model
            self.model = PPO.load(
                self._get_results_dir() + os.sep + BaselineExperiment.SAVED_MODEL_NAME)
        else:
            os.makedirs(self._get_results_dir(), exist_ok=True)
            self.model = PPO("MlpPolicy", self.env, verbose=0)

        # Environment wrappers
        # (The order of these seems to be important)
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        # self.env = RecordEpisodeStatsWrapper(self.env)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            print("Reward shaping: ON")
            self.env = RewardShaping(self.env)
        self.env = Monitor(self.env, self._get_results_dir() + os.sep + to_datestring(time.time()) + "-monitor.csv",
                           allow_early_resets=True, info_keywords=('success', 'mode'))
        check_env(self.env, warn=True)

        # This is to use the env with all the wrappers for the model.
        self.model.set_env(self.env)

    def run(self):
        if not self.load_model:
            print("No pretrained model supplied, training from scratch.")
            self.train()
        else:
            print(f"Skipping training because pretrained model {self.load_model} was supplied.")

        self.evaluate()
        self.env.close()

    def train(self):
        print(f"Training model for {self.TRAIN_EPISODES} episodes")
        self.env.metadata['mode'] = 'train'
        checkpoint_callback = CheckpointCallback(save_freq=self.MAX_TIMESTEPS_PER_EPISODE * 500, save_path=self.results_dir + os.sep + 'checkpoints',
                                                 name_prefix=BaselineExperiment.SAVED_MODEL_NAME)
        self.model.learn(total_timesteps=self.TRAIN_EPISODES * self.MAX_TIMESTEPS_PER_EPISODE,
                         callback=checkpoint_callback)
        self.model.save(self.results_dir + os.sep + BaselineExperiment.SAVED_MODEL_NAME)

    def evaluate(self):
        print("Evaluating model.")
        self.env.metadata['mode'] = 'test-prenovelty'
        from stable_baselines3.common.evaluation import evaluate_policy
        obs = self.env.reset()
        done = False
        evaluate_policy(self.model, self.env, self.trials_pre_novelty, deterministic=False, render=self.render)


def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
    return datetime.utcfromtimestamp(unixtime).strftime(format)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("-N", "--novelty_name", default='axetobreakeasy',
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("-TP", "--trials_pre_novelty", default=1, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN", "--trials_post_learning", default=5, help="Number of trials post recovering from novelty",
                    type=int)
    ap.add_argument("-P", "--print_every", default=200, help="Number of epsiodes you want to print the results",
                    type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, smart-exploration", type=str)
    ap.add_argument("-T", "--transfer", default=None, type=str)
    ap.add_argument("-R", "--render", default=False, type=bool)
    ap.add_argument("--load_model", default=False, type=str)
    ap.add_argument("--train_episodes", default=100, type=int)
    ap.add_argument("--reward_shaping", default=False, type=bool)
    ap.add_argument("--algorithm", default="PPO", type=str)

    args = vars(ap.parse_args())
    if args['experiment'] == 'baseline':
        experiment1 = BaselineExperiment(args)
    else:
        experiment1 = RapidExperiment(args)
    experiment1.run()
