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
import json
import logging

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from params import *

from stable_baselines3 import PPO

from SimpleDQN import SimpleDQN
from baselines.callbacks import CustomEvalCallback

set_random_seed(42, using_cuda=True)

from baselines.util import get_difference_in_obs_action_space
from brain import Brain
from baselines.wrappers import *
from params import *
from gym_novel_gridworlds.novelty_wrappers import inject_novelty

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
    def write_params_to_file(self, data):
        db_file_name = self.results_dir + os.sep + "params.json"
        out_file = open(db_file_name, "w") 
        json.dump(data, out_file, indent = 6) 
        out_file.close() 


class RapidExperiment(Experiment):
    HEADER_TRAIN = ['Trial_no', 'Epsilon', 'Rho', 'Timesteps', 'Reward', 'Done']
    HEADER_TEST = ['Mode', 'Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done'] # Mode = 0: Planning, 1 = Learning, 2= Recovery

    def __init__(self, args):
        super(RapidExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                              "_" + args['novelty_name'] + "_"+ args['learner'] + "_"+ args['exploration_mode'])
        os.makedirs(self.results_dir, exist_ok=True)

        if args['learner'] == 'both':
            self.guided_action = True
            self.guided_policy = True
        elif args['learner'] == 'action_biasing':
            self.guided_action = True
            self.guided_policy = False
        elif args['learner'] == 'guided_policy':
            self.guided_action = False
            self.guided_policy = True
        else:
            self.guided_action = False
            self.guided_policy = False

        self.exploration_mode  = args['exploration_mode']

        self.write_row_to_results(self.HEADER_TRAIN, "train")
        self.write_row_to_results(self.HEADER_TEST, "test")
        data_to_json = {    "MAX_EPSILON": MAX_EPSILON, 
                            "MIN_EPSILON": MIN_EPSILON, 
                            "MAX_RHO": MAX_RHO, 
                            "MIN_RHO": MIN_RHO, 
                            "EXPLORATION_STOP": EXPLORATION_STOP,
                            'EPS_to_Eval': EPS_TO_EVAL,
                            'Eval_Interval': EVAL_INTERVAL,
                            'seed': random_seed,
                            'Reward_check': SCORE_TO_CHECK,
                            'Dones_Check': NO_OF_SUCCESSFUL_DONE,
                    } 
        self.write_params_to_file(data_to_json)

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
        self.env.seed(random_seed)
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
            # self.actions_bump_up.update({'Forward': env_post_actions['Forward']})  # add all the movement actions
            # self.actions_bump_up.update({'Left': env_post_actions['Left']})
            # self.actions_bump_up.update({'Right': env_post_actions['Right']})

        for action in env_post_actions.keys() - env_pre_actions.keys():  # add new actions
            self.actions_bump_up.update({action: env_post_actions[action]})
            # print ("new_item_in_the_world = ", self.new_item_in_world)
        brain1.generate_pddls(self.env, self.new_item_in_world)  # update the domain file

        # now we run post novelty trials
        # for post_novelty_trial in range(self.trials_post_learning):
        post_novelty_trial = 0
        while post_novelty_trial <= self.trials_post_learning:
            obs = self.env.reset()
            # brain1.generate_pddls(env, self.new_item_in_world)
            plan, game_action_set = brain1.call_planner("domain", "problem", self.env)  # get a plan
            # print("game action set aftert the planner = {}".format(game_action_set))
            result, failed_action, step_count = brain1.execute_plan(self.env, game_action_set, obs)
            post_novelty_trial += 1
            # print ("result = {}  Failed Action =  {}".format(result, failed_action))
            if not result and failed_action is not None:  # cases when the plan failed for the first time and the agent needs to learn a new action using RL
                # print ("Instantiating a RL Learner to learn a new action to solve the impasse.")
                flag_to_check_new_item_in_inv = False
                if self.new_item_in_world is not None:
                    for new_item in self.new_item_in_world:
                        if self.env.inventory_items_quantity[new_item] > 0:
                            flag_to_check_new_item_in_inv = True
                            post_novelty_trial=0
                            print("item accidentally obtained in inv.. resetting..")
                            # time.sleep(3)
                            break
                if flag_to_check_new_item_in_inv == True: # we need to reset the failed action set since we are resetting the environment.
                    brain1.failed_action_set = {}
                    continue 
                # if self.env.inventory_items_quanity[i] for i in self.new_item_in_world
                self.write_row_to_results([1, 0, 0, step_count, 0 - step_count, 0], "train")
                # print ("In brain self.guided_action = {}  self.guided_policy = {}".format(self.guided_action, self.guided_policy))
                print("Failed action set is: ", brain1.failed_action_set)
                self.learned, data, data_eval = brain1.call_learner(failed_action=failed_action,
                                                                    actions_bump_up=self.actions_bump_up,
                                                                    new_item_in_the_world=self.new_item_in_world,
                                                                    env=self.env, transfer=args['transfer'],
                                                                    plan = game_action_set,
                                                                    guided_action=self.guided_action,
                                                                    guided_policy=self.guided_policy,
                                                                    exploration_mode=self.exploration_mode)
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
    N_PLACEHOLDERS_INVENTORY = 2
    N_PLACEHOLDERS_LIDAR = 2
    N_PLACEHOLDERS_ACTIONS = 3
    EVAL_EVERY_N_EPISODES = 500

    def __init__(self, args):
        self.TRAIN_EPISODES = args["train_episodes"]
        self.load_model = args["load_model"]
        self.reward_shaping = args["reward_shaping"]
        self.algorithm = args["algorithm"]

        super(BaselineExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                                 f"{to_datestring(time.time())}-baseline-{self.algorithm}-{self.TRAIN_EPISODES}episodes-"
                                                 f"{'rewardshapingon' if self.reward_shaping else 'rewardshapingoff'}")

        self.env = StatePlaceholderWrapper(self.env, n_placeholders_inventory=self.N_PLACEHOLDERS_INVENTORY,
                                           n_placeholders_lidar=self.N_PLACEHOLDERS_LIDAR)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=self.N_PLACEHOLDERS_ACTIONS)

        if self.load_model:
            print(f"Attempting to load pretrained model {self.load_model}.")
            self.experiment_id = self.load_model.split(os.sep)[0]
            self.model = PPO.load(
                Experiment.DATA_DIR + os.sep + self.load_model)
        else:
            os.makedirs(self._get_results_dir(), exist_ok=True)
            self.model = PPO("MlpPolicy", self.env, verbose=0)

        # Environment wrappers
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            self.env = RewardShaping(self.env)

        # This is to use the env with all the wrappers for the model.
        self.model.set_env(self.env)

    def run(self):
        if not self.load_model:
            print("No pretrained model supplied, training from scratch.")
            self.pre_novelty()
        else:
            print(f"Skipping training because pretrained model {self.load_model} was supplied.")

        print("Evaluating model performance pre-novelty.")
        self.env.metadata['mode'] = 'prenovelty-test'
        evaluate_policy(self.model, self.env, self.trials_pre_novelty, deterministic=False, render=self.render)

        if self.novelty_name:
            print(f"Injecting novelty {self.novelty_name} and starting to relearn.")
            self.post_novelty_learn()
            print(f"Evaluating final policy on {self.novelty_name}")
            self.post_novelty_recover()
        self.env.close()

    def pre_novelty(self):
        print(f"Training model for {self.TRAIN_EPISODES} episodes")
        self.env.metadata['mode'] = 'prenovelty-train'
        self.env = Monitor(self.env, self._get_results_dir() + os.sep + "prenovelty-monitor.csv",
                           allow_early_resets=True, info_keywords=('success', 'mode'))
        check_env(self.env, warn=True)

        checkpoint_callback = CheckpointCallback(save_freq=self.MAX_TIMESTEPS_PER_EPISODE * 500,
                                                 save_path=self._get_results_dir() + os.sep + 'prenovelty-checkpoints',
                                                 name_prefix=BaselineExperiment.SAVED_MODEL_NAME)
        self.model.set_env(self.env)
        self.model.learn(total_timesteps=self.TRAIN_EPISODES * self.MAX_TIMESTEPS_PER_EPISODE,
                         callback=checkpoint_callback)
        self.model.save(self._get_results_dir() + os.sep + 'prenovelty_model')



    def post_novelty_learn(self):
        # Recreate env to inject novelty
        self.env = gym.make(ENV_ID)
        self.env = inject_novelty(self.env, self.novelty_name)

        # Wrap env with correct placeholder numbers
        d_obs_inventory, d_obs_lidar, d_actions = get_difference_in_obs_action_space(self.novelty_name)
        self.env = StatePlaceholderWrapper(self.env,
                                           n_placeholders_inventory=self.N_PLACEHOLDERS_INVENTORY - d_obs_inventory,
                                           n_placeholders_lidar=self.N_PLACEHOLDERS_LIDAR - d_obs_lidar)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=self.N_PLACEHOLDERS_ACTIONS - d_actions)

        # Rewrap the environment with everything else
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            self.env = RewardShaping(self.env)

        self.env = Monitor(self.env, f"{self._get_results_dir() + os.sep + self.novelty_name}-monitor.csv",
                           allow_early_resets=True, info_keywords=('success', 'mode'))
        check_env(self.env, warn=True)
        self.env.metadata['mode'] = 'learn-postnovelty-train'

        self.model.set_env(self.env)

        print(f"Evaluation - Model: {self.algorithm}, NOVELTY: {self.novelty_name}, EPISODES: {self.TRAIN_EPISODES}")
        checkpoint_callback = CheckpointCallback(save_freq=self.MAX_TIMESTEPS_PER_EPISODE * self.EVAL_EVERY_N_EPISODES,
                                                 save_path=self._get_results_dir() + os.sep + self.novelty_name + '-checkpoints',
                                                 name_prefix=self.novelty_name + "-" + BaselineExperiment.SAVED_MODEL_NAME)
        eval_callback = CustomEvalCallback(evaluate_every_n=BaselineExperiment.EVAL_EVERY_N_EPISODES,
                                           trials_post_learning=self.trials_post_learning)

        self.model.learn(total_timesteps=self.TRAIN_EPISODES * self.MAX_TIMESTEPS_PER_EPISODE,
                         callback=[checkpoint_callback, eval_callback])
        self.model.save(self._get_results_dir() + os.sep + self.novelty_name + "-" + BaselineExperiment.SAVED_MODEL_NAME)

    def post_novelty_recover(self):
        # evaluate the final policy
        self.env.metadata['mode'] = 'test-recovery-postnovelty'
        evaluate_policy(self.model, self.env, self.trials_post_learning, deterministic=False, render=self.render)


class PolicyGradientExperiment(Experiment):
    HEADER_TRAIN = ['episode', 'timesteps', 'reward', 'success']
    HEADER_TEST = ['trial', 'episode', 'timesteps', 'reward', 'success']
    MAX_TIMESTEPS_PER_EPISODE = 500
    SAVED_MODEL_NAME = 'model'
    N_PLACEHOLDERS_INVENTORY = 2
    N_PLACEHOLDERS_LIDAR = 2
    N_PLACEHOLDERS_ACTIONS = 3
    EVAL_EVERY_N_EPISODES = 500

    def __init__(self, args):
        self.TRAIN_EPISODES = args["train_episodes"]
        self.load_model = args["load_model"]
        self.reward_shaping = args["reward_shaping"]
        self.algorithm = args["algorithm"]

        super(PolicyGradientExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                                 f"{to_datestring(time.time())}-policygradient-{self.TRAIN_EPISODES}episodes")
        self.CHECKPOINT_DIR = f"{self._get_results_dir()}{os.sep}prenovelty-checkpoints"
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        self.env = StatePlaceholderWrapper(self.env, n_placeholders_inventory=self.N_PLACEHOLDERS_INVENTORY,
                                           n_placeholders_lidar=self.N_PLACEHOLDERS_LIDAR)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=self.N_PLACEHOLDERS_ACTIONS)

        # Environment wrappers
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            self.env = RewardShaping(self.env)

        self.model = SimpleDQN(int(self.env.action_space.n), int(self.env.observation_space.shape[0]),
                               NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON,
                               False, {}, self.env.actions_id, random_seed)

        if self.load_model:
            print(f"Attempting to load pretrained model {self.load_model}.")
            self.experiment_id = self.load_model.split(os.sep)[0]
            self.model.load_model("", "", path_to_load=Experiment.DATA_DIR + os.sep + self.load_model)
        else:
            os.makedirs(self._get_results_dir(), exist_ok=True)

        self.model.set_explore_epsilon(MAX_EPSILON)


    def run(self):
        if not self.load_model:
            print("No pretrained model supplied, training from scratch.")
            self.pre_novelty()
        else:
            print(f"Skipping training because pretrained model {self.load_model} was supplied.")

        if self.novelty_name:
            print(f"Injecting novelty {self.novelty_name} and starting to relearn.")
            self.post_novelty_learn()
            print(f"Evaluating final policy on {self.novelty_name}")
            self.post_novelty_recover()
        self.env.close()

    def pre_novelty(self):
        self.env.metadata['mode'] = 'prenovelty-train'
        self.env = Monitor(self.env, self._get_results_dir() + os.sep + "prenovelty-monitor.csv",
                           allow_early_resets=True, info_keywords=('success', 'mode'))
        check_env(self.env, warn=True)

        self._train_policy_gradient()
        self.model.save_model("", "", path_to_save=f"{self._get_results_dir()}{os.sep}prenovelty_model.npz")

        print("Evaluating model performance pre-novelty.")
        self.env.metadata['mode'] = 'prenovelty-test'
        for i in range(self.trials_pre_novelty):
            self._single_eval_episode()

    def post_novelty_learn(self):
        # Recreate env to inject novelty
        self.env = gym.make(ENV_ID)
        self.env = inject_novelty(self.env, self.novelty_name)

        # Wrap env with correct placeholder numbers
        d_obs_inventory, d_obs_lidar, d_actions = get_difference_in_obs_action_space(self.novelty_name)
        self.env = StatePlaceholderWrapper(self.env,
                                           n_placeholders_inventory=self.N_PLACEHOLDERS_INVENTORY - d_obs_inventory,
                                           n_placeholders_lidar=self.N_PLACEHOLDERS_LIDAR - d_obs_lidar)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=self.N_PLACEHOLDERS_ACTIONS - d_actions)

        # Rewrap the environment with everything else
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            self.env = RewardShaping(self.env)

        self.env = Monitor(self.env, f"{self._get_results_dir() + os.sep + self.novelty_name}-monitor.csv",
                           allow_early_resets=True, info_keywords=('success', 'mode'))
        check_env(self.env, warn=True)

        self.model = SimpleDQN(int(self.env.action_space.n), int(self.env.observation_space.shape[0]),
                               NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON,
                               False, {}, self.env.actions_id, random_seed)
        print(f"Attempting to load pretrained model {self.load_model}.")
        self.model.load_model("", "", path_to_load=f"{self._get_results_dir()}{os.sep}prenovelty_model.npz")

        self.env.metadata['mode'] = 'learn-postnovelty-train'
        self._train_policy_gradient()
        self.model.save_model("", "", path_to_save=f"{self._get_results_dir()}{os.sep}{self.novelty_name}_model.npz")

    def post_novelty_recover(self):
        # evaluate the final policy
        self.env.metadata['mode'] = 'test-recovery-postnovelty'
        for i in range(self.trials_post_learning):
            self._single_eval_episode()

    def _train_policy_gradient(self, eval_every_n: int=-1, mode: str= 'prenovelty-test'):
        self.Epsilons = []
        self.Rhos = []
        self.Steps = []
        self.R = []
        self.Done = []

        print(f"Training model for {self.TRAIN_EPISODES} episodes")
        for episode in range(0, self.TRAIN_EPISODES):
            # Evaluate every n episodes
            if eval_every_n > 0 and episode % eval_every_n == 0:
                mode_before = self.env.metadata['mode']
                self.env.metadata['mode'] = mode
                for i in range(self.trials_post_learning):
                    self._single_eval_episode()
                self.env.metadata['mode'] = mode_before
            if episode % 500 == 0 and episode > 0:
                path = f"{self.CHECKPOINT_DIR}{os.sep}{BaselineExperiment.SAVED_MODEL_NAME}-{str(episode)}episodes"
                self.model.save_model("", "", path_to_save=path)
            reward_per_episode = 0
            episode_timesteps = 0
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                      math.exp(-LAMBDA * episode)
            self.model._explore_eps = epsilon
            self.rho = MIN_RHO + (MAX_RHO - MIN_RHO) * \
                       math.exp(-LAMBDA * episode)
            obs = self.env.reset()
            info = self.env.get_info()

            while True:
                action = self.model.process_step(obs, True)
                obs, rew, done, info = self.env.step(action)

                episode_timesteps += 1

                self.model.give_reward(rew)
                # here we update the network with the observation and reward
                reward_per_episode += rew  # save reward
                if done or episode_timesteps > PolicyGradientExperiment.MAX_TIMESTEPS_PER_EPISODE:
                    self.model.finish_episode()
                    if episode % UPDATE_RATE == 0:
                        self.model.update_parameters()
                    if done:
                        if episode % PRINT_EVERY == 0:
                            print("--EP >>{}, steps>>{},  Rew>>{}, done({})>>{}, eps>>{} rho>>{} \n".format(episode,
                                                                                                            episode_timesteps,
                                                                                                            reward_per_episode,
                                                                                                            NO_OF_DONES_TO_CHECK,
                                                                                                            np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]),
                                                                                                            round(self.model._explore_eps, 3),
                                                                                                            round(self.rho, 3)))
                        self.Epsilons.append(self.model._explore_eps)
                        self.Rhos.append(self.rho)
                        self.Steps.append(episode_timesteps)
                        self.Done.append(1)
                        self.R.append(reward_per_episode)
                        if episode > NO_OF_EPS_TO_CHECK:
                            if np.mean(self.R[-NO_OF_EPS_TO_CHECK:]) > SCORE_TO_CHECK:  # check the average reward for last 70 episodes
                                # for future we can write an evaluation function here which runs a evaluation on the current policy.
                                if np.sum(self.Done[-NO_OF_DONES_TO_CHECK:]) > NO_OF_SUCCESSFUL_DONE:  # and check the success percentage of the agent > 80%.
                                    print("The agent has learned to reach the subgoal")
                                    return True
                        break
                    elif episode_timesteps >= PolicyGradientExperiment.MAX_TIMESTEPS_PER_EPISODE:
                        if episode % PRINT_EVERY == 0:
                            print("--EP >>{}, steps>>{},  Rew>>{}, done({})>>{}, eps>>{} rho>>{} \n".format(episode,
                                                                                                            episode_timesteps,
                                                                                                            reward_per_episode,
                                                                                                            NO_OF_DONES_TO_CHECK,
                                                                                                            np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]),
                                                                                                            round(self.model._explore_eps, 3),
                                                                                                            round(self.rho, 3)))
                        self.Epsilons.append(self.model._explore_eps)
                        self.Rhos.append(self.rho)
                        self.Steps.append(episode_timesteps)
                        self.Done.append(0)
                        self.R.append(reward_per_episode)
                        break

    def _single_eval_episode(self):
        obs = self.env.reset()
        info = self.env.get_info()
        done = False

        while not done:
            action = self.model.process_step(obs, True)
            obs, rew, done, info = self.env.step(action)


def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
    return datetime.utcfromtimestamp(unixtime).strftime(format)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="rapid")
    ap.add_argument("-N", "--novelty_name", default=None,
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("-TP", "--trials_pre_novelty", default=100, help="Number of trials pre novelty", type=int)
    ap.add_argument("-TN", "--trials_post_learning", default=1, help="Number of trials post recovering from novelty",
                    type=int)
    ap.add_argument("-P", "--print_every", default=200, help="Number of epsiodes you want to print the results",
                    type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, both, action_biasing, guided_policy", type=str)
    ap.add_argument("-T", "--transfer", default=None, type=str)
    ap.add_argument("-R", "--render", default=False, type=bool)
    ap.add_argument("-E", "--exploration_mode", default='uniform' , help="uniform, ucb", type=str)

    ap.add_argument("--load_model", default=None, type=str)
    ap.add_argument("--train_episodes", default=10, type=int)
    ap.add_argument("--reward_shaping", dest="reward_shaping", action="store_true")
    ap.add_argument("--no_reward_shaping", dest="reward_shaping", action="store_false")
    ap.set_defaults(reward_shaping=True)

    ap.add_argument("--algorithm", default="PPO", type=str)

    args = vars(ap.parse_args())
    if args['experiment'] == 'baseline':
        experiment1 = BaselineExperiment(args)
    elif args['experiment'] == 'policy_gradient':
        experiment1 = PolicyGradientExperiment(args)
    else:
        experiment1 = RapidExperiment(args)
    experiment1.run()
