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
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from params import *

from stable_baselines3 import PPO

from SimpleDQN import SimpleDQN
from baselines.callbacks import CustomEvalCallback

from baselines.util import get_difference_in_obs_action_space, to_datestring, find_max_trial_number
from brain import Brain
from baselines.wrappers import *
from params import *
from gym_novel_gridworlds.novelty_wrappers import inject_novelty

ENV_ID = 'NovelGridworld-Pogostick-v1'  # always remains the same.


class Experiment:
    DATA_DIR = Path('./data/')
    CHECKPOINT_EVERY_N = 200
    EVALS_AFTER_LEARNING = 100
    EVALS_DURING_LEARNING = 10
    EVAL_EVERY_N_EPISODES = 100
    MAX_TIMESTEPS_PER_EPISODE = 500

    N_PLACEHOLDERS_INVENTORY = 2
    N_PLACEHOLDERS_LIDAR = 2
    N_PLACEHOLDERS_ACTIONS = 5

    def __init__(self, args, trial_id):
        self.trial_id = trial_id
        self.novelty_name = args['novelty_name']
        self.render = args['render']
        self.reward_shaping = args["reward_shaping"]
        self.experiment_id = args["experiment_id"]
        self.train_episodes = args["train_episodes"]
        self.load_model = args["load_model"]
        self.learner = args["learner"]
        self.exploration_mode = args["exploration_mode"]
        self.algorithm = args["algorithm"]

        self.evals_during_learning = Experiment.EVALS_DURING_LEARNING if self.novelty_name != "prenovelty" else 0

        if not self.experiment_id:
            self.experiment_id = f"{uuid.uuid4().hex}-{to_datestring(time.time())}-{self.algorithm}-" \
                                 f"{self.train_episodes}episodes-" \
                                 f"{'rewardshapingon' if self.reward_shaping else 'rewardshapingoff'}"

        if self.novelty_name is None:
            self.novelty_name = "prenovelty"

        self.experiment_dir = self._get_experiment_dir()
        os.makedirs(self._get_trial_dir(), exist_ok=False)

        set_random_seed(trial_id, using_cuda=True)

        self.env = gym.make(ENV_ID)

        if self.novelty_name != "prenovelty":
            self.env = inject_novelty(self.env, self.novelty_name)

        d_obs_inventory, d_obs_lidar, d_actions = get_difference_in_obs_action_space(self.novelty_name)
        self.env = StatePlaceholderWrapper(self.env,
                                           n_placeholders_inventory=Experiment.N_PLACEHOLDERS_INVENTORY - d_obs_inventory,
                                           n_placeholders_lidar=Experiment.N_PLACEHOLDERS_LIDAR - d_obs_lidar)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=Experiment.N_PLACEHOLDERS_ACTIONS - d_actions)

        # Environment wrappers
        self.env = EpisodicWrapper(self.env, self.MAX_TIMESTEPS_PER_EPISODE, verbose=True)
        self.env = InfoExtenderWrapper(self.env)
        if self.reward_shaping:
            self.env = RewardShaping(self.env)
        monitor_filename = str(self._get_trial_dir() / f"{self.novelty_name}-{self.trial_id}-monitor.csv")
        self.env = Monitor(self.env, monitor_filename,
                           allow_early_resets=True, info_keywords=('success', 'mode', "episode_counter"))

        check_env(self.env, warn=True)

    def _get_experiment_dir(self):
        return Experiment.DATA_DIR / self.experiment_id

    def _get_novelty_dir(self):
        return self._get_experiment_dir() / self.novelty_name

    def _get_trial_dir(self):
        return self._get_novelty_dir() / f"trial-{self.trial_id}"

    @abstractmethod
    def run(self):
        pass

    def write_row_to_results(self, data, tag):
        db_file_name = self.experiment_dir + os.sep + str(tag) + "results.csv"
        with open(db_file_name, 'a') as f:  # append to the file created
            writer = csv.writer(f)
            writer.writerow(data)

    def write_params_to_file(self, data):
        db_file_name = self.experiment_dir + os.sep + "params.json"
        out_file = open(db_file_name, "w") 
        json.dump(data, out_file, indent = 6) 
        out_file.close() 


class RapidExperiment(Experiment):
    HEADER_TRAIN = ['Trial_no', 'Epsilon', 'Rho', 'Timesteps', 'Reward', 'Done']
    HEADER_TEST = ['Mode', 'Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done'] # Mode = 0: Planning, 1 = Learning, 2= Recovery

    def __init__(self, args):
        super(RapidExperiment, self).__init__(args, self.HEADER_TRAIN, self.HEADER_TEST,
                                              "_" + args['novelty_name'] + "_"+ args['learner'] + "_"+ args['exploration_mode'])
        os.makedirs(self.experiment_dir, exist_ok=True)

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
        for pre_novelty_trial in range(self.evals_pre_novelty):
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
        while post_novelty_trial <= self.evals_post_learning:
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
    def __init__(self, args, trial_id):
        super(BaselineExperiment, self).__init__(args, trial_id)

        if self.novelty_name != "prenovelty":
            self.model = PPO.load(self._get_experiment_dir() / "prenovelty" / "model_prenovelty")
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=0, gamma=GAMMA)

        # Override if a particular model is supplied
        if self.load_model:
            self.model = PPO.load(self._get_experiment_dir() / self.load_model)

        # This is to use the env with all the wrappers for the model.
        self.model.set_env(self.env)

    def run(self):
        print(f"Starting trial {self.trial_id} for Experiment ID: {self.experiment_id} | Training {self.algorithm} on "
              f"novelty: {self.novelty_name}")
        self.env.metadata['mode'] = 'learn'
        checkpoint_callback = CheckpointCallback(
            save_freq=self.MAX_TIMESTEPS_PER_EPISODE * Experiment.CHECKPOINT_EVERY_N,
            save_path=str(self._get_trial_dir() / 'checkpoints'))
        max_episodes_stop_callback = StopTrainingOnMaxEpisodes(max_episodes=self.train_episodes)
        eval_callback = CustomEvalCallback(evaluate_every_n=BaselineExperiment.EVAL_EVERY_N_EPISODES,
                                           n_eval_episodes=self.evals_during_learning, render=self.render)

        self.model.set_env(self.env)
        self.model.learn(total_timesteps=self.train_episodes * self.MAX_TIMESTEPS_PER_EPISODE,
                         callback=CallbackList([checkpoint_callback, eval_callback, max_episodes_stop_callback]))
        self.model.save(self._get_trial_dir() / f'model_{self.novelty_name}')

        print(f"Evaluation - Model: {self.algorithm}, NOVELTY: {self.novelty_name}, AFTER_N_EPISODES: {self.train_episodes}")
        self.env.metadata['mode'] = 'eval'
        evaluate_policy(self.model, self.env, Experiment.EVALS_AFTER_LEARNING, deterministic=False, render=self.render)

        self.env.close()


class PolicyGradientExperiment(Experiment):
    def __init__(self, args, trial_id):
        super(PolicyGradientExperiment, self).__init__(args, trial_id)

        self.model = SimpleDQN(int(self.env.action_space.n), int(self.env.observation_space.shape[0]),
                               NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON,
                               False, {}, self.env.actions_id, trial_id, self.learner, self.exploration_mode)

        if self.novelty_name != "prenovelty":
            self.model.load_model("", "", path_to_load=self._get_experiment_dir() / "prenovelty" / "model_prenovelty.npz")

        # Override if a particular model is supplied
        if self.load_model:
            self.model.load_model("", "", path_to_load=self._get_experiment_dir() / self.load_model)

        self.model.set_explore_epsilon(MAX_EPSILON)

        self.CHECKPOINT_DIR = self._get_trial_dir() / 'checkpoints'
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

    def run(self):
        print(f"Starting trial {self.trial_id} for Experiment ID: {self.experiment_id} | Training {self.algorithm} on "
              f"novelty: {self.novelty_name}")
        self.env.metadata['mode'] = 'learn'
        self._train_policy_gradient()
        self.model.save_model("", "", path_to_save=self._get_trial_dir() / f"model_{self.novelty_name}.npz")

        print(f"Evaluation - Model: {self.algorithm}, NOVELTY: {self.novelty_name}, AFTER_N_EPISODES: {self.train_episodes}")
        self.model.set_explore_epsilon(MIN_EPSILON)
        self.env.metadata['mode'] = 'eval'
        for i in range(Experiment.EVALS_AFTER_LEARNING):
            self._single_eval_episode()
        self.env.close()

    def _train_policy_gradient(self):
        self.R = []
        self.Done = []

        print(f"Training policy gradient for {self.train_episodes} episodes")

        for episode in range(0, self.train_episodes):
            # Evaluate every n episodes
            if self.novelty_name != 'prenovelty' and episode > 0 and episode % self.EVAL_EVERY_N_EPISODES == 0:
                self.env.metadata['mode'] = "eval"
                for i in range(self.evals_during_learning):
                    self._single_eval_episode()
                self.env.metadata['mode'] = "learn"
            if episode > 0 and episode % Experiment.CHECKPOINT_EVERY_N == 0:
                self.model.save_model("", "", path_to_save=self.CHECKPOINT_DIR /
                                                           f"model-{self.novelty_name}"
                                                           f"-{str(episode)}episodes.npz")
            reward_per_episode = 0
            episode_timesteps = 0
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * episode)
            self.model._explore_eps = epsilon
            self.rho = MIN_RHO + (MAX_RHO - MIN_RHO) * math.exp(-LAMBDA * episode)
            obs = self.env.reset()

            while True:
                action = self.model.process_step(obs, True, episode_timesteps)
                obs, rew, done, info = self.env.step(action)
                if self.render:
                    self.env.render()

                episode_timesteps += 1
                self.model.give_reward(rew)
                # here we update the network with the observation and reward
                reward_per_episode += rew  # save reward
                if done:
                    self.model.finish_episode()
                    if episode % UPDATE_RATE == 0:
                        self.model.update_parameters()
                    if episode % PRINT_EVERY == 0:
                        print("--EP >>{}, steps>>{},  Rew>>{}, done({})>>{}, eps>>{} rho>>{} \n".format(episode,
                                                                                                        episode_timesteps,
                                                                                                        reward_per_episode,
                                                                                                        NO_OF_DONES_TO_CHECK,
                                                                                                        np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]),
                                                                                                        round(self.model._explore_eps, 3),
                                                                                                        round(self.rho, 3)))
                    crafted_pogostick = info['success']
                    if crafted_pogostick:
                        self.Done.append(1)
                        self.R.append(reward_per_episode)
                        if episode > NO_OF_EPS_TO_CHECK:
                            # check for convergence
                            running_success_mean = np.sum(self.Done[-NO_OF_DONES_TO_CHECK:])
                            if running_success_mean > NO_OF_SUCCESSFUL_DONE:
                                print(f"Agent converged on novelty: {self.novelty_name} with {running_success_mean}% success.")
                                return True
                        break
                    else:
                        self.Done.append(0)
                        self.R.append(reward_per_episode)
                        break

    def _single_eval_episode(self):
        # Need to backup the model state to be able to go back to training
        previous_epsilon = self.model._explore_eps
        self.model._explore_eps = MIN_EPSILON
        _xs, _hs, _dlogps, _drs = self.model._xs, self.model._hs, self.model._dlogps, self.model._drs
        _grad_buffer = self.model._grad_buffer
        _rmsprop_cache = self.model._rmsprop_cache

        self.model.reset()

        obs = self.env.reset()
        info = self.env.get_info()
        done = False
        timestep = 0

        while not done:
            action = self.model.process_step(obs, True, timestep)
            timestep += 1
            obs, rew, done, info = self.env.step(action)
            if self.render:
                self.env.render()

        self.model._explore_eps = previous_epsilon

        # reset values as they were before
        self.model._xs, self.model._hs, self.model._dlogps, self.model._drs = _xs, _hs, _dlogps, _drs
        self.model._grad_buffer = _grad_buffer  # update buffers that add up gradients over a batch
        self.model._rmsprop_cache = _rmsprop_cache  # rmsprop memory


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_id", default=None)
    ap.add_argument("--algorithm", default="PPO", type=str)

    ap.add_argument("-N", "--novelty_name", default=None,
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("-P", "--print_every", default=200, help="Number of epsiodes you want to print the results",
                    type=int)
    ap.add_argument("--n_trials", default=5, help="Number of times to repeat the full experiment",
                    type=int)
    ap.add_argument("-L", "--learner", default='epsilon-greedy', help="epsilon-greedy, both, action_biasing, guided_policy", type=str)
    ap.add_argument("-T", "--transfer", default=None, type=str)
    ap.add_argument("-R", "--render", default=False, type=bool)
    ap.add_argument("-E", "--exploration_mode", default='uniform', help="uniform, ucb", type=str)

    ap.add_argument("--load_model", default=None, type=str)
    ap.add_argument("--train_episodes", default=10, type=int)
    ap.add_argument("--reward_shaping_off", dest="reward_shaping", action="store_false")
    ap.set_defaults(reward_shaping=True)

    args = vars(ap.parse_args())

    n_trials = args['n_trials']

    args["experiment_id"] = args['experiment_id'] if args['experiment_id'] else f"{uuid.uuid4().hex}-{to_datestring(time.time())}-{args['algorithm']}-" \
                                 f"{args['train_episodes']}episodes-" \
                                 f"{'rewardshapingon' if args['reward_shaping'] else 'rewardshapingoff'}"
    args["novelty_name"] = args["novelty_name"] if args["novelty_name"] else "prenovelty"

    novelty_path = Path(Experiment.DATA_DIR) / args["experiment_id"] / args["novelty_name"]
    os.makedirs(novelty_path, exist_ok=True)
    for _ in range(0, n_trials):
        trial_id = find_max_trial_number(novelty_path) + 1
        if args['algorithm'] == 'PPO':
            experiment1 = BaselineExperiment(args, trial_id)
        elif args['algorithm'] == 'policy_gradient':
            experiment1 = PolicyGradientExperiment(args, trial_id)
        else:
            experiment1 = RapidExperiment(args, trial_id)

        experiment1.run()
