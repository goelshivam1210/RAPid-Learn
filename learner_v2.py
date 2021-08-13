'''
Author: Shivam Goel
Email: goelshivam1210@gmail.com
'''

import time
import numpy as np
import math
import tensorflow as tf
import os
import copy
import matplotlib.pyplot as plt 

# from NGLearner_util_v2 import PolycraftDynamicsChecker, get_create_success_func_from_predicate_set, default_param_dict, GridworldMDP, add_reset_probas, informed_random_action, check_permissible, update_reset_probas, reset_to_interesting_state
from NGLearner_util_v2 import get_create_success_func_from_predicate_set, default_param_dict, GridworldMDP, add_reset_probas
from learning.dqn import DQNLambda_Agent
# from polycraft_tufts.rl_agent.dqn_lambda.learning.utils import make_session
from learning.utils import make_session
from generate_pddl import *
# Params
# STEPCOST_PENALTY = 0.012
MAX_TIMESTEPS = 150
MAX_EPISODES = 10000

class Learner:
    def __init__(self, failed_action, env, novelty_flag=False) -> None:
        # self.encounter_novelty_flag = novelty_flag
        # self.resettable_env_items_quantity = copy.deepcopy(env.items_quantity)
        # self.resettable_env_inventory_items_quantity = copy.deepcopy(env.inventory_items_quantity)
        self.env_to_reset = copy.deepcopy(env) # copy the complete environment instance
        # self.resettable_map = copy.deepcopy(env.map_size)
        self.env = GridworldMDP(env, False)
        if failed_action == "Break":
            failed_action = "Break tree_log"
        self.failed_action = failed_action
        self.learned_failed_action = False
        if not self.learned_failed_action:
            self.mode = 'exploration' # we are exploring the novel environment to stitch the plan by learning the new failed action.
        else:
            self.mode = 'exploitation' # Need to set the exploration coefficient to Zero, as we have already learned the policy

        operator_map = {
            "approach crafting_table tree_log": ['facing tree_log'],
            "approach tree_tap crafting_table": ['facing tree_log'],
            "approach air tree_log": ['facing tree_log'],
            "approach air crafting_table": ['facing tree_log'],
            "approach minecraft:crafting_table": ['facing crafting_table'],
            "Break tree_log": ['increase inventory_log 1'],
            "Craft_plank": ['increase inventory plank 1'],
            "Craft_stick": ['increase inventory stick 1'],
            "Craft_tree_tap": ['increase inventory tree_tap 1'],
            "Craft_pogo_stick": ['increase inventory pogo_stick 1'],
            "Extract_rubber": ['increase inventory rubber 1'],
        }

        self.desired_effects = operator_map[failed_action]
        print("desired effects: ", self.desired_effects)
        self.create_success_func = get_create_success_func_from_predicate_set(self.desired_effects)
        self.success_func = None

        self.session = make_session(0)
        default_param_dict['eps_lambda'] = -math.log(0.01) / 4000.
        agent = DQNLambda_Agent(0, self.env, scope='exploration', session=self.session,
                                **default_param_dict)
        self.learning_agent = agent
        add_reset_probas(self, exploration=True)

        self.reset_near_values = None
        self.reset_select_values = None
        self.updated_spaces = False
        self.can_motion_plan = True
        self.can_trade_plan = False

        add_reset_probas(self, exploration=True)

    def reset_trial_vars(self):
        self.last_action = None
        self.last_obs = None
        if not self.learned_failed_action:
            self.mode = 'exploration'
        else:
            self.mode = 'exploitation'
        self.found_relevant_during_reset = False
        self.found_relevant_exp_state = 0
        self.last_reset_pos = None
        self.placed_tap = False
        self.resetting_state = False
        self.motion_planning = False
        self.failed_last_motion_plan = False
        self.novel_item_encountered = False
        self.last_outcome = None
        self.impermissible_performed = False
        self.impermissible_reason = None
        self.trial_time = 300

    def learn_policy(self, novelty_name, transfer=False):
        self.reset_trial_vars()
        self.env.run_SENSE_RECIPES_and_update()
        self.env.run_SENSE_ALL_and_update('NONAV')
        if not self.env.first_space_init:
            self.env.generate_obs_action_spaces()

        self.R = [] # storing rewards per episode
        self.Done = [] # storing goal completion per episode

        # if not self.success_func:
        self.learned_failed_action = self.run_episode(transfer, novelty_name)

        if self.learned_failed_action:
            self.learning_agent.save_model(novelty_name = 
            novelty_name, operator_name=self.failed_action)
            return True
        else:
            return False

    # Run episodes for certain time steps.
    def run_episode(self, transfer= None, novelty_name = None):
        done = False
        obs = self.env.observation()
        info = self.env.get_info()

        possible_outcomes = {
            'plannable':0,
            'unplannable':1
        }
        # if transfer:
        # self.learning_agent.load_model(novelty_name = novelty_name, operator_name = self.failed_action)

        # episode_timesteps = 0
        for episode in range(MAX_EPISODES):
            # time.sleep(2)
            reward_per_episode = 0
            episode_timesteps = 0
            obs = self.env.mdp_gridworld_reset(reset_from_failed_state = True, env_instance = copy.deepcopy(self.env_to_reset))
            self.success_func = self.create_success_func(self.env.observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
            while True:
                obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)

                # time.sleep(3)
                episode_timesteps += 1
                self.is_success_met = self.success_func(self.env.observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
                if self.is_success_met:    
                    done = True
                    rew = 1000
                else:
                    done = False
                    rew = -1
                # here we update the network with the observation and reward
                self.learning_agent.store_effect(action, rew, done)
                self.learning_agent.timesteps_trained += 1
                self.learning_agent.check_update()

                reward_per_episode += rew # save reward
                if done:
                    print ("EP >> {}, Timesteps >> {},  Rew >> {}, done = {}, done rate (20) = {}".format(episode, episode_timesteps, reward_per_episode, done, np.mean(self.Done[-20:])))
                    print("\n")
                    self.Done.append(1)
                    self.R.append(reward_per_episode)
                    if episode > 70:
                        if np.mean(self.R[-20:]) > 900: # check the average reward for last 70 episodes
                            # for future we can write an evaluation function here which runs a evaluation on the current policy.
                            if  np.sum(self.Done[-20:]) > 17: # and check the success percentage of the agent > 80%.
                                print ("The agent has learned to reach the subgoal")

                                # plotting function. Just for testing purposes.
                                plt.plot(self.R)
                                plt.xlabel("Episodes")
                                plt.ylabel("Reward per episode")
                                plt.title("Learning to Break a Log with Axe: Performance")
                                plt.grid(True)
                                plt.legend()
                                # plt.show()
                                return True  
                    break
                elif episode_timesteps >= MAX_TIMESTEPS:
                    print ("EP >> {}, Timesteps >> {},  Rew >> {} done = {} done rate (20) = {}".format(episode, episode_timesteps, reward_per_episode, done, self.learning_agent.epsilon, np.mean(self.Done[-20:])))
                    print("\n")
                    self.Done.append(0)
                    self.R.append(reward_per_episode)
                    break
        return False

    def step_env(self, action=None, orig_obs=None, info=None, done=False, store_transition=None, evaluate=False):

        if orig_obs is None:
            orig_obs = self.env.observation()
        if info is None:
            info = self.env.get_info()

        obs = orig_obs.copy()

        self.learning_agent.store_obs(obs)

        if self.mode == 'exploration':
            if np.random.random() < self.learning_agent.epsilon:
                action = self.learning_agent.get_action(obs, info, 0)
            else:
                action = self.learning_agent.get_action(obs, info, 0)
        else: # self.mode is exploitation -> greedy policy used
            action = self.learning_agent.get_action(obs, info, 0)
        
        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action)
        self.last_action = self.env.all_actions[action]
        self.last_obs = orig_obs.copy()
        info = info_

        return obs2, action, done, info

    def play_learned_policy(self, env, novelty_name, operator_name):
        self.env = GridworldMDP(env, False, render=False) # reinstantiate a new env instance.
        self.mode = 'exploitation' # want to act greedy
        self.is_success_met = False        
        done = False
        obs = self.env.observation()
        info = self.env.get_info()
        self.learning_agent.load_model(novelty_name = novelty_name, operator_name = self.failed_action)
        episode_timestep = 0
        while True:

            obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)
            self.is_success_met = self.success_func(self.env.observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
            episode_timestep += 1 
            if self.is_success_met:
                done = True
                return True
            if episode_timestep >= 225:
                done = False
                return False

    def update_planner(self):
        domain_file = None
        return domain_file
        # pass

if __name__ == '__main__':
    failed_action = None
    learn = Learner()
    # learn.play_learned_policy()