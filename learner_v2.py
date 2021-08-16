'''
Author: Shivam Goel
Email: goelshivam1210@gmail.com
'''

import time
import numpy as np
import math
import os
import copy
import matplotlib.pyplot as plt 
from NGLearner_util_v2 import get_create_success_func_from_predicate_set
from generate_pddl import *
from SimpleDQN import *
# Params
# STEPCOST_PENALTY = 0.012
MAX_TIMESTEPS = 150
MAX_EPISODES = 10000
NUM_HIDDEN = 16
GAMMA = 0.95
LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MAX_EPSILON = 0.1
random_seed = 2

class Learner:
    def __init__(self, failed_action, env, novelty_flag=False) -> None:

        self.env_to_reset = copy.deepcopy(env) # copy the complete environment instance
        self.env = env
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

        agent = SimpleDQN(int(env.action_space.n),int(env.observation_space.shape[0]),NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
        agent.set_explore_epsilon(MAX_EPSILON)
        self.learning_agent = agent
        self.reset_near_values = None
        self.reset_select_values = None
        self.updated_spaces = False
        self.can_motion_plan = True
        self.can_trade_plan = False

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
        # self.env.run_SENSE_RECIPES_and_update()
        # self.env.run_SENSE_ALL_and_update('NONAV')
        # if not self.env.first_space_init:
        #     self.env.generate_obs_action_spaces()

        self.R = [] # storing rewards per episode
        self.Done = [] # storing goal completion per episode

        # if not self.success_func:
        self.learned_failed_action = self.run_episode(transfer, novelty_name)

        if self.learned_failed_action:
            self.learning_agent.save_model(0,0,0)
            return True
        else:
            return False

    # Run episodes for certain time steps.
    def run_episode(self, transfer= None, novelty_name = None):
        done = False
        obs = self.env.get_observation()
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
            # self.env.render()
            obs = self.env.reset(reset_from_failed_state = True, env_instance = copy.deepcopy(self.env_to_reset))
            # print(self.env.inventory_items_quantity)
            # time.sleep(3)
            info = self.env.get_info()
            self.success_func = self.create_success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
            while True:
                obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)
                # print("inventory: ", self.env.inventory_items_quantity)
                # time.sleep(0.3)
                episode_timesteps += 1
                self.is_success_met = self.success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
                if self.is_success_met:    
                    done = True
                    rew = 1000
                else:
                    done = False
                    rew = -1
                # agent.give_reward(reward)
                self.learning_agent.give_reward(rew)
                # here we update the network with the observation and reward
                reward_per_episode += rew # save reward

                if done or episode_timesteps > MAX_TIMESTEPS:
                        
                    self.learning_agent.finish_episode()
                    if episode % 10 == 0:
                        self.learning_agent.update_parameters()

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
                        print ("EP >> {}, Timesteps >> {},  Rew >> {} done = {} done rate (20) = {}".format(episode, episode_timesteps, reward_per_episode, done, np.mean(self.Done[-20:])))
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

        # self.learning_agent.store_obs(obs)

        if self.mode == 'exploration':
            action = self.learning_agent.process_step(obs, True)
        else: # self.mode is exploitation -> greedy policy used
            action = self.learning_agent.process_step(obs,False)
        
        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action)
        # self.env.render()
        # self.last_action = self.env.all_actions[action]
        self.last_obs = orig_obs.copy()
        info = info_

        return obs2, action, done, info

    def play_learned_policy(self, env, novelty_name, operator_name):
        # self.env = GridworldMDP(env, False, render=False) # reinstantiate a new env instance.
        self.env.render()
        self.env = env
        self.mode = 'exploitation' # want to act greedy
        self.is_success_met = False        
        done = False
        obs = self.env.get_observation()
        info = self.env.get_info()
        self.learning_agent.load_model(0,0,0)
        episode_timestep = 0
        self.success_func = self.create_success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met

        while True:
            
            obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)
            self.env.render()
            self.is_success_met = self.success_func(self.env.get_observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
            episode_timestep += 1 
            if self.is_success_met:
                done = True
                # print("inventory: ", self.env.inventory_items_quantity)
                # time.sleep(20)
                return True
            if episode_timestep >= 225:
                done = False
                return False


if __name__ == '__main__':
    failed_action = None
    learn = Learner()
    # learn.play_learned_policy()