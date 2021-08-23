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
from utils import AStarOperator, AStarPlanner

# Params
# STEPCOST_PENALTY = 0.012
MAX_TIMESTEPS = 150
MAX_EPISODES = 100000
NUM_HIDDEN = 16
GAMMA = 0.95
LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MIN_EPSILON = 0.05
MAX_EPSILON = 0.70
random_seed = 2
EXPLORATION_STOP = 30000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay
MIN_RHO = 0.20 # constant for using guided policies.
MAX_RHO = 0.80 

class Learner:
    def __init__(self, failed_action, env, actions_bump_up, action_biasing = False, new_item_in_the_world = None, guided_policy = False, novelty_flag=False) -> None:

        self.env_to_reset = copy.deepcopy(env) # copy the complete environment instance
        self.env = env
        if failed_action == "Break":
            failed_action = "Break tree_log"
        self.failed_action = failed_action
        self.actions_bump_up = actions_bump_up
        self.new_item_in_the_world = new_item_in_the_world
        self.guided_policy = guided_policy
        self.learned_failed_action = False
        self.action_biasing_flag = action_biasing
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

        agent = SimpleDQN(int(env.action_space.n),int(env.observation_space.shape[0]),NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON, self.action_biasing_flag, self.actions_bump_up, self.env.actions_id, random_seed)
        agent.set_explore_epsilon(MAX_EPSILON)

        # if transfer == True:
        #     agent.load_model()

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
            self.learning_agent.save_model(novelty_name, self.failed_action)
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
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                math.exp(-LAMBDA * episode)
            self.learning_agent._explore_eps = epsilon
            self.rho = MIN_RHO + (MAX_RHO - MIN_RHO) * \
                math.exp(-LAMBDA * episode)

            # self.env.render()
            obs = self.env.reset(reset_from_failed_state = True, env_instance = copy.deepcopy(self.env_to_reset))
            info = self.env.get_info()
            self.success_func = self.create_success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
            # Get location of novel item -> Run motion planner to go to that location
            # Store state, action and reward
            if self.new_item_in_the_world is not None and self.guided_policy:
                rand_e = np.random.uniform() # use the policy with some rho constant probability.
                if rand_e < self.rho:
                    policies = []
                    # self.new_item_in_the_world.add('crafting_table')
                    # self.new_item_in_the_world.add('tree_log')
                    # print ("self.new_items_in_the_world = ", self.new_item_in_the_world)
                    for item in self.new_item_in_the_world:
                        action = "approach " + str(self.env.block_in_front_str) + " " + str(item)
                        policy = self.run_motion_planner(action)
                        policies.append(policy)
                        # now execute the sub-plan
                    policy = policies[0] # randomly choose a policy from all the guided policies
                    for j in range (len(policy)):
                        # env.render()
                        obs, reward, done, info = self.step_env(orig_obs = obs, info = info, done = done, action = self.env.actions_id[policy[j]])
                        episode_timesteps += 1
                        rew = -1
                        self.learning_agent.give_reward(rew)
                        reward_per_episode += rew
                        done = False

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
                        if episode % 200 == 0:
                            print ("EP >> {}, Timesteps >> {},  Rew >> {}, done = {}, done rate (20) = {}, eps = {}".format(episode, episode_timesteps, reward_per_episode, done, np.mean(self.Done[-20:]), round(self.learning_agent._explore_eps, 3)))
                            print("\n")
                        self.Done.append(1)
                        self.R.append(reward_per_episode)
                        if episode > 70:
                            if np.mean(self.R[-20:]) > 970: # check the average reward for last 70 episodes
                                # for future we can write an evaluation function here which runs a evaluation on the current policy.
                                if  np.sum(self.Done[-20:]) > 19: # and check the success percentage of the agent > 80%.
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
                        if episode % 200 == 0:
                            print ("EP >> {}, Timesteps >> {},  Rew >> {}, done = {}, done rate (20) = {}, eps = {}".format(episode, episode_timesteps, reward_per_episode, done, np.mean(self.Done[-20:]), round(self.learning_agent._explore_eps, 3)))
                            # print ("EP >> {}, Timesteps >> {},  Rew >> {} done = {} done rate (20) = {}".format(episode, episode_timesteps, reward_per_episode, done, np.mean(self.Done[-20:])))
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

        if self.mode == 'exploration' and action is None:
            action = self.learning_agent.process_step(obs, True)
        elif self.mode is not 'exploration' and action is None: # self.mode is exploitation -> greedy policy used
            action = self.learning_agent.process_step(obs,False)
        elif action is not None:
            action = self.learning_agent.process_step(obs, False, action)
        
        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action) 
        # self.env.render()
        # if action == self.env.actions_id['Spray']:
        #     print ("used action = ", self.env.actions_id['Spray'])
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
        self.learning_agent.load_model(novelty_name=novelty_name, operator_name=self.failed_action)
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


    def run_motion_planner(self, action):
        # Instantiation of the AStar Planner with the 
        # ox = 
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        sx = self.env.agent_location[1]
        sy = self.env.agent_location[0]
        so = self.env.agent_facing_str
        # print ("agent is at {}, {} and facing {}".format(sy, sx, so))
        binary_map = copy.deepcopy(self.env.map)
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

        if loc2 in self.env.items:
            locs = np.asarray((np.where(self.env.map == self.env.items_id[loc2]))).T
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
        # print("Goal location: {} {}".format(gx_, gy_) )
        # print ("rxs and rys generated from the plan = {} {}".format(rxs, rys))
        sx, sy, plan = astar_operator.generateActionsFromPlan(sx, sy, so, rxs, rys, ro)
        return plan


if __name__ == '__main__':
    failed_action = None
    learn = Learner()
    # learn.play_learned_policy()