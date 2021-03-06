'''
Author: Shivam Goel
Email: goelshivam1210@gmail.com
'''
import time
import numpy as np
import math
import os
import copy
import subprocess
import matplotlib.pyplot as plt
 
from brain import *
from generate_pddl import *
from NGLearner_util_v2 import get_create_success_func_from_predicate_set, RewardFunctionGenerator
from generate_pddl import *
from SimpleDQN import *
from utils import AStarOperator, AStarPlanner
from params import *

action_map = {'moveforward':'Forward',
        'turnleft':'Left',
        'turnright':'Right',
        'break': 'Break',
        'place':'Place_tree_tap',
        'extractrubber':'Extract_rubber',
        'craftplank': 'Craft_plank',
        'craftstick':'Craft_stick',
        'crafttree_tap': 'Craft_tree_tap',
        'craftpogo_stick': 'Craft_pogo_stick',
        'select': 'Select'}

class Learner:
    def __init__(self, failed_action, env, plan, actions_bump_up, guided_action = False,\
                 new_item_in_the_world=None, guided_policy=False, novelty_flag=False, exploration_mode = 'uniform') -> None:

        self.env_to_reset = copy.deepcopy(env) # copy the complete environment instance
        self.env = env
        self.failed_action = failed_action
        self.actions_bump_up = actions_bump_up
        self.new_item_in_the_world = new_item_in_the_world
        self.guided_policy = guided_policy
        self.learned_failed_action = False
        self.guided_action = guided_action
        self.exploration_mode = exploration_mode
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
            "Break": ['increase inventory_log 1'],
            "Craft_plank": ['increase inventory plank 1'],
            "Craft_stick": ['increase inventory stick 1'],
            "Craft_tree_tap": ['increase inventory tree_tap 1'],
            "Craft_pogo_stick": ['increase inventory pogo_stick 1'],
            "Extract_rubber": ['increase inventory rubber 1'],
        }

        self.desired_effects = operator_map[failed_action]
        print("desired effects: ", self.desired_effects)
        self.create_success_func = get_create_success_func_from_predicate_set(self.desired_effects)
        self.reward_funcs = RewardFunctionGenerator(plan, self.failed_action)

        self.fsuccess_func = None
        # self.clever_exploration_flag = False
        # print ("self.guided_action = {}  self.guided_policy = {}".format(self.guided_action, self.guided_policy))
        # if self.guided_action and self.guided_policy:
        #     self.clever_exploration_flag = True

        agent = SimpleDQN(int(env.action_space.n),int(env.observation_space.shape[0]),
                        NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,
                        self.guided_action, self.actions_bump_up,
                        self.env.actions_id, random_seed, self.guided_policy, self.exploration_mode)
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

    def learn_policy(self, novelty_name, learned_policies_dict, failed_action_set, transfer=None):
        self.reset_trial_vars()
        self.novelty_name = novelty_name
        self.learned_policies_dict = learned_policies_dict
        self.failed_action_set = failed_action_set

        self.R = [] # storing rewards per episode
        self.Done = [] # storing goal completion per episode
        self.Steps = [] # storing the number of steps in each episode
        self.Epsilons = [] # storing the epsilons in the list for each episode
        self.Rhos = []  # storing the rhos for each episode
        data = [self.R, self.Done, self.Steps, self.Epsilons, self.Rhos]

        self.R_eval_mean = [] # storing rewards per episode
        self.Done_eval_mean = [] # storing goal completion per episode
        self.Steps_eval_mean = [] # storing the number of steps in each episode
        self.Episode_eval = [] # storing the number of episodes during evaluations
        data_eval = [self.R_eval_mean, self.Done_eval_mean, self.Steps_eval_mean, self.Episode_eval]
        self.learned_failed_action = self.run_episode(transfer, novelty_name)

        if self.learned_failed_action:
            self.learning_agent.save_model(novelty_name, self.failed_action)
            return True, data, data_eval
        else:
            return False, data, data_eval

    # Run episodes for certain time steps.
    def run_episode(self, transfer=None, novelty_name=None):
        done = False
        obs = self.env.get_observation()
        info = self.env.get_info()

        possible_outcomes = {
            'plannable':0,
            'unplannable':1
        }
        if transfer is not None: # here we transfer the weights to jumpstart the agent
            print("Loading the transferred policy")
            self.learning_agent.load_model(novelty_name = novelty_name, operator_name = self.failed_action, transfer_from = transfer)

        for episode in range(MAX_EPISODES):
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
            self.reward_funcs.store_init_info(info) # Reward success func storing the initial info
            
            # evaluating 
            if episode > 0 and episode % EVAL_INTERVAL == 0: # lets evaluate the current learnt policy, used for plot generation.
                R_eval = []
                Done_eval = []
                Steps_eval = []
                for episode_per_eval in range(EPS_TO_EVAL):
                    self.mode = 'exploitation'
                    timesteps, done = self.evaluate_policy()
                    Steps_eval.append(timesteps)
                    if done:
                        Done_eval.append(1)
                        R_eval.append(1000-timesteps)
                    else:
                        Done_eval.append(0)
                        R_eval.append(0-timesteps)
                # save all the trials mean score
                self.Episode_eval.append(episode)
                self.R_eval_mean.append(np.mean(R_eval))
                self.Done_eval_mean.append(np.mean(Done_eval))
                self.Steps_eval_mean.append(np.mean(Steps_eval))
                # self.Episode_eval.append(episode)
                self.mode = 'exploration'
                self.learning_agent.reset()
                print ("Evals: Rew(Avg{}) >> {} done >> {}".format(EPS_TO_EVAL, np.mean(R_eval), np.mean(Done_eval)))
                continue
            # clever exploration part (Guided policy)
            # Get location of novel item -> Run motion planner to go to that location
            # Store state, action and reward
            if self.new_item_in_the_world is not None and self.guided_policy:
                rand_e = np.random.uniform() # use the policy with some rho constant probability.
                if rand_e < self.rho:
                    policies = []
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

            goal_reached = False
            while True:
                obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done, timestep = episode_timesteps)
                episode_timesteps += 1
                self.is_success_met = self.success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
                self.reward_success_met = self.reward_funcs.check_success(info)
                if self.is_success_met or self.reward_success_met:
                    # print("Success met")
                    if self.reward_funcs.catastrophic_condition_met(self.env):    
                        # print("Reached goal \n")
                        done = True
                        goal_reached = True
                        rew = POSITIVE_REINFORCEMENT
                    else:
                        # print ("catastrophy met \n")
                        done = True # this done is to come out of the episode loop
                        goal_reached = False # this is used to save whether the DONE is true or not. Done means when the agent reached the goal.
                        rew = NEGATIVE_REINFORCEMENT # penalizing for reaching an unplannable state
                else:
                    done = False
                    rew = NORMAL_REINFORCEMENT
                # agent.give_reward(reward)
                self.learning_agent.give_reward(rew)
                # here we update the network with the observation and reward
                reward_per_episode += rew # save reward

                if done or episode_timesteps > MAX_TIMESTEPS:
                        
                    self.learning_agent.finish_episode()
                    self.learning_agent.reset_action_counter() 
                    if episode % UPDATE_RATE == 0:
                        self.learning_agent.update_parameters()

                    if done:
                        if episode>0 and episode%PRINT_EVERY == 0:
                            print ("{}--EP >>{}, steps>>{},  Rew>>{}, done({})>>{}, eps>>{} rho>>{} ".format(self.novelty_name, episode, episode_timesteps, reward_per_episode, NO_OF_DONES_TO_CHECK, np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]), round(self.learning_agent._explore_eps, 3), round(self.rho, 3)))
                            print("\n")
                        self.Epsilons.append(self.learning_agent._explore_eps)
                        self.Rhos.append(self.rho)
                        self.Steps.append(episode_timesteps)
                        if goal_reached:
                            self.Done.append(1)
                        else:
                            self.Done.append(0)
                        self.R.append(reward_per_episode)
                        if episode > NO_OF_EPS_TO_CHECK:
                            if np.mean(self.R[-NO_OF_EPS_TO_CHECK:]) > SCORE_TO_CHECK and np.mean(self.R[-10:]) > SCORE_TO_CHECK: # check the average reward for last 70 episodes
                                # for future we can write an evaluation function here which runs a evaluation on the current policy.
                                if  np.sum(self.Done[-NO_OF_DONES_TO_CHECK:]) > NO_OF_SUCCESSFUL_DONE and np.mean(self.Done[-10:]) > NO_OF_SUCCESSFUL_DONE/NO_OF_DONES_TO_CHECK: # and check the success percentage of the agent > 80%.
                                    if abs(np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]) - np.mean(self.Done[-10:])) < 0.05:
                                        print ("The agent has learned to reach the subgoal")
                                        # plt.show()
                                        return True
                        break
                    elif episode_timesteps >= MAX_TIMESTEPS:
                        if episode>0 and episode%PRINT_EVERY == 0:
                            print ("{}--EP >>{}, steps>>{},  Rew>>{}, done({})>>{}, eps>>{} rho>>{} ".format(self.novelty_name, episode, episode_timesteps, reward_per_episode, NO_OF_DONES_TO_CHECK, np.mean(self.Done[-NO_OF_DONES_TO_CHECK:]), round(self.learning_agent._explore_eps, 3), round(self.rho, 3)))
                            print("\n")
                        self.Epsilons.append(self.learning_agent._explore_eps)
                        self.Rhos.append(self.rho)
                        self.Steps.append(episode_timesteps)
                        self.Done.append(0)
                        self.R.append(reward_per_episode)
                        break
        return False

    def step_env(self, action=None, orig_obs=None, info=None, done=False, store_transition=None, evaluate=False, timestep = 0):

        if orig_obs is None:
            orig_obs = self.env.get_observation()
            print(orig_obs.shape)
        if info is None:
            info = self.env.get_info()

        obs = orig_obs.copy()

        # self.learning_agent.store_obs(obs)
        # print ("self.actions_bump_up = ", self.actions_bump_up)

        if self.mode == 'exploration' and action is None:
            action = self.learning_agent.process_step(obs, True, timestep=timestep)
        elif self.mode is not 'exploration' and action is None: # self.mode is exploitation -> greedy policy used
            action = self.learning_agent.process_step(obs,False, timestep=timestep)
        elif action is not None:
            action = self.learning_agent.process_step(obs, False, action)
        
        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action) 
        # self.env.render()
        self.last_obs = orig_obs.copy()
        info = info_

        return obs2, action, done, info

    def play_learned_policy(self, env, novelty_name, operator_name):
        # self.env.render()
        self.env = env
        self.mode = 'exploitation' # want to act greedy
        self.is_success_met = False        
        done = False
        obs = self.env.get_observation()
        info = self.env.get_info()
        self.learning_agent.load_model(novelty_name=novelty_name, operator_name=self.failed_action)
        episode_timestep = 0
        self.success_func = self.create_success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
        self.reward_funcs.store_init_info(info) 

        while True:
            
            obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)
            # self.env.render()
            self.is_success_met = self.success_func(self.env.get_observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
            self.reward_success_met = self.reward_funcs.check_success(info)

            episode_timestep += 1 
            if self.is_success_met or self.reward_success_met:
                done = True
                if self.reward_success_met:
                    return True, True
                else:
                    return True, False
            if episode_timestep >= 300:
                done = False
                return False, False

    def evaluate_policy(self):
        # evaluate the policy while learning.    
        obs = self.env.reset()
        generate_prob_pddl("PDDL", self.env)
        plan, game_action_set = self.call_planner("domain", "problem", self.env) # get a plan
        is_success = self.execute_plan_before_learnt(game_action_set)
        # if is_success:
        #     print("plan = ",game_action_set)
        return self.env.step_count, is_success 

    def execute_plan_before_learnt(self, plan):
        i = 0
        while (i < len(plan)):
            sub_plan = []
            if plan[i] == self.failed_action: # the action we are evaluating, essentially we will play the policy in this
                obs, reward, done, info = self.env.step(self.env.actions_id[plan[i]])
                if info['result']==False:
                    # print ("I am playing policy")
                    result, is_reward_met = self.play_policy() # this is called in learner_v2 since we want to use the learner instance
                    if result == False:
                        return False
                    else:
                        i+=1
                        if is_reward_met: # When the result is true, we check if the success was met through the reward_func, and if yes, remove the actions from plan that generate the req obj
                            for action_to_remove in self.reward_funcs.actions_that_generate_objects_required: # now we remove the actions for those objects
                                try:
                                    while True:
                                        plan.remove(action_to_remove)
                                except ValueError:
                                    pass                            
                else: # action was successfully executed
                    i+=1
                # play the policy that is being learned (No need to load model).
                # 2 novelties -> Scrape plank and FCT easy
                # 1: break tree log -> Scrape plank break tree log
            elif plan[i] is not self.failed_action and plan[i] in self.failed_action_set: # cases when we already have a learned policy for the action.
                #Load the other learned policy and play
                obs, reward, done, info = self.env.step(self.env.actions_id[plan[i]])
                if info['result']==False:
                    played, is_reward_met = self.learned_policies_dict[plan[i]].play_learned_policy(self.env, novelty_name=self.novelty_name, operator_name=plan[i]) # returns whether the policy was successfully played or not            
                    if played == False:
                        return False
                    else:
                        i+=1
                        if is_reward_met: # When the result is true, we check if the success was met through the reward_func, and if yes, remove the actions from plan that generate the req obj
                            for action_to_remove in self.reward_funcs.actions_that_generate_objects_required: # now we remove the actions for those objects
                                try:
                                    while True:
                                        plan.remove(action_to_remove)
                                except ValueError:
                                    pass
                else:
                    i += 1                                                    
            elif 'approach' in plan[i]: # when we need motion planner for the approach actions.
                sub_plan = self.run_motion_planner(plan[i])
                for j in range (len(sub_plan)):
                    obs, reward, done, info = self.env.step(self.env.actions_id[sub_plan[j]])
                    # self.env.render()
                    if done:
                        if self.env.inventory_items_quantity[self.env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                            return True
                i +=1
            else:
                obs, reward, done, info = self.env.step(self.env.actions_id[plan[i]])
                i+=1
                if info['result'] == False:
                    return False
                if done:
                    if self.env.inventory_items_quantity[self.env.goal_item_to_craft] >= 1: # success measure(goal achieved)
                        return True
                

    def call_planner(self, domain, problem, env):
        '''
            Given a domain and a problem file
            This function return the ffmetric Planner output.
            In the action format
        '''
        self.pddl_dir = "PDDL"
        run_script = "Metric-FF-v2.1/./ff -o "+self.pddl_dir+os.sep+domain+".pddl -f "+self.pddl_dir+os.sep+problem+".pddl -s 0"
        output = subprocess.getoutput(run_script)
        plan, game_action_set = self._output_to_plan(output)
        return plan, game_action_set

    def _output_to_plan(self, output):
        '''
        Helper function to perform regex on the output from the planner.
        ### I/P: Takes in the ffmetric output and
        ### O/P: converts it to a action sequence list.
        '''

        ff_plan = re.findall(r"\d+?: (.+)", output.lower()) # matches the string to find the plan bit from the ffmetric output.
        # print ("ffplan = {}".format(ff_plan))
        action_set = []
        for i in range (len(ff_plan)):
            if ff_plan[i].split(" ")[0] == "approach":
                action_set.append(ff_plan[i])
            elif ff_plan[i].split(" ")[0] == "select":
                to_append = ff_plan[i].split(" ")
                sep = "_"
                to_append = sep.join(to_append).capitalize()
                action_set.append(to_append)

            else:
                action_set.append(ff_plan[i].split(" ")[0])

        if "unsolvable" in output:
            print ("Plan not found with FF! Error: {}".format(
                output))
        if ff_plan[-1] == "reach-goal":
            ff_plan = ff_plan[:-1]
        
        # print ("game action set  = {}".format(action_set))
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)
        # print ("game action set = {}".format(game_action_set))
        for i in range(len(game_action_set)):
            if game_action_set[i].split(" ")[0] != "approach" and game_action_set[i].split("_")[0] != "Select":
                game_action_set[i] = action_map[game_action_set[i]]
        # print ("game action set = {}".format(game_action_set))
        for i in range(len(game_action_set)):
            if game_action_set[i] in action_map:
                game_action_set[i] = self.env.actions_id[game_action_set[i]]
        # print (game_action_set)
        return action_set, game_action_set

    def play_policy(self):
        # this runs the policy to be evaluated, while the learning is going on.
        self.is_success_met = False        
        done = False
        obs = self.env.get_observation()
        info = self.env.get_info()
        # self.learning_agent.load_model(novelty_name=novelty_name, operator_name=self.failed_action)
        episode_timestep = 0
        self.success_func = self.create_success_func(obs,info) # self.success_func returns a boolean indicating whether the desired effects were met
        self.reward_funcs.store_init_info(info) 
        while True:            
            obs, action, done, info = self.step_env(orig_obs=obs, info=info, done=done)
            # self.env.render()
            self.is_success_met = self.success_func(self.env.get_observation(),self.env.get_info()) # self.success_func returns a boolean indicating whether the desired effects were met
            self.reward_success_met = self.reward_funcs.check_success(self.env.get_info())

            episode_timestep += 1
            if self.is_success_met or self.reward_success_met:
                done = True
                if self.reward_success_met:
                    return True, True
                else:
                    return True, False
            if episode_timestep >= 300:
                done = False
                return False, False

    def run_motion_planner(self, action):
        # Instantiation of the AStar Planner  
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
        sx, sy, plan = astar_operator.generateActionsFromPlan(sx, sy, so, rxs, rys, ro)
        return plan

if __name__ == '__main__':
    failed_action = None
    learn = Learner()
    # learn.play_learned_policy()