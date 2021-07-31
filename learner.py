'''
Author: Shivam Goel
Email: goelshivam1210@gmail.com
'''

import time
import numpy as np
import math
import tensorflow as tf
import os

from NGLearner_util import PolycraftDynamicsChecker, get_create_success_func_from_predicate_set, default_param_dict, GridworldMDP, add_reset_probas, informed_random_action, check_permissible, update_reset_probas, reset_to_interesting_state
from learning.dqn import DQNLambda_Agent
from polycraft_tufts.rl_agent.dqn_lambda.learning.utils import make_session

STEPCOST_PENALTY = 0.012

class Learner:
    def __init__(self, failed_action, env, novelty_flag=False) -> None:
        self.encounter_novelty_flag = novelty_flag
        # self.env = PolycraftMDP(env, False, render=False)
        # self.env.run_SENSE_ALL_and_update()
        # self.env.generate_obs_action_spaces()
        self.env = GridworldMDP(env, False, render=True)
        if failed_action == "Break":
            failed_action = "Break tree_log"
        self.failed_action = failed_action

        #TODO: Also (or just) pass in failed effects with action using operator definitions,
        #      using hardcoded map for now
        operator_map = {
            "approach crafting_table tree_log": ['near tree_log'],
            "approach tree_tap crafting_table": ['near tree_log'],
            "approach air tree_log": ['near tree_log'],
            "approach air crafting_table": ['near tree_log'],
            "approach minecraft:crafting_table": ['near crafting_table'],
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

        self.dynamics_checker = PolycraftDynamicsChecker(self.env, self)
        self.dynamics_checker.update_from_failure(self.failed_action, {'cause': 'planning'})

        # Semi-impermissible actions that would leave us in unsolvable states if allowed pre-novelty
        # EW: do we want to use this or not?
        self.impermissible_actions = ['break_last_tree', 'craft_unnecessary_stick', 'craft_unnecessary_tap',
                                      'craft_new_recipe', 'placed_tap', 'extracted_rubber', 'broke_tap']

        self.reset_near_values = None
        self.reset_select_values = None
        self.updated_spaces = False
        self.can_motion_plan = True
        self.can_trade_plan = False

        add_reset_probas(self, exploration=True)


        self.learn_state()

        # self.novelty_recovery = NoveltyRecoveryAgent(self.env, encountered_novelty= self.encounter_novelty_flag)

    def reset_trial_vars(self):
        self.last_action = None
        self.last_obs = None
        self.mode = 'exploration'
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

    def learn_state(self):
        self.reset_trial_vars()
        return self.run_trial(300)

        ## Perform learning until successfully completing the task N times in a row
        # consecutive_successes = 0
        # while consecutive_successes < 10:
        #     success = self.run_trial(300)
        #     consecutive_successes = consecutive_successes + 1 if success else 0
        # return

    def learn_policy(self):
        pass

    def run_trial(self, time_per_round):
        self.env.run_SENSE_RECIPES_and_update()
        self.env.run_SENSE_ALL_and_update('NONAV')
        if not self.env.first_space_init:
            self.env.generate_obs_action_spaces()
        # self.reset_trial_variables()
        ep_success, relevant_outcome = self.learn_for_time(time_per_round)
        if relevant_outcome == 'recovered':
            self.mode = 'learning'
        return ep_success

    def learn_for_time(self, time_to_train, reset_every=1):
        start = time.time()
        longest_ep_time = 0
        ep_num = 0

        move_near_id, select_id = None, None
        self.success_func = self.create_success_func(self.env.observation(),self.env.get_info())
        ep_start_time = time.time()

        while (time_to_train - (time.time() - start) > longest_ep_time + 20):

            found_plannable, relevant_outcome = self.run_episode(50)

            # break on finding novel item type, need to reset learners and buffers
            if relevant_outcome == 6:
                return False, relevant_outcome

            # change if reset_every isn't 1 (really should be)
            update_reset_probas(self, move_near_id, select_id, relevant_outcome)

            if found_plannable:
                return True, relevant_outcome

            longest_ep_time = max(longest_ep_time, time.time() - ep_start_time)
            ep_num += 1

            # Reset agent to some 'interesting' state between 'episodes' (near obj, holding obj, ...)
            if ep_num % reset_every == 0:
                move_near_id, select_id = reset_to_interesting_state(self)
                if self.found_relevant_during_reset:
                    # move this
                    possible_outcomes = {'irrelevant': 0,
                                         'detrimental': 1,
                                         'beneficial': 2,
                                         'plannable': 3,
                                         'recovered': 4,
                                         'lost_progress': 5,
                                         'novel_item': 6,
                                         }
                    return True, possible_outcomes[self.last_outcome]

            ep_start_time = time.time()

        # Times up, did not manage to complete subgoal
        print(Fore.YELLOW + "Did not find plannable state")
        return False, relevant_outcome


    # Run single episode using exploration agent
    def run_episode(self, ep_t_limit):
        ep_t = 0
        done = False
        obs = self.env.observation()
        info = self.env.get_info()

        #TODO: update outcome list
        possible_outcomes = {'irrelevant': 0,
                             'detrimental': 1,
                             'beneficial': 2,
                             'plannable': 3,
                             'recovered': 4,
                             'lost_progress': 5,
                             'novel_item': 6,
                             }
        relevant_outcome = 0

        while True:
            obs, rew, found_plannable, info = self.step_env(orig_obs=obs, info=info, done=done)

            relevant_outcome = max(relevant_outcome, possible_outcomes[self.last_outcome])

            # break on encountering novel item type - need to reset learners and buffers
            if relevant_outcome == 6:
                return False, relevant_outcome

            ep_t += 1

            #Associated with 'impermissible' actions so we don't reset after placing
            #  a tap. Also need to return done on the correct time step
            if ep_t >= ep_t_limit - 2 and not self.placed_tap:
                done = True

            if (ep_t >= ep_t_limit - 1 or found_plannable) and not self.placed_tap:
                return found_plannable, relevant_outcome

            if self.env.game_over:
                print(Fore.YELLOW + "[run_exploration_episode] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                return self.env.game_success, None

    def step_env(self, action=None, orig_obs=None, info=None, done=False, store_transition=None, evaluate=False):
        #Cutting reset or motion planning execution short because we encountered a novelty or progressed unexpectedly
        if self.found_relevant_during_reset:
            if self.resetting_state:
                return orig_obs, 0, done, None
            elif self.motion_planning:
                return orig_obs, 0, done, None
            else:
                self.found_relevant_during_reset = False

        if self.env.game_over:
            print(Fore.YELLOW + "[step_env] Env has indicated that the game is over, stopping execution and resetting to start next trial")
            return orig_obs, 0, True, None

        # Notion of success in step here is different in this version
        # success: found_plannable in exploration, success_func->True in learning
        success = False

        if orig_obs is None:
            orig_obs = self.env.observation()
        if info is None:
            info = self.env.get_info()

        #EW: Removing general moveTO changes, not relevant for first batch of test cases
        #    See later if we want to reincorporate
        # if self.failed_action.split()[0] == 'moveTo':
        #     obs = self.reformat_moveTo_obs(orig_obs.copy(), self.operator_names[self.control_agent_ind].split()[1])
        obs = orig_obs.copy()

        self.learning_agent.store_obs(obs)

        permissible = True
        reason = None

        ## Choose to action ##
        if action is None:
            # Always informed random action until we've found reward, then weave in exploration agent
            if self.mode == 'exploration':
                if self.found_relevant_exp_state == 0:
                    action = informed_random_action(self, info)
                else:
                    # Do eps-greedy with greedy case being manually defined inform_random_action func
                    if np.random.random() < self.learning_agent.epsilon:
                        action = informed_random_action(self, info)
                        permissible = True
                    else:
                        action = self.learning_agent.get_action(obs, info, 0)
                        permissible, reason = check_permissible(self, action, info)

                    #TODO: Either remove completely or reincorporate checks to allow these actions over time
                    #Don't allow impermissible (irreversible) actions pre-novelty too early in exploration
                    if not permissible:
                        # don't consider placed_tap and extracted_rubber as impermissible actions in the same way
                        if reason == 'placed_tap' or reason == 'extracted_rubber' or reason == 'broke_tap':
                            print('Impermissible action due to tap/extract sequence on step {} with action {}'.format(reason, action))
                            action = informed_random_action(self, info)
                            print('choosing {} instead'.format(action))
                            permissible = True

            #TODO: Do we still want a distinction between exploration and learning modes?
            #In learning mode
            else:
                # eps_greedy same as case 2 above, but NEVER allow impermissible (updated notion of impermissible)
                epsilon = 0 if evaluate else self.learning_agent.epsilon

                if np.random.random() < epsilon:
                    action = informed_random_action(self, info)
                    permissible = True
                else:
                    action = self.learning_agent.get_action(obs, info, 0)
                    permissible, reason = check_permissible(self, action, info)

                if not permissible:
                    print(Fore.YELLOW + 'Learning agent chose impermissible action {} due to reason {} - not allowing it do so'.format(self.env.all_actions[action], reason))

                    # Give dummy effect to learner with negative reward to discourage this action
                    # Otherwise since it's never allowed the agent will continuously try to use it
                    #Don't want to penalize things during placed_tap
                    if self.learning_agent is not None and store_transition:
                        # What's an appropriate negative reward to give?
                        # Needs to be worse than that imposed by step_cost
                        self.learning_agent.store_effect(action, -100, False)
                        self.learning_agent.timesteps_trained += 1
                        self.learning_agent.check_update()

                        # store original obs back into agent again for actual effect update
                        self.learning_agent.store_obs(obs)

                    action = informed_random_action(self, info)
                    permissible = True

        #EW: remove this?
        #Execution trick: If we select the same action in a row without the observation changing,
        #   choose random instead with some probability (possible we may need to just try something
        #   multiple times so don't explicitly forbid)
        if self.env.all_actions[action] == self.last_action and np.all(orig_obs == self.last_obs):
            if np.random.random() < 0.5:
                action = informed_random_action(self, info)
                permissible = True

        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action)
        self.last_action = self.env.all_actions[action]
        self.last_obs = orig_obs.copy()

        # if game_over and success, store transition, else quit
        if self.env.game_over and not self.env.game_success:
            print(Fore.YELLOW + "[step_env] Env has indicated that the game is over and not successful, stopping execution and resetting to start next trial")
            return orig_obs, 0, True, None

        #TODO: replace this with just 'recovered'?
        #Check outcome of action for 'novelties' or progress
        outcome, material_diffs = self.dynamics_checker.check_for_novelty(info, self.env.all_actions[action], info_)
        self.last_outcome = outcome

        if outcome == 'novel_item':
            self.novel_item_encountered = True
            return None, None, False, None

        ## Respond to action ##
        # In exploration mode, want to check if novelty occurred, and if so whether it was helpful or not
        if self.mode == 'exploration':
            if outcome != 'irrelevant':
                self.found_relevant_exp_state += 1

            # Care first if we've found a plannable state or recovered the exact effects of a lost operator
            if outcome == 'plannable' or outcome == 'recovered':
                # # If we're exploring towards moveTo, we're already learning moveTo on the side
                # # If we ever explore towards a different operator, we don't want to reward moveTo
                # if outcome == 'recovered' and self.failed_action.startswith('moveTo'):
                #     rew = -1
                # else:
                rew = 1000
                success = True

                #EW: would remove impermissible from set here if using
                # if impermissible action occured at any point during this episode, but we found a plannable state, no longer impermissibe

            elif outcome == 'detrimental':
                rew = -100
            elif outcome == 'irrelevant':
                #EW: Do we want these crafted reward?
                if self.env.all_actions[action].split()[0] == 'CRAFT' or self.env.all_actions[action].split()[0] == 'SELECT_ITEM' or self.env.all_actions[action].startswith('PLACE'):
                    rew = -50
                else:
                    rew = -1
            elif outcome == 'beneficial':
                rew = 200

            #EW: 'recovered_pending' case should never happen because using NG (used when entity isn't
            #    sucked up upon break)
            else:
                print('ERROR, unknown outcome')
                quit()

            rew -= round(info_['last_step_cost'] * STEPCOST_PENALTY)

        # In learning mode we simply want to satisfy the goal effect set
        elif self.mode == 'learning':
            planning_success = False

            # Again, not same notion of success
            if outcome == 'plannable' or outcome == 'recovered':
                planning_success = True

            # Check if we've satisfied the operator's effect set
            success = self.success_func(obs2, info_)
            # If we satisfy original operator's effect set, we have recovered
            if success:
                outcome == 'recovered'
                self.last_outcome = 'recovered'
                planning_success = True

            #EW: Would intermittently check if we can replan using planner here

            # Plannable states outside of our effect set should also count as successes
            if planning_success and not success:
                print(Fore.CYAN + "Found plannable state that does not fall into goal effect set for our operator")
                success = planning_success
            if success:
                rew = 1000
            # 'beneficial' here should really only line up with what we found during exploration of the
            #  successful trial that was novel
            elif outcome == 'beneficial' and not RESTRICT_BENEFICIAL:
                rew = 200
            #EW: do we want domain specific crafted rewards?
            else:
                if self.env.all_actions[action].split()[0] == 'CRAFT' or self.env.all_actions[action].split()[0] == 'SELECT_ITEM' or self.env.all_actions[action].startswith('PLACE'):
                    rew = -50
                else:
                    rew = -1
            rew -= round(info_['last_step_cost'] * STEPCOST_PENALTY)

        info = info_

        if self.learning_agent is not None and store_transition:
            self.learning_agent.store_effect(action, rew, success or done)
            self.learning_agent.timesteps_trained += 1
            self.learning_agent.check_update()

            if done or success:
                self.learning_agent.eps_trained += 1
                self.learning_agent.update_epsilon()

        #EW: moveTo learning checks for opened paths would be here

        #Indicate to stop execution during reset or motion planning
        if self.resetting_state and success:
            print(Fore.LIGHTYELLOW_EX + 'Either moved forwards or backwards during reset_to_interesting_state, stopping reset execution to take note of effect')
            self.found_relevant_during_reset = True
        elif self.motion_planning and success:
            print(Fore.LIGHTYELLOW_EX + 'Either moved forwards or backwards during motion_planning, stopping reset execution to take note of effect')
            self.found_relevant_during_reset = True
        return obs2, rew, success, info


if __name__ == '__main__':
    failed_action = None
    # env =  

    learn = Learner()


    pass




        
