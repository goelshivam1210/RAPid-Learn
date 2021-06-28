import time
import numpy as np
import math
import tensorflow as tf
import os
import socket
from polycraft_tufts.rl_agent.dqn_lambda.envs.polycraft_mdp import PolycraftMDP
from polycraft_tufts.rl_agent.dqn_lambda.learning.dqn import DQNLambda_Agent
from polycraft_tufts.rl_agent.dqn_lambda.config.default_config import default_param_dict
from polycraft_tufts.rl_agent.dqn_lambda.learning.utils import make_session
from polycraft_tufts.rl_agent.dqn_lambda.detectors import get_create_success_func_from_predicate_set, get_create_success_func_from_failed_operator, PolycraftDynamicsChecker, get_world_quant, get_inv_quant, get_entity_quant
from polycraft_tufts.rl_agent.dqn_lambda.utils import AStarPlanner, HardcodedOperator, AStarOperator, CraftingOperator, ExtractRubberOperator
from polycraft_tufts.utils.utils import recv_socket_data
from polycraft_tufts.rl_agent.dqn_lambda.config.argparse_v2 import args

# for colored print statements
from colorama import Fore, init

init(autoreset=True)

USING_TOURNAMENT_MANAGER = args['manager']
CONNECT_TO_TRADE = args['trade']
MAX_STEP_COST = args['stepcost']
TIME_PER_ROUND = args['time']
HOST = args['host']
TRADE_PORT = args['trade_port']
ENV_PORT = args['env_port']
TASK_DIR = args['jsons_dir']
STEPCOST_PENALTY = args['stepcost_penalty']
RESTRICT_BENEFICIAL = args['restrict_beneficial']
# RESTRICT_ACTIONS = args['restrict']

# if connecting to TRADE, then we don't need to handle any NG details
if args['env'] == 'novelgridworld' and not CONNECT_TO_TRADE:
    ##Standalone NG
    import gym
    import gym_novel_gridworlds
    from gym_novel_gridworlds.novelty_wrappers import inject_novelty
    from polycraft_tufts.rl_agent.dqn_lambda.envs.novelgridworld_standalone.novelgridworld_mdp import GridworldMDP

#TODO: complete integration with planner such that novelty detection is active during RL exploration, goal
#   is to update belief such that when possible RL can be discarded and replaced with immediately optimal
#   plans upon discovery of the solution to the novelty (rather than completely learning the policy which may
#   be very difficult)
class DummyPlanningAgentV2:
    def __init__(self, env, operator_map, novel_items=[]):
        self.env = env
        self.operator_map = operator_map
        self.learning_operators = []
        self.failed_operators = []
        self.novelty_recovery_agent = None
        self.updated_due_to_recipe = False
        self.current_res_cp = None
        self.current_subplan_step = None
        self.detrimental_res_cps = []

        #Overall idea - split task into subtasks based on progress in resource space, each with their own original
        #   subplans. Update our notion of what the execution path should be based off of jumping multiple steps,
        #   having trouble progressing past a certain step, or falling back steps
        #When integrated with planner, can use planner to more generally replan and indicate which subgoal we are
        #   currently interested in

        # Same operators
        # new resource checkpoints:                                                       SubPlan applied:
        # 0 log's worth                                                                [moveTo log, break log]
        # 1 log's worth                                                                [moveTo log, break log]
        # 2 log's worth + min 1 tree left                                              [moveTo CT, craft tap]
        # 1 tree tap + min 1 tree left                                                 [moveTo log, extractRubber]
        # 1 rubber + min 1 tree left                                                   [moveTo log, break log]
        # 1 rubber + 1 log worth                                                       [moveTo CT, craft pogo]
        # 1 pogostick                                                                  Done
        # Rank in order of relevance and check going up the list to determine what to do
        self.resource_checkpoint_subplans = [
            # 0 logs worth in inv
            ['moveTo minecraft:log', 'break minecraft:log'],
            # 1 logs worth in inv
            ['moveTo minecraft:log', 'break minecraft:log'],
            # 2 logs worth in inv, min 1 tree left in world
            ['moveTo minecraft:crafting_table', 'craft polycraft:tree_tap'],
            # 1 tree tap + min 1 tree left (+ 1 log in inv optional)
            ['moveTo minecraft:log', 'extractRubber'],
            # 1 rubber + min 1 tree left
            ['moveTo minecraft:log', 'break minecraft:log'],
            # 1 rubber + 1 log worth
            ['moveTo minecraft:crafting_table', 'craft polycraft:wooden_pogo_stick'],

            ##Valid checkpoints not on original plan path
            # 3 logs in inv, min 1 tree left in world -> irrelevant check because would require same operator as 2 logs in inv min 1 tree left in world
            # 1 tree tap 1 log in inv -> irrelevant check because would require same operator as 1 tree tap 1 log in world
        ]
        self.init_resource_checkpoint_subplans = self.resource_checkpoint_subplans.copy()
        self.cps_attempts_without_progress = np.zeros(len(self.resource_checkpoint_subplans))
        self.operators_failed_per_cp = [[], [], [], [], [], []]
        # Allow one consecutive failure due to hardcoded operators per cp.
        # Due to rare bugs (mainly in polycraft), allow each operator two chances to
        #   succeed, if they fail twice in a row, consider them unusable
        #   (E.g. have found that sometime a log entity drops from a tree, hits a fence,
        #       bounces away from the agent, and isn't caught in the entity sense.
        #       Makes it seem like no log came out when that wasn't the case)
        # Additionally, at any step check if we can replan with the original operator and use it
        #   (in case novelties are not uniform and may be on a case by case basis)
        self.failed_hardcoded_once_cps = np.zeros(
            (len(self.resource_checkpoint_subplans), len(self.resource_checkpoint_subplans[0])))
        self.advanced_cp_during_moveTo = np.zeros(len(self.resource_checkpoint_subplans))

        # If we get to a res cp N times without progressing past that, discard that notion
        # of a cp and negatively reward that
        # Mirrors stepping back in plan execution to explore from an earlier state
        self.max_res_cp = 0
        self.cp_attempts_without_progress = 0

    def _get_operator(self, operator_str):
        try:
            operator = self.operator_map[operator_str]
            return operator
        # This should never be the case if everything is initialized properly
        except KeyError:
            print(Fore.RED + "ERROR: No such operator exists: ", operator_str)
            quit()

    def add_learner_connection(self, novelty_recovery_agent):
        self.novelty_recovery_agent = novelty_recovery_agent

    def add_relearning_operator(self, operator_str):
        self.learning_operators.append(operator_str)

    # TODO: rename
    def reset_plan_step(self):
        self.current_res_cp = None
        self.current_subplan_step = 0
        # allow one consecutive failure due to hardcoded operators per cp
        self.failed_hardcoded_once_cps = np.zeros((len(self.resource_checkpoint_subplans), len(self.resource_checkpoint_subplans[0])))

        #TODO: Can likely discard this -> planner will get an instance of this object first
        #Before each trial, check if there is a novel object in recipe space, if so get an instance of it
        new_recipe_items = []
        #TODO: pull in current knowledge base from planner explicitly to make task independent, for now just
        #   mirroring planner's prenovelty knowledge
        #Any recipe we were not aware of prenovelty is novel
        for item in self.env.ingredients_quantity_dict:
            if item not in ['minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
                            'minecraft:crafting_table']:
                new_recipe_items.append(item)

        new_entity_items = []
        for entity in self.env.entities_location:
            if entity not in ['minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
                            'minecraft:crafting_table']:
                new_entity_items.append(entity)

        if len(new_recipe_items) > 0 or len(new_entity_items) > 0:
            if not self.novelty_recovery_agent.updated_spaces:
                print(Fore.CYAN + 'New item type detected, updating env spaces before episode execution')
                self.env.generate_obs_action_spaces(self.novelty_recovery_agent.novel_items)
                self.env.run_SENSE_ALL_and_update('NONAV')
                self.novelty_recovery_agent.updated_spaces = True
            # if len(new_recipe_items) > 0:
            print(Fore.CYAN + 'New item type has been added to the domain, checking if instance exists as an entity or in the inventory, otherwise attempting to use planner to craft an instance before execution if possible')
            if len(new_recipe_items) > 1:
                print(Fore.YELLOW + "Warning - more than one novel recipe goal item has been introduced, only exploring the first")
                print(Fore.LIGHTYELLOW_EX + "TODO: Explore each in turn or alternating if possible")
        #Try to obtain singular instance of everything new, if recipe and entity sets are different, then
        #   recipe should override entity -> we will have it in the inv at the time of calling it
        count = 0
        for item in new_recipe_items:
            print(Fore.CYAN + 'Checking to obtain new instance of novel recipe {} before plan execution'.format(item))
            self.obtain_instance(item, True)
            self.env.execute_action('SELECT_ITEM {}'.format(item))
            count += 1
            #If for some reason a huge amount of different types fill the board, pick up a few each round,
            #   deterministically, otherwise let the agent figure out which one matters
            if count > 4:
                break
        count = 0
        for item in new_entity_items:
            print(Fore.CYAN + 'Checking to obtain new instance of novel entity {} before plan execution'.format(item))
            self.obtain_instance(item, False)
            self.env.execute_action('SELECT_ITEM {}'.format(item))
            count += 1
            if count > 4:
                break

    def obtain_instance(self, item, craftable=False):
        info = self.env.get_info()
        in_inv = get_inv_quant(info, item) > 0
        as_entity = get_entity_quant(info, item) > 0
        if in_inv or as_entity:
            if in_inv:
                print(Fore.CYAN + 'Instance of new item type already exists in the inventory, not planning to obtain instance before plan execution')
            elif as_entity:
                if self.novelty_recovery_agent.can_motion_plan:
                    print(Fore.CYAN + 'Instance of new type already exists as entity in the world, attempting to pick up before plan execution')
                    # instead of adding a new operator, just go and pick up the item
                    new_operator = AStarOperator("pickUp {}".format(item), item,
                                                 ['increase inventory {} 1'.format(item)])
                    obs = self.env.observation()
                    planning_success = new_operator.reset(obs, info, self.env)
                    if planning_success:
                        action = new_operator.get_action()
                        while action is not None:
                            self.env.execute_action(self.env.all_actions[action])
                            # This should really never happen
                            if self.env.game_over:
                                print(
                                    Fore.YELLOW + "[obtain_instance] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                                return
                            action = new_operator.get_action()
                        obs = self.env.observation()
                        info = self.env.get_info()
                        success = new_operator.check_success(obs, info)

                        if not success:
                            print(Fore.LIGHTYELLOW_EX + 'Entity of new type exists in world, but A* failed')
                        else:
                            print(Fore.CYAN + 'Obtained instance of new entity')
                    else:
                        # TODO: Allow for moveTo operator learning toward entities if this is the case
                        print(Fore.LIGHTYELLOW_EX + 'Entity of new type exists in world, but we cannot plan to it, not picking up before plan execution')
                # TODO: Allow for moveTo operator learning toward entities if this is the case
                else:
                    print(Fore.YELLOW + 'Instance of new type exists as entity, but motion planning execution has failed previously, not picking up before plan execution')
        # Make planner craft the item - this should already be the case currently
        elif craftable:
            print('TODO: Novel recipe detected. Get TRADE planner to craft novel item recipe for us - this should already have occured')
            # if self.novelty_recovery_agent.sock is not None and self.novelty_recovery_agent.can_trade_plan:
            #     plan = self.novelty_recovery_agent.trade_plan_to_goal('[fluent_geq inventory {}} 1]'.format(new_recipe_items[0]))
            #     print(plan)
            #     #Might have to perform last craft action independently or to operator map
            #     # plan = ['moveTo minecraft:log']
            #     #Execute this plan before rest of plan
            #     if plan is not None:
            #         s = self.execute_plan_independent(600, plan)
            #     else:
            #         print(Fore.LIGHTYELLOW_EX+'TRADE planner could not plan to craft novel recipe, not attempting')

    def reset_all(self):
        print(Fore.CYAN + 'Resetting planner due to unexpected novel item type requiring learner reset')
        self.learning_operators = []
        self.failed_operators = []
        self.current_res_cp = None
        self.current_subplan_step = 0
        self.max_res_cp = -1
        self.resource_checkpoint_subplans = self.init_resource_checkpoint_subplans.copy()
        self.cps_attempts_without_progress = np.zeros(len(self.resource_checkpoint_subplans))
        self.detrimental_res_cps = []
        self.operators_failed_per_cp = [[], [], [], [], [], []]
        # allow one consecutive failure due to hardcoded operators per cp
        self.failed_hardcoded_once_cps = np.zeros((len(self.resource_checkpoint_subplans), len(self.resource_checkpoint_subplans[0])))
        self.advanced_cp_during_moveTo = np.zeros(len(self.resource_checkpoint_subplans))

        continue_execution = True
        print(Fore.LIGHTYELLOW_EX + 'TODO: If item is novel to planner, revert to explicit exploration')

    # plan based off of resource checkpoints after each sub-plan completion rather than each individual plan step
    def execute_resource_plan(self, time_to_run, independent=False):
        start = time.time()
        self.current_subplan_step = 0

        if self.env.game_over:
            print(Fore.YELLOW + "[execute_resource_plan] Env has indicated that the game is over, stopping execution and resetting to start next trial")
            return self.env.game_success, None, None, None

        #Plan and execute until task completeion, timeout, or stepcost limit
        while True:

            if self.env.game_over:
                return self.env.game_success, None, None, None

            # After each CP execution, replan from resulting CP
            if self.novelty_recovery_agent.last_res_cp is None:
                info = self.env.get_info()
                resource_step = self.get_current_resource_checkpoint(info)
            else:
                resource_step = self.novelty_recovery_agent.last_res_cp

            #Note: this block concerns itself with 'moveTo' vs other operators, but the overall idea is more
            #   about when we find that a subplan step can be skipped. It just so happens that moveTo is first
            #   in every subplan.
            if self.current_res_cp is not None and resource_step > self.current_res_cp:
                self.cps_attempts_without_progress[self.current_res_cp] = 0
                # If we jump forward a res step during relearning of a moveTo, modify that subplan to just be the getRes one
                #   Remember, don't want to use moveTo learners because they're actually shared between all moveTos
                #   But this shows that there's a way to get the item in a novel way, which we want to learn
                if self.resource_checkpoint_subplans[self.current_res_cp][self.current_subplan_step - 1].startswith('moveTo'):
                    # Doing same as with failure, giving it one pass. Very possible rare event resulting in replanning
                    #   made this occur even if 99% of the time we want to keep moveTo
                    # E.g. break fails because entity dropped, pick up entity during moveTO, don't want to ditch moveTo+break which works
                    self.advanced_cp_during_moveTo[self.current_res_cp] += 1
                    if self.advanced_cp_during_moveTo[self.current_res_cp] < 2:
                        print(Fore.LIGHTYELLOW_EX + 'Allowing res cp jump from moveTo once consecutively for this cp due to rare outcomes in polycraft, next occurance will ditch moveTo for general getRes learner')
                    else:
                        print(Fore.CYAN + 'Progressed res cp while executing or learning moveTO -> means theres another way to get a desired resource. Ditching moveTo and using general getRes learner for this cp')
                        self.resource_checkpoint_subplans[self.current_res_cp] = self.resource_checkpoint_subplans[self.current_res_cp][:self.current_subplan_step - 1] + self.resource_checkpoint_subplans[self.current_res_cp][self.current_subplan_step:]
                        # Kind of hacky, should always just be one operator left in the subplan, create learner from that one, not the original
                        self.novelty_recovery_agent.failure_info = {'operator': self.operator_map[self.resource_checkpoint_subplans[self.current_res_cp][-1]]}
                        # add notion that we are learning this new operator
                        self.operators_failed_per_cp[self.current_res_cp].append(self.resource_checkpoint_subplans[self.current_res_cp][-1])
                        self.failed_operators.append(self.resource_checkpoint_subplans[self.current_res_cp][-1])
                        self.novelty_recovery_agent.create_learner_from_explorer(True)
                        self.novelty_recovery_agent.num_exploration_failures = 0
                # Otherwise we succeeded on this CP not during moveTO
                # This means moveTo occurred, succeeded, and then the other operator succeeded
                # Worst case, moveTo is meaningless but possible, and all learning goes to the other operator anyway
                # Best case, we prevent case where we ditch moveTo when it's very useful
                # Reset notion of succeeding during moveTo
                else:
                    self.advanced_cp_during_moveTo[self.current_res_cp] = 0

                if resource_step > self.max_res_cp:
                    self.max_res_cp = resource_step

            if resource_step != self.current_res_cp:
                self.current_res_cp = resource_step
                self.current_subplan_step = 0

            # If we've achieved the last cp, we're done
            if resource_step == 6:
                return True, None, None, True

            step_plan = self.resource_checkpoint_subplans[resource_step]

            if self.current_subplan_step == len(step_plan):
                print(Fore.YELLOW + "Back in planning with subplan step exceeding length of current step plan, what happened?")
                print(Fore.LIGHTYELLOW_EX + "Operator may have indicated success while not progressing a res_cp, shouldn't ever be the case. Must be sensing issue, replanning")
                self.current_subplan_step = 0
                info = self.env.get_info()
                resource_step = self.get_current_resource_checkpoint(info)
                self.current_res_cp = resource_step
                continue

            print(Fore.BLUE + '\nOn resource step {} with plan {}'.format(resource_step, step_plan))

            init_plan_step = self.current_subplan_step

            #Perform each step of subplan in turn
            for i in range(init_plan_step, len(step_plan)):

                self.current_subplan_step = i + 1

                if (not USING_TOURNAMENT_MANAGER) and time_to_run - (time.time() - start) < 20:
                    print(Fore.RED + 'Ran out of time while executing preset plan')
                    return False, -1, None, False

                if i != init_plan_step:
                    # If executing plan and not finding novelty, techincally everything should be fine so we don't
                    # ever have to check the cp between successful subplan steps using the original impl
                    if self.novelty_recovery_agent.last_res_cp is None:
                        info = self.env.get_info()
                        resource_step = self.get_current_resource_checkpoint(info)
                        self.novelty_recovery_agent.last_res_cp = resource_step
                    else:
                        resource_step = self.novelty_recovery_agent.last_res_cp
                    if resource_step != self.current_res_cp:
                        print(resource_step)
                        print(self.current_res_cp)
                        print(self.novelty_recovery_agent.last_res_cp)
                        self.current_subplan_step -= 1
                        break

                operator_str = step_plan[i]
                if USING_TOURNAMENT_MANAGER:
                    print(Fore.BLUE + "\nexecuting plan step {}".format(operator_str))
                else:
                    print(Fore.BLUE + "\nexecuting plan step {} with time_left {}".format(operator_str, time_to_run - (time.time() - start)))

                # If we're learning this operator AND have failed this operator on this specific step before, then continue with learning
                # Otherwise try original operator until it fails, then switch to same learner
                if operator_str in self.learning_operators and operator_str in self.operators_failed_per_cp[self.current_res_cp]:

                    # Always try original moveTo first if failed previously due to planning failure -
                    #   possible some paths are blocked and some aren't
                    if operator_str.startswith('moveTo') and self.novelty_recovery_agent.can_motion_plan:
                        success, outcome, info = self.execute_operator(operator_str)
                        if self.env.game_over:
                            print(Fore.YELLOW + "[execute_resource_plan] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                            return self.env.game_success, None, None, None
                        if success:
                            print(Fore.CYAN + "Relearning operator {}, but could plan a path. Used original A* agent".format(operator_str))
                            # Using Operator Impl won't use step_env so need to update res cp
                            info_ = self.env.get_info()
                            rs = self.get_current_resource_checkpoint(info_)
                            if rs > resource_step:
                                self.novelty_recovery_agent.last_res_cp = rs
                                break
                            continue
                        elif outcome == 1:
                            print(Fore.CYAN + "Relearning operator {}, but could plan a path. Used original A* agent but execution failed".format(operator_str))
                            # Weird bug where block_in_front doesnt update accurately from sense_all, causes endless
                            #   loop where we think the original operator will work by placing a crafting table but
                            #   the block in front is actually occupied
                            # Two cases - in front of a breakable item unknowingly
                            #           - in front of an unbreakable item unknowingly
                            # Temp solution, try turn and break_block to see if things actually update correctly, so next iter it will work
                            if operator_str == 'moveTo minecraft:crafting_table':
                                self.env.step(self.env.actions_id['BREAK_BLOCK'])
                                self.env.step(self.env.actions_id['TURN -90'])
                            return success, outcome, info, False
                        # otherwise planning failure, this was already the case previously, go on with learner

                    # Otherwise apply learner for step operator
                    print(Fore.BLUE + "Attempting to apply new learner with failed operator's effects\n")
                    #The notion of 'independent' mode was meant for when the planner indicated something as
                    #   as plannable and returned a plan, but we didn't want to fully commit to that plan
                    #   as true in case execution failed due to knowledge discrepency in planner
                    #We are currently never using this mode as we aren't using the planner to indicate
                    #   'plannable' states or give plans for subgoals RL could use
                    #Goal is to be able to use planner to indicate plannable states to create and refine
                    #   effect sets for general operators, rather than using this sort of 'dummy planner'
                    if independent:
                        init_mode = self.novelty_recovery_agent.mode
                    self.novelty_recovery_agent.mode = 'learning'
                    success = self.novelty_recovery_agent.learn_for_time(time_to_run - (time.time() - start),operator_str)
                    if self.env.game_over:
                        print(Fore.YELLOW + "[execute_resource_plan] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                        return self.env.game_success, None, None, None
                    if independent:
                        self.novelty_recovery_agent.mode = init_mode

                    if not success:
                        # If novel item encountered, need to reset plan and learners
                        if self.novelty_recovery_agent.novel_item_encountered:
                            can_continue = self.novelty_recovery_agent.reset_due_to_novel_type()
                            if not can_continue:
                                return False, -2, None, False
                            else:
                                print('continuing plan execution after novel object during learning')
                                return self.execute_resource_plan(time_to_run - (time.time() - start))
                        # Else we're out of time or exceeded step cost
                        else:
                            if not independent:
                                # Mirror what's done in exploration with cps_attempts_without_progress
                                self.cps_attempts_without_progress[self.current_res_cp] += 1
                                if self.cps_attempts_without_progress[self.current_res_cp] > 5:
                                    self.cps_attempts_without_progress[self.current_res_cp] = 0
                                    # Even in learning case, if novelty difficulty changes this could be the case
                                    print(Fore.YELLOW + 'Tried learning on too many episodes from a checkpoint without making progress, Indicating cp as detrimental to attempt bridging from earlier cp')
                                    self.detrimental_res_cps.append(self.current_res_cp)
                            return False, -1, None, False
                    else:
                        if not independent:
                            self.novelty_recovery_agent.mode = 'planning'

                # Otherwise if we've failed on this step but don't have a learner, means we didn't finish exploration
                elif operator_str in self.failed_operators and operator_str in self.operators_failed_per_cp[self.current_res_cp]:
                    print(Fore.BLUE + "Reached failed operator without a paired learning operator, passing control to NoveltyRecovery agent to explore\n")
                    if self.novelty_recovery_agent.failure_info is None or self.novelty_recovery_agent.failure_info['operator'] is None:
                        print(Fore.LIGHTRED_EX + 'WARN: resuming exploration but failure info is None, should not be the case. Setting original operator with same name as failed operator')
                        self.novelty_recovery_agent.failure_info = {'operator': self.operator_map[operator_str]}
                    return False, None, None, False

                # Else execute original operator
                else:
                    success, outcome, info = self.execute_operator(operator_str)
                    #if game's over need to return
                    if self.env.game_over:
                        print(Fore.YELLOW + "[execute_resource_plan] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                        return self.env.game_success, None, None, None
                    #If original operator Impl fails, need to resort to exploration or learning
                    if not success:
                        #Give one chance at failure
                        self.failed_hardcoded_once_cps[self.current_res_cp][i] += 1
                        if self.failed_hardcoded_once_cps[self.current_res_cp][i] < 2:
                            print(Fore.LIGHTYELLOW_EX + 'Allowing hardcoded operator failure once consecutively for this cp due to rare outcomes in polycraft')
                            # Try replanning
                            # Using Operator Impl won't use step_env so need to update res cp
                            info_ = self.env.get_info()
                            rs = self.get_current_resource_checkpoint(info_)
                            if rs != resource_step:
                                self.novelty_recovery_agent.last_res_cp = rs
                            self.current_subplan_step = 0
                            break
                        #Indicate operator as having failed
                        else:
                            self.failed_operators.append(operator_str)
                            self.operators_failed_per_cp[self.current_res_cp].append(operator_str)
                            return False, outcome, info, False
                    # Special notion, can no longer motion plan if movement execution fails
                    if operator_str.startswith('moveTo'):
                        self.novelty_recovery_agent.failed_last_motion_plan = False

                    # Using Operator Impl won't use step_env so need to update res cp
                    info_ = self.env.get_info()
                    rs = self.get_current_resource_checkpoint(info_)
                    if rs > resource_step:
                        # reset consecutive failure count on success
                        self.failed_hardcoded_once_cps[self.current_res_cp][i] = 0
                        self.novelty_recovery_agent.last_res_cp = rs
                        break

    #Execute original operator Impl
    def execute_operator(self, operator_str):
        operator = self._get_operator(operator_str)
        info_, a_ = None, None
        obs = self.env.observation()
        info = self.env.get_info()
        planning_success = operator.reset(obs, info, self.env)
        if planning_success:
            action = operator.get_action()
            while action is not None:
                # Need to keep details for dynamics update
                info_ = self.env.get_info()
                a_ = action
                self.env.step(action)
                if self.env.game_over:
                    print(Fore.YELLOW + "[execute_operator] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                    return self.env.game_success, None, None
                action = operator.get_action()
            obs = self.env.observation()
            info = self.env.get_info()
            success = operator.check_success(obs, info)

            # If failed, return results so recover agent can handle response
            if not success:
                if operator_str == 'extractRubber':
                    print(info_, info, obs, a_)
                    if info['block_in_front']['name'] == 'polycraft:tree_tap':
                        print(Fore.YELLOW + 'Hardcoded extractRubber operator failed for some reason, attempting to pick tap back up to avoid going and creating a new one')
                        self.env.step(self.env.actions_id['BREAK_BLOCK'])
                        if self.env.game_over:
                            print(Fore.YELLOW + "[execute_operator] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                            return self.env.game_success, None, None
                        self.env.step(self.env.actions_id['MOVE w'])
                        if self.env.game_over:
                            print(Fore.YELLOW + "[execute_operator] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                            return self.env.game_success, None, None
                    else:
                        print(Fore.YELLOW + 'Hardcoded extractRubber operator failed for some reason, and block in front is not a tree_tap, cannot attempt to pick tap back up to avoid going and creating a new one')

                outcome = 1 if planning_success else 0
                return False, 1, {'operator': operator,
                                  'info': info_,
                                  'action': a_,
                                  'info2': info}
            else:
                return True, None, None
        else:
            return False, 0, {'operator': operator,
                              'info': info_,
                              'action': a_,
                              'info2': info}

    #Examine state of world to determine current checkpoint in resource space
    # (Determines which suplan to execute next)
    def get_current_resource_checkpoint(self, info):

        tap_count = get_inv_quant(info, 'polycraft:tree_tap')
        rubber_count = get_inv_quant(info, 'polycraft:sack_polyisoprene_pellets')
        stick_count = get_inv_quant(info, 'minecraft:stick')
        planks_count = get_inv_quant(info, 'minecraft:planks')
        log_count = get_inv_quant(info, 'minecraft:log')

        # TODO: Either use planner or compute these thresholds each round dynamically based off of
        #   recipes. Possible, but a little involved. Currently assuming planner can handle any recipe novelties
        #   so not on priority
        #Compute resources needed still to get to goal, then use res_cps to determine next course of action
        sticks_needed = 4 if tap_count > 0 or rubber_count > 0 else 1
        planks_needed_for_sticks = 0 if stick_count >= sticks_needed else np.ceil(((sticks_needed - stick_count) / 4)) * 2
        planks_needed = 2 if tap_count > 0 or rubber_count > 0 else 5
        planks_needed += planks_needed_for_sticks
        logs_needed = 0 if planks_count >= planks_needed else (np.ceil((planks_needed - planks_count) / 4) - log_count)

        # Have pogostick, done
        if get_inv_quant(info, 'polycraft:wooden_pogo_stick') > 0:
            return 6
        # Have rubber and no longer need logs
        elif rubber_count > 0 and logs_needed <= 0:
            # craft pogo
            return 5
        # Have rubber and need logs (must be log in world)
        elif rubber_count > 0 and logs_needed > 0:  # and \
            # get_world_quant(info, 'minecraft:log') > 0:
            # get log
            return 4
        # Have tap (must be log in world)
        elif tap_count > 0:  # and get_world_quant(info, 'minecraft:log') > 0:
            # extract rubber
            return 3
        # Need <= 1 log (means we have two's worth) (must be log in world)
        elif logs_needed <= 0:  # and get_world_quant(info, 'minecraft:log') > 0:
            # craft tap
            return 2
        # Needs <= 2 logs (means we have one's worth)
        elif logs_needed <= 1:
            return 1
        # need 3 logs - means we have nothing
        else:
            return 0

#Main class that maintains all learners and explorers and supplies observations and effects/rewards to the agents.
#Goal: Integrate more heavily with planner
#Current: communicates with DummyPlanningAgent and DynamicsChecker to emulate information that would be
#         available through the planner
class NoveltyRecoveryAgent:
    def __init__(self, env, trade_sock=None, init_capacity=100000, seed=0, novel_items=[], encountered_novelty=False):

        self.env = env
        self.sock = trade_sock

        # TODOFuture: not using TRADE planner atm
        self.can_trade_plan = False

        # Initial operators for DummyPlanner to use
        operator_map = {
            "moveTo minecraft:log": AStarOperator("moveTo minecraft:log", 'minecraft:log', ['near minecraft:log']),
            "moveTo minecraft:crafting_table": AStarOperator("moveTo minecraft:crafting_table",
                                                             'minecraft:crafting_table',
                                                             ['near minecraft:crafting_table']),
            "break minecraft:log": HardcodedOperator("break minecraft:log", [3, 0, 1], ['increase inventory_log 1']),
            "craft polycraft:tree_tap": CraftingOperator('craft polycraft:tree_tap', "polycraft:tree_tap",
                                                         ['increase inventory polycraft:tree_tap 1']),
            "craft polycraft:wooden_pogo_stick": CraftingOperator('craft polycraft:wooden_pogo_stick',
                                                                  "polycraft:wooden_pogo_stick",
                                                                  ['increase inventory polycraft:wooden_pogo_stick 1']),
            "extractRubber": ExtractRubberOperator("extractRubber", 'minecraft:log',
                                                   ['increase inventory polycraft:sack_polyisoprene_pellets 1']),
        }
        planning_agent = DummyPlanningAgentV2(self.env, operator_map)

        # Modes - 0:exploration, 1:learning, 2:planning
        self.mode = 'planning'
        self.planner = planning_agent
        # Don't want to do this with real planner
        self.planner.add_learner_connection(self)
        self.failure_reason = None
        self.failure_info = None
        self.control_agent = None
        self.control_agent_ind = None

        ##RL learning details##
        # Need to share tf session between all agents
        self.session = make_session(seed)
        self.seed = seed
        self.learning_agents = []
        self.operator_names = []
        # Because of numeric fluents, goal may depend on initial state when the operator is called,
        #   so we need to create the success criterion on the fly
        self.create_success_funcs = []
        self.success_funcs = []

        # Maintain buffer of experience from post-novelty env
        # Only init buffer upon novelty encounter (reset_buffers())
        self.buffer_capacity = init_capacity

        ##Exploration details##
        # For now we will have a single exploration agent, that can be copied to a learning agent if desired
        self.exploration_agent = None
        # Exploration will allow 'impermissible' actions in exploration in case they are necessary to find a
        #   opening novelty, but we want to pay careful attention to them
        self.impermissible_performed = False
        self.impermissible_reason = None
        # Let's see how using the same exploration agent towards different failed operators (will this ever be the case?) works
        # In the end, it will always want to find novelties, so shouldn't be a bad thing
        self.last_outcome = None

        self.num_exp_operators = 0
        self.current_exp_ind = 0
        self.num_exploration_failures = 0
        self.bridge_attempts_without_replan = 0
        self.last_res_cp = None
        self.found_relevant_exp_state = 0
        self.found_relevant_during_reset = False

        # If we lost moveTo functionality, we are likely going to need to recover it, but it's possible we recover
        #   the actual original goal (e.g. getting a log for moveTo tree) without ever recovering 'near tree'
        # It will usually be the case, but we need to make special accomodations in reformatting observations
        #   for a general moveTo operator, so we cannot flexibly swap between 'novelty_bridge' and 'moveTo'
        # For now we will maintain both until we know that one or the other is the case (should be quick)
        self.moveToAgent = None
        # If action dynamics change so that we cannot motion plan anymore, we cannot use it to reset to promising states
        self.can_motion_plan = True
        self.failed_last_motion_plan = False
        # Don't want to interrupt execution during motion planning
        self.motion_planning = False

        self.novel_items = novel_items
        self.encountered_novelty = encountered_novelty
        if encountered_novelty:
            self.reset_buffers()
        self.novel_item_encountered = False
        self.updated_spaces = False
        self.reset_near_values = []
        self.reset_select_values = []
        self.resetting_state = False

        # Semi-impermissible actions that would leave us in unsolvable states if allowed pre-novelty
        # Last three are just so we don't leave a tap on the ground and fail on later operators when we don't have to
        self.impermissible_actions = ['break_last_tree', 'craft_unnecessary_stick', 'craft_unnecessary_tap',
                                      'craft_new_recipe', 'placed_tap', 'extracted_rubber', 'broke_tap']
        # Will detect unexpected outcomes and 1. check in with the planner to see if we're in a plannable state
        #                                 or  2. charactize the outcome as beneficial, detrimental, or irrelevant
        self.dynamics_checker = PolycraftDynamicsChecker(self.env, self.sock, self)

        if USING_TOURNAMENT_MANAGER:
            self.main_loop_tournament()
        else:
            if args['env'] == 'polycraft':
                self.main_loop()
            else:
                self.main_loop_gridworld()

    # Main loop while using tournament_manager (reliant on game_over signal)
    def main_loop_tournament(self):
        round_step_costs = []
        while True:
            if CONNECT_TO_TRADE:
                print(Fore.BLUE + 'Waiting for START signal from TRADE')
                start = False
                while not start:
                    start_signal = recv_socket_data(self.sock).decode('UTF-8')
                    if start_signal == 'START\n':
                        print(Fore.GREEN + 'Recieved START signal')
                        start = True
                    else:
                        print(Fore.BLUE + 'havent recieved signal, got \'{}\' instead, sleeping for 5 sec...'.format(
                            start_signal))
                        time.sleep(5)

            ep_success = self.run_trial()
            while True:
                print('Waiting for game over signal from polycraft')
                if self.env.game_over:
                    self.env.game_over = False
                    print('Game over signal recieved')
                    print('Giving some time for env to complete reset, sleeping 5 seconds')
                    time.sleep(5)
                    break
                print('Sleeping 5 seconds to try again')
                time.sleep(5)
                self.env.run_SENSE_ALL_and_update()

            # print(Fore.BLUE + "Total step cost accrued over episode was {}".format(self.env.accumulated_step_cost))
            print(Fore.LIGHTBLUE_EX + '\n\nTrial summary')
            print(Fore.LIGHTBLUE_EX + 'Goal success: {}'.format(self.env.game_success))
            print(Fore.LIGHTBLUE_EX + "Estimated step cost accrued over episode was {}".format(
                self.env.accumulated_step_cost))
            round_step_costs.append(self.env.accumulated_step_cost)
            self.env.accumulated_step_cost = 0
            # print(Fore.LIGHTBLUE_EX + "Estimated time spent on trial was {}".format(time.time() - start))
            print(Fore.LIGHTBLUE_EX + 'Operators we have failed:\n{}'.format(self.planner.failed_operators))
            print(Fore.LIGHTBLUE_EX + 'Operators we are relearning\n{}:'.format(self.planner.learning_operators))
            print(Fore.LIGHTBLUE_EX + 'Current pogo plan:\n{}'.format(self.planner.resource_checkpoint_subplans))
            print(Fore.LIGHTBLUE_EX + 'Current detrimental cps: {}'.format(self.planner.detrimental_res_cps))
            print(Fore.LIGHTBLUE_EX + 'Max res cp: {}'.format(self.planner.max_res_cp))
            print(Fore.LIGHTBLUE_EX + 'Res cp we ended on: {}'.format(self.planner.current_res_cp))
            print(Fore.LIGHTBLUE_EX + 'step_costs per round so far: \n', round_step_costs)

            if CONNECT_TO_TRADE:
                if self.env.game_success:
                    print(Fore.BLUE + 'Sending END SUCCESS to TRADE')
                    self.sock.send(str.encode('END SUCCESS\n'))
                else:
                    print(Fore.BLUE + 'Sending END FAIL to TRADE')
                    self.sock.send(str.encode('END FAIL\n'))

            # reset game success
            self.env.game_success = False

    # Main loop while not using tournament_manager (self managed game_over and reset)
    def main_loop(self):
        task_list = []
        num_rounds = 0
        for file in sorted(os.listdir(TASK_DIR)):
            if file.endswith('.json'):
                task_list.append('RESET domain ' + TASK_DIR + file)
                num_rounds += 1
        round_step_costs = []
        num_successes = 0
        for round_num in range(num_rounds):
            # TODO: not sure if this works
            if CONNECT_TO_TRADE:
                print(Fore.BLUE + 'Waiting for START signal from TRADE')
                start = False
                while not start:
                    start_signal = recv_socket_data(self.sock).decode('UTF-8')
                    if start_signal == 'START\n':
                        print(Fore.GREEN + 'Recieved START signal')
                        start = True
                    else:
                        print(Fore.BLUE + 'havent recieved signal, got \'{}\' instead, sleeping for 5 sec...'.format(
                            start_signal))
                        time.sleep(5)

            print(Fore.BLUE + "\n\n\n\nround num {} start".format(round_num))
            self.env.run_a_command_and_update_map(task_list[round_num], sleep_time=5)
            ep_success = self.run_trial(TIME_PER_ROUND)
            round_step_costs.append(self.env.accumulated_step_cost)

            print(Fore.LIGHTBLUE_EX + '\n\nTrial summary')
            print(Fore.LIGHTBLUE_EX + 'Goal success: {}'.format(ep_success))
            print(Fore.LIGHTBLUE_EX + "Estimated step cost accrued over episode was {}".format(
                self.env.accumulated_step_cost))
            self.env.accumulated_step_cost = 0
            # print(Fore.LIGHTBLUE_EX + "Estimated time spent on trial was {}".format(time.time() - start))
            print(Fore.LIGHTBLUE_EX + 'Operators we have failed:\n{}'.format(self.planner.failed_operators))
            print(Fore.LIGHTBLUE_EX + 'Operators we are relearning\n{}:'.format(self.planner.learning_operators))
            print(Fore.LIGHTBLUE_EX + 'Current pogo plan:\n{}'.format(self.planner.resource_checkpoint_subplans))
            print(Fore.LIGHTBLUE_EX + 'Current detrimental cps: {}'.format(self.planner.detrimental_res_cps))
            print(Fore.LIGHTBLUE_EX + 'Max res cp: {}'.format(self.planner.max_res_cp))
            print(Fore.LIGHTBLUE_EX + 'Res cp we ended on: {}'.format(self.planner.current_res_cp))
            print(Fore.LIGHTBLUE_EX + 'step_costs per round so far: \n', round_step_costs)

            if CONNECT_TO_TRADE:
                if ep_success:
                    self.sock.send(str.encode('END SUCCESS\n'))
                else:
                    self.sock.send(str.encode('END FAIL\n'))

            if ep_success:
                num_successes += 1
        print(Fore.BLUE + 'num successes after {} rounds: \n'.format(num_rounds), num_successes)
        print(Fore.BLUE + 'step_costs per round: \n', round_step_costs)

    # Main loop while not using NG
    def main_loop_gridworld(self, num_rounds=100):
        round_step_costs = []
        num_successes = 0
        for round_num in range(num_rounds):
            # TODO: not sure if this works
            if CONNECT_TO_TRADE:
                print(Fore.BLUE + 'Waiting for START signal from TRADE')
                start = False
                while not start:
                    start_signal = recv_socket_data(self.sock).decode('UTF-8')
                    if start_signal == 'START\n':
                        print(Fore.GREEN + 'Recieved START signal')
                        start = True
                    else:
                        print(Fore.BLUE + 'havent recieved signal, got \'{}\' instead, sleeping for 5 sec...'.format(
                            start_signal))
                        time.sleep(5)

            print(Fore.BLUE + "\n\n\n\nround num {} start".format(round_num))
            if (round_num + 1) % 10 == 0:
                self.env.render_bool = True
            else:
                self.env.render_bool = False
            # self.env.reset()
            ep_success = self.run_trial(TIME_PER_ROUND)
            round_step_costs.append(self.env.accumulated_step_cost)

            self.env.env.reset()

            print(Fore.LIGHTBLUE_EX + '\n\nTrial summary')
            print(Fore.LIGHTBLUE_EX + 'Goal success: {}'.format(ep_success))
            print(Fore.LIGHTBLUE_EX + "Estimated step cost accrued over episode was {}".format(
                self.env.accumulated_step_cost))
            self.env.accumulated_step_cost = 0
            # print(Fore.LIGHTBLUE_EX + "Estimated time spent on trial was {}".format(time.time() - start))
            print(Fore.LIGHTBLUE_EX + 'Operators we have failed:\n{}'.format(self.planner.failed_operators))
            print(Fore.LIGHTBLUE_EX + 'Operators we are relearning\n{}:'.format(self.planner.learning_operators))
            print(Fore.LIGHTBLUE_EX + 'Current pogo plan:\n{}'.format(self.planner.resource_checkpoint_subplans))
            print(Fore.LIGHTBLUE_EX + 'Current detrimental cps: {}'.format(self.planner.detrimental_res_cps))
            print(Fore.LIGHTBLUE_EX + 'Max res cp: {}'.format(self.planner.max_res_cp))
            print(Fore.LIGHTBLUE_EX + 'Res cp we ended on: {}'.format(self.planner.current_res_cp))
            # print(Fore.LIGHTBLUE_EX + 'step count for ep: {}'.format(self.env.env.step_count))
            print(Fore.LIGHTBLUE_EX + 'step_costs per round so far: \n', round_step_costs)

            if CONNECT_TO_TRADE:
                if ep_success:
                    self.sock.send(str.encode('END SUCCESS\n'))
                else:
                    self.sock.send(str.encode('END FAIL\n'))

            if ep_success:
                num_successes += 1
        print(Fore.BLUE + 'num successes after {} rounds: \n'.format(num_rounds), num_successes)
        print(Fore.BLUE + 'step_costs per round: \n', round_step_costs)

    #Run a single trial, independent of Env details
    def run_trial(self, time_per_round=0):
        print('\n\n\n\nStarting new trial')
        start = time.time()
        self.env.accumulated_step_cost = 0
        self.env.run_SENSE_RECIPES_and_update()
        print('First senseAll in new round')
        print(self.env.run_SENSE_ALL_and_update('NONAV'))
        if not self.env.first_space_init:
            self.env.generate_obs_action_spaces()
        self.reset_trial_variables()
        if time_per_round is not None:
            print(Fore.BLUE + "Passing control to NoveltyRecoveryAgent for {} secs\n".format(time_per_round))
        else:
            print(Fore.BLUE + "Passing control to NoveltyRecoveryAgent\n")
        ep_success = self.control_for_time(time_per_round)

        if time_per_round > 0:
            print(Fore.BLUE + "Agent returned control with time_left == {}\n".format(time_per_round - (time.time() - start)))
        else:
            print(Fore.BLUE + "Agent returned control after {} secs\n".format(time.time() - start))

        return ep_success

    #Reset working memory of details relevant only to current trial
    def reset_trial_variables(self):
        self.impermissible_performed = False
        self.impermissible_reason = None
        self.last_outcome = None
        self.mode = 'planning'
        self.planner.reset_plan_step()
        self.last_reset_pos = None
        self.current_exp_ind = 0
        self.placed_tap = False
        self.last_action = None
        self.last_obs = None
        self.exceeded_step_cost = False
        self.bridge_attempts_without_replan = 0
        self.last_res_cp = None
        self.found_relevant_during_reset = False
        self.resetting_state = False

    #Encountering a novel type changes the action and observation space of learning agents, requires
    #   full re-init
    #TODO: append input and output nodes rather than re-init, and initialize values based off of
    #   nearest similar type or action
    def reset_due_to_novel_type(self):
        # identify novel type
        for item in self.env.items_id:
            if item not in self.env.mdp_items_id:
                self.novel_items.append(item)

        # Update spaces and reset buffers
        self.novel_item_encountered = False
        self.reset_trial_variables()
        self.env.generate_obs_action_spaces(self.novel_items)
        self.updated_spaces = True
        self.reset_buffers()

        # Reset all learners and explorers
        # Ideally this should never happen after much learning/exploring
        self.moveToAgent = None
        self.motion_planning = False
        self.exploration_agent = None
        self.learning_agents = []
        self.operator_names = []
        self.create_success_funcs = []
        self.success_funcs = []
        self.failure_reason = None
        self.failure_info = None
        self.control_agent = None
        self.control_agent_ind = None
        self.reset_near_values = []
        self.reset_select_values = []

        self.found_relevant_exp_state = 0

        # Clear current tf session to resuse from scratch
        tf.keras.backend.clear_session()
        self.session = make_session(self.seed)

        continue_execution = self.planner.reset_all()
        return continue_execution

    # Main logic loop - handles passing between planning, learning, and execution
    def control_for_time(self, time_to_run, reset_every=1):
        if USING_TOURNAMENT_MANAGER:
            print(Fore.BLUE + 'NoveltyRecovery Agent assuming control... In {} mode'.format(self.mode))
        else:
            print(Fore.BLUE + 'NoveltyRecovery Agent assuming control for {} seconds... In {} mode'.format(time_to_run,self.mode))
        if not USING_TOURNAMENT_MANAGER and time_to_run < 30:
            print(Fore.BLUE + 'Time to run is less than 30, stopping execution to prevent running over trial time'.format(time_to_run, self.mode))
            return False
        elif self.exceeded_step_cost:
            print(Fore.BLUE + 'Got too close to step cost limit with {}, stopping execution to prevent going over '.format(self.env.accumulated_step_cost))
            return False
        elif self.env.game_over:
            print(
                Fore.BLUE + '[control_for_time] Env indicated that game is over, stopping execution and resetting for next trial')

        start = time.time()
        # Planning mode:
        #  Execute hardcoded plan until failure step
        #      First instance: characterize novelty, update dynamics, and create exploration agent
        #      Following instances: just pass control to exploration or learning agent
        if self.mode == 'planning':
            assert self.planner is not None, 'ERROR, cannot ask NoveltyRecoveryAgent to execute plan without supplying planner'
            success, failure_reason, failure_info, plan_complete = self.planner.execute_resource_plan(time_to_run)

            if self.env.game_over:
                return success

            if not success:
                # Failure reasons:
                #   -1 - out of time (or stepcost)
                #   0  - planning failure
                #   1  - execution failure
                # If ran out of time, not much we can do
                if failure_reason == -1:
                    print(Fore.RED + 'Ran out of time in plan execution or exceeded step cost limit, returning')
                    return False

                elif failure_info is not None and failure_info['operator'].name in self.planner.learning_operators:
                    print(Fore.LIGHTCYAN_EX + 'Already have a learner for this operator which failed on a different step, not creating a new learner')
                    print(Fore.LIGHTCYAN_EX + 'Set to use learner for this step as well')
                    self.mode = 'planning'
                    return self.control_for_time(time_to_run - (time.time() - start))

                # Presumably failed to due novelty, update observation/action spaces for RL agents
                if not self.updated_spaces:
                    self.env.generate_obs_action_spaces(self.novel_items)
                    self.updated_spaces = True

                # Also begin to store transitions and init buffers with updated notion of any novel items
                if not self.encountered_novelty:
                    self.reset_buffers()
                    self.encountered_novelty = True

                # Get info about the failure from the dummy planner
                if failure_info is not None:
                    operator = failure_info['operator']
                    self.failure_reason = failure_reason
                    self.failure_info = failure_info
                    operator_str = None if operator is None else operator.name
                    print(Fore.YELLOW + "plan step failed with code {} on operator {}".format(failure_reason,operator_str))
                else:
                    print(Fore.LIGHTYELLOW_EX + 'Plan step failed but no failure info provided - this should only be the case upon resuming unsuccessful exploration')

                # Planning faiure
                if failure_reason == 0:
                    # Planning failed for some reason before trying anything in env (likely some object(s) no longer exists)
                    if operator is None:
                        # TODO: this is currently never the case because planner doesn't check if the initial
                        #   state is plannable or not, but we can add checks and special responses for it
                        # Would have to consider possibility that current actions aren't good enough (so consider detrimental if they waste resources)
                        print(Fore.RED + 'ERROR: planner failure without sending failed operator info -> this should never be the case')

                    # Treat moveTo specially -> want to have general moveTo learner independent of goal type
                    elif operator_str.startswith('moveTo'):
                        # Would result in initial planning failure if object didn't exist
                        print(Fore.CYAN + 'Failed due to issue in motion planning - need to clear path to goal object')
                        self.dynamics_checker.update_from_failure(operator.name, {'cause': 'planning'})

                        # Create general moveTo learner to use and start exploring
                        if self.moveToAgent is None:
                            self.create_learning_agent(operator_str, self.failure_info['operator'].effect_set)
                            self.create_exploration_agent()
                            self.mode = 'exploration'
                        # Otherwise we have a moveTo learner, skip exploration and use that learner
                        else:
                            self.create_learning_agent(operator_str, self.failure_info['operator'].effect_set)
                            self.planner.add_relearning_operator(self.failure_info['operator'].name)
                            print(
                                Fore.CYAN + "Already learning moveTo operator - skipping exploration and going straight to learning applying general moveTo operator")
                            self.mode = 'learning'
                            self.learn_for_time(time_to_run - (time.time() - start), operator_str)
                            self.mode = 'planning'

                    # TODO: remove these cases that should never happen
                    # Should never fail on crafting planning with new crafting operator unless no crafting table exists
                    elif operator_str.startswith('craft'):
                        # Would result in initial planning failure if object didn't exist
                        print(Fore.LIGHTRED_EX + 'Failed while creating plan for crafting operator - this should never be the case if moveTo crafting table has succeeded prior to this step')
                        print(Fore.LIGHTRED_EX + 'Have recipes for existing items been modified?')
                        self.dynamics_checker.update_from_failure(operator.name, {'cause': 'planning'})
                        self.create_exploration_agent()
                        self.mode = 'exploration'

                    #Also should never be the case currently
                    elif operator_str == 'extractRubber':
                        print(Fore.LIGHTRED_EX + 'Failed while creating plan for extractRubber operator - this should never be the case if moveTo log has succeeded prior to this step')
                        self.dynamics_checker.update_from_failure(operator.name, {'cause': 'planning'})
                        self.create_exploration_agent()
                        self.mode = 'exploration'
                    # Not planning for any other case
                    else:
                        print(Fore.RED + 'ERROR: recieved planning error, shouldnt be doing any planning for this operator, what happened?')
                        quit()

                # Execution failure
                elif failure_reason == 1:

                    if operator_str.startswith('moveTo'):
                        print(Fore.YELLOW + 'Found path for moveTo but execution failed, need to relearn moveTo')
                        if not self.failed_last_motion_plan:
                            print(Fore.LIGHTYELLOW_EX + 'Failed motion planning execution, but could be due to items popping up mid execution or rare bug in motion planner -> allowing another attempt')
                            self.failed_last_motion_plan = True
                        else:
                            print(Fore.YELLOW + 'Failed motion planning execution twice in a row, disabling')
                            self.can_motion_plan = False

                        self.dynamics_checker.update_from_failure(operator.name, {'cause': 'execution'})

                        # Create general moveTo learner to use and start exploring
                        if self.moveToAgent is None:
                            self.create_learning_agent(operator_str, self.failure_info['operator'].effect_set)
                            self.create_exploration_agent()
                            self.mode = 'exploration'
                        # Otherwise we have a moveTo learner, skip exploration and use that learner
                        else:
                            self.create_learning_agent(operator_str, self.failure_info['operator'].effect_set)
                            self.planner.add_relearning_operator(self.failure_info['operator'].name)
                            print(
                                Fore.CYAN + "Already learning moveTo operator - skipping exploration and going straight to learning applying general moveTo operator")
                            self.mode = 'learning'
                            self.learn_for_time(time_to_run - (time.time() - start), operator_str)
                            self.mode = 'planning'

                    else:
                        # update knowledge of NoveltyRecoveryAgent to not consider blocking action a novelty
                        self.dynamics_checker.update_from_failure(operator.name)
                        self.create_exploration_agent()
                        self.mode = 'exploration'

                # If no failure info provided, should just be picking back up in exploration at some point between trials
                else:
                    print(Fore.CYAN + 'Should have reached failed step where exploration was incomplete last episode')
                    self.mode = 'exploration'

            # Planner indicated successful completion of plan
            else:
                if plan_complete:
                    return True
                print(Fore.RED + "Plan not complete, but ending plan execution - This should never be the case anymore?")
                print(Fore.LIGHTRED_EX + "Did we indicate a state as plannable when that wasn't the case?")

            # return to top of control loop with new mode
            return self.control_for_time(time_to_run - (time.time() - start))

        # Will be passed control upon plan step failure (either from planner or static prenovelty plan if self-contained)
        elif self.mode == 'exploration':
            # TODO: See if this is necessary, shouldn't be
            if self.failure_info is None:
                print(Fore.LIGHTRED_EX+'Entering exploration with no failure_info, setting failed operator as current step')
                self.failure_info = self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][self.planner.current_subplan_step-1]

            print(Fore.BLUE + "In exploration mode with failure_reason {} for operator {}".format(self.failure_reason,self.failure_info['operator'].name))

            if self.exploration_agent is None:
                print(Fore.RED + 'ERROR, entering exploration without having initialized an exploration agent - should never be the case')
                quit()

            # Explore until plannable
            found_plannable, relevant_outcome = self.explore_for_time(time_to_run)

            print(Fore.CYAN + 'Explore_for_time returned with: {} {}'.format(found_plannable, relevant_outcome))

            # recovered lost operator functionality
            if relevant_outcome == 4:
                # If recovered lost moveTo - switch to moveTo learning agent, don't transfer explorer
                if self.failure_info['operator'].name.split()[0] == 'moveTo':
                    print(Fore.CYAN + "recovered moveTO, ditching explorer and using general moveTo learner")
                    self.planner.add_relearning_operator(self.failure_info['operator'].name)
                    self.failure_info = None
                    self.failure_reason = None
                # Copy explorer to learner for given operator
                else:
                    print(Fore.GREEN + 'Rediscovered action effects of lost operator! Attempting to create learning agent for operator and fit into original plan')
                    self.create_learner_from_explorer(True)

                # Go back to planning to complete episode
                self.num_exploration_failures = 0
                self.mode = 'planning'
                print(Fore.CYAN + 'Switching to resume planning...')
                return self.control_for_time(time_to_run - (time.time() - start))

            # Reset due to novelty
            elif relevant_outcome == 6:
                print(Fore.CYAN + "Updating spaces and resetting all learner/explorers/buffers to accomodate unexpected item type")
                can_continue = self.reset_due_to_novel_type()
                # Should always be can_continue in this version
                if not can_continue:
                    print('cant continue plan execution after novel object encountered during exploration')
                    return False
                else:
                    self.mode = 'planning'
                    print('continuing plan execution after novel object encountered during exploration')
                    return self.control_for_time(time_to_run - (time.time() - start))

            # Found plannable state as indicated by TRADE planner without exact lost operator functionality
            #TODO: this branch would be used if recieving plannable states from TRADE planner -> reintegrate
            elif relevant_outcome == 3:
                print(Fore.GREEN + 'Found plannable state in exploration! Attempting to use planner to complete task...')
                # Dynamics checker will store plan sent back upon reception
                planner_success = self.planner.execute_plan_independent(self.dynamics_checker.trade_plan)
                # planner_success = input('Did planner finish successfully? y/N') == 'y'

                if planner_success:
                    if self.failure_info['operator'].name.split()[0] == 'moveTo':
                        print(Fore.LIGHTCYAN_EX + 'Found plannable state from exploring moveTO, ditching moveTo and using general learner')
                        self.planner.cps_attempts_without_progress[self.planner.current_res_cp] = 0
                        self.planner.resource_checkpoint_subplans[self.planner.current_res_cp] = \
                        self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][
                        :self.planner.current_subplan_step - 1] + \
                        self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][
                        self.planner.current_subplan_step:]
                        # Kind of hacky, should always just be one operator left in the subplan, create learner from that one, not moveto
                        self.failure_info['operator'] = self.planner.operator_map[
                            self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1]]
                        # add notion that we are learning this new operator
                        self.planner.operators_failed_per_cp[self.planner.current_res_cp].append(
                            self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1])
                        self.planner.failed_operators.append(
                            self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1])

                    self.create_learner_from_explorer(True)
                    self.num_exploration_failures = 0

                    # Bc planner only handles learning cases, not exploration
                    print(Fore.GREEN + 'TRADE Plan completed task successfully')
                    self.planner.cps_attempts_without_progress[self.planner.current_res_cp] = 0
                    self.mode = 'planning'
                    return True
                else:
                    print(Fore.YELLOW + 'Planner indicated state as plannable but failed in execution')
                    print(Fore.LIGHTYELLOW_EX + 'We cannot trust the planner anymore, must attempt to solve task without using trade planner assistance')
                    self.can_trade_plan = False
                    self.mode == 'planning'
                    return self.control_for_time(time_to_run - (time.time() - start))

            # Jumped forward a Res checkpoint(s), go back to planning
            elif found_plannable:
                print(Fore.GREEN + 'Jumped forward more than one resource checkpoint, or one resource checkpoint in a different way')
                # If from moveTo explorer, ditch moveTo for that res step
                if self.failure_info['operator'].name.split()[0] == 'moveTo':
                    self.planner.cps_attempts_without_progress[self.planner.current_res_cp] = 0
                    self.planner.resource_checkpoint_subplans[self.planner.current_res_cp] = self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][:self.planner.current_subplan_step - 1] + self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][self.planner.current_subplan_step:]
                    # Kind of hacky, should always just be one operator left in the subplan, create learner from that one, not moveto
                    self.failure_info['operator'] = self.planner.operator_map[self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1]]
                    # add notion that we are learning this new operator
                    self.planner.operators_failed_per_cp[self.planner.current_res_cp].append(self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1])
                    self.planner.failed_operators.append(self.planner.resource_checkpoint_subplans[self.planner.current_res_cp][-1])

                self.create_learner_from_explorer(True)
                self.num_exploration_failures = 0

                # Go back to planning to complete episode
                self.mode = 'planning'
                print(Fore.CYAN + 'Switching to resume planning...')
                return self.control_for_time(time_to_run - (time.time() - start))
            #Failed exploration
            else:
                self.planner.cps_attempts_without_progress[self.planner.current_res_cp] += 1
                if self.planner.cps_attempts_without_progress[self.planner.current_res_cp] > 5:
                    self.planner.cps_attempts_without_progress[self.planner.current_res_cp] = 0
                    print(Fore.YELLOW + 'Explored too many episodes from a checkpoint without making progress, Indicating cp as detrimental to attempt bridging from earlier cp')
                    self.planner.detrimental_res_cps.append(self.planner.current_res_cp)
                if relevant_outcome == 5:
                    self.planner.current_res_cp = self.last_res_cp
                    self.mode == 'planning'
                    print(Fore.YELLOW + 'Dropped back a resource checkpoint, this shouldnt be the case...')
                    return self.control_for_time(time_to_run - (time.time() - start))
                else:
                    # Otherwise we didn't find solution in exploration, out of time or exceeded step cost
                    print(Fore.LIGHTRED_EX + 'Exploration ended without finding a solution - should be out of time or exceeded step cost in trial')

                    # Didn't succeed return False
                    return False

        # THIS SHOULD NEVER BE THE CASE - planner will call learn_for_time directly
        elif self.mode == 'learning':
            print(Fore.RED + 'control_for_time in mode learning should never be called - learn_for_time should always directly be called')
            self.mode = planning
            return self.control_for_time(time_to_run - (time.time() - start))
        else:
            print(Fore.RED + 'Entered unknown mode, {}, switching to planning'.format(self.mode))
            self.mode = 'planning'
            return self.control_for_time(time_to_run - (time.time() - start))

    # FutureTODO: Reasoning explicitly over exploration results/successful trajectory
    #             to define better learning agent (better spaces)
    # Found goal during exploration for some operator, transition to using a learner for that operator
    def create_learner_from_explorer(self, recovered_operator):
        if self.failure_info['operator'].name in self.planner.learning_operators:
            print(Fore.LIGHTYELLOW_EX + 'Asked to make learner for already existing learner {}, doing nothing'.format(recovered_operator))
            self.failure_info = None
            self.failure_reason = None
            return

        # First, attempt to concretely define goal based on known cases if applicable
        #   I.E. recreate original operator with new policy
        if recovered_operator:
            self.operator_names.append(self.failure_info['operator'].name)
            effect_set = self.failure_info['operator'].effect_set
            self.planner.add_relearning_operator(self.failure_info['operator'].name)

            # If fitting operator back into plan, it's possible we may need to explore again on a future operator failing
            # Copy rather than append and reuse exploration_agent if necessary
            copied_agent = DQNLambda_Agent(self.seed, self.env, scope=''.join(
                ch for ch in self.failure_info['operator'].name if ch.isalnum()),
                                           session=self.session, **default_param_dict)

            copied_agent.copy_variables_from_scope('exploration')
            self.learning_agents.append(copied_agent)

            self.create_success_funcs.append(get_create_success_func_from_predicate_set(effect_set))
            self.success_funcs.append(None)

            # Memory buffer will be empty, want to repopulate
            # Feed transitions from buffer
            self.feed_transitions_from_buffer(len(self.learning_agents) - 1)

        # Novelty_bridge case - found plannable state as indicated by TRADE planner OR progressed
        #   resource checkpoints in some novel way
        #   Have separate learner for that resource checkpoint to learn this new task
        # As a fallback, sense_all in ADE and use entire predicate set as goal set
        else:
            print(Fore.LIGHTRED_EX + 'Believe novelty_bridge is deprecated, should not be in this conditional')
            print(Fore.LIGHTYELLOW_EX + 'TODO: get current fluent state of world to set as goal')

            # Fallback, use entire predicate set from sense_all:
            # Single operator that bridges gap introduced by novelty
            self.operator_names.append('novelty_bridge{}'.format(self.num_exp_operators))
            self.num_exp_operators += 1
            # Get entire set from planner
            print(
                Fore.LIGHTYELLOW_EX + 'Using impossible effect set for novelty_bridge until we can get fluent state from planner - relying on novel outcomes and intermittent checks to try replanning')
            effect_set = ['increase inventory minecraft:log 1000']

            # If learning novelty_bridge, then should no longer require any new agents
            # So we don't have to copy the variables
            # ** Don't want to crash in case planner sends back 'yes plannable' and then fails in execution
            # ** Copying explorer to separate agent just in case we need to explore again
            copied_agent = DQNLambda_Agent(self.seed, self.env, scope='novelty_bridge{}'.format(self.num_exp_operators),
                                           session=self.session, **default_param_dict)
            copied_agent.copy_variables_from_scope('exploration')
            self.num_exp_operators += 1
            self.learning_agents.append(copied_agent)
            self.create_success_funcs.append(get_create_success_func_from_predicate_set(effect_set))
            self.success_funcs.append(None)

            # Memory buffer will be empty, want to repopulate
            # Feed transitions from buffer - BUT GET RID OF 'beneficial' rewards
            #   We only want to give goal reward to prevent learning auxilliary novelties
            # ^Currently does so
            self.feed_transitions_from_buffer(len(self.learning_agents) - 1)

        self.add_reset_probas()
        self.control_agent = self.learning_agents[-1]
        self.control_agent_ind = len(self.learning_agents) - 1
        print(Fore.CYAN + 'Creating new learning agent towards effect set:', effect_set)

        # need to reset failure details in case we encounter another failure down the line
        self.failure_info = None
        self.failure_reason = None

    ############## Experience Buffer Functions ##############
    def store_transition(self, obs, action, done, info, info2):
        info['last_outcome'] = self.last_outcome
        # Never getting to this point because checked in step_env
        if self.buffer_ind >= self.buffer_capacity:
            if self.buffer_ind == self.buffer_capacity:
                print(Fore.LIGHTYELLOW_EX + "Can't store any more transitions, need to write code to expand buffer size for now overwritting oldest transitions")
                self.buffer_ind = 0
                self.buffer_full = True
            return
        self.obs_buffer[self.buffer_ind] = obs
        self.action_buffer[self.buffer_ind] = action
        self.done_buffer[self.buffer_ind] = done
        # TODO: CHANGE THIS, waste of space
        # Issue is that we need an extra info per episode to be able to determine whether the last
        #   transition actually led to success or not given updated info and operator to be able
        #   to accurately compute reward
        if self.buffer_full:
            self.info_buffer[self.buffer_ind] = info
            self.info2_buffer[self.buffer_ind] = info2
        else:
            self.info_buffer.append(info)
            self.info2_buffer.append(info2)
        self.buffer_ind += 1

    def reset_buffers(self):
        obs = self.env.observation()
        self.obs_buffer = np.empty([self.buffer_capacity, *obs.shape], dtype=obs.dtype)
        self.action_buffer = np.empty([self.buffer_capacity], dtype=np.int32)
        self.done_buffer = np.empty([self.buffer_capacity], dtype=np.bool)
        self.info_buffer = []
        self.info2_buffer = []
        self.buffer_ind = 0
        self.buffer_full = False

    def feed_transitions_from_buffer(self, agent_id):
        found_success = False
        t = 0
        # Don't want to mess with class variable in case this is called in the middle of an episode for whatever reason
        num_transitions = self.buffer_capacity if self.buffer_full else self.buffer_ind
        success_func = None
        for i in range(num_transitions):
            # eventually reformat obs here if possible (to exclude irrelevant inputs or actions)
            obs = self.obs_buffer[i]
            action = self.action_buffer[i]
            # Technically don't need a done buffer but need to make sure we don't overlap episodes
            # Done buffer should be on actual experience stream interruption, not pseudo-episode definition
            done = self.done_buffer[i]
            info = self.info_buffer[i]
            info2 = self.info2_buffer[i]

            operator_str = self.failure_info['operator'].name
            # ONLY allow transitions from exploration towards the same operator to be included
            # Will lose episode where we jump cps from moveTo. FutureTODO -> go back and edit operator name for those transitions
            if operator_str == info['operator_name']:
                if t == 0:
                    success_func = self.create_success_funcs[agent_id](obs, info)
                t += 1
                self.learning_agents[agent_id].store_obs(obs)
                success = success_func(obs, info2)
                if success or info['last_outcome'] == 'plannable' or info['last_outcome'] == 'recovered':
                    rew = 1000
                    found_success = True
                elif info['permissible'] in self.impermissible_actions:
                    if info['permissible'] == 'placed_tap' or info['permissible'] == 'extracted_rubber' or info[
                        'permissible'] == 'broke_tap':
                        rew = -1
                    else:
                        print(info['permissible'])
                        rew = -1000
                # smudging details on last one, setting 'placed_tap' as the reason in these on placing the tap when that's not the case
                elif info['last_outcome'] == 'lost_progress' or info['new_res_cp'] in self.planner.detrimental_res_cps and not info['reason'] == 'placed_tap':
                    rew = -500
                else:
                    if info['last_outcome'] == 'beneficial' and not RESTRICT_BENEFICIAL:
                        rew = 200
                    else:
                        # More strongly penalize craft,select,place actions because they should not be used often in learning at all
                        # Next step -> restrict action space to exclude these actions when we can, but filter back in for some cases
                        if self.env.all_actions[action].split()[0] == 'CRAFT' or self.env.all_actions[action].split()[0] == 'SELECT_ITEM' or self.env.all_actions[action].startswith('PLACE'):
                            rew = -50
                        else:
                            rew = -1
                rew -= round(info2['last_step_cost'] * STEPCOST_PENALTY)

                self.learning_agents[agent_id].store_effect(action, rew, success or done)

                # train based on timestep intervals not episode intervals
                self.learning_agents[agent_id].timesteps_trained += 1
                self.learning_agents[agent_id].check_update()

                if success or done:
                    self.learning_agents[agent_id].eps_trained += 1
                    t = 0

    ############## Learning Agent Functions ##############
    # Upon failure of an operator, create a learning agent to recover subgoal
    def create_learning_agent(self, operator_name, effect_set=None,
                              feed_transitions=False):  # Future: action_dict? obs_space?
        # operator name must be unique from previously existing operators in agent, using as tf scope
        if operator_name in self.operator_names:
            print(Fore.YELLOW + "WARNING: adding learning agent for already existing operator_str {}".format(operator_name))
            print(Fore.YELLOW + "Hopefully this is due to an operator failing for a different reason (planning vs execution) or the plan failing on a different instance of an operator that was transformed to novelty_bridge - going to use same learner")
            return

        # Regardless of case, add operator name to operator set and create success functions
        self.operator_names.append(operator_name)

        # If supplied effect set
        if effect_set is not None:
            self.create_success_funcs.append(get_create_success_func_from_predicate_set(effect_set))
        # Otherwise try to create automatically based off the failed operator
        else:
            print(Fore.YELLOW + 'creating success func from failed operator - dont think this should be the case')
            self.create_success_funcs.append(get_create_success_func_from_failed_operator(operator_name))
        self.success_funcs.append(None)

        # Need separate success criteria for different moveTo operators (i.e. what block the agent is near) but
        # want to use the same operator regardless of case
        # Keep notion of different moveTo operators and success funcs, but just always use general moveTo agent when executing
        if operator_name.startswith('moveTo'):
            print(Fore.CYAN + 'adding specific moveTo operator to operator list but using general agent')
            self.learning_agents.append(None)
            # Base moveTo effect set doesn't use info or obs
            self.success_funcs[-1] = self.create_success_funcs[-1](None, None)
            if self.moveToAgent is None:
                print(Fore.CYAN + 'making general moveToAgent')
                default_param_dict['eps_lambda'] = -math.log(0.01) / 4000.
                # scope must be alphanumeric only
                agent = DQNLambda_Agent(self.seed, self.env, scope='moveTo',
                                        session=self.session, **default_param_dict)
                self.moveToAgent = agent
        # Otherwise always create new agent
        else:
            default_param_dict['eps_lambda'] = -math.log(0.01) / 4000.
            # scope must be alphanumeric only
            agent = DQNLambda_Agent(self.seed, self.env, scope=''.join(ch for ch in operator_name if ch.isalnum()),
                                    session=self.session, **default_param_dict)

            self.learning_agents.append(agent)
            # Immediately prepopulate with experience stored in buffers here
            if feed_transitions:
                found_success = self.feed_transitions_from_buffer(len(self.learning_agents) - 1)

        # if we went straight to learning - add placeholder for exploration probas
        if len(self.reset_near_values) == 0:
            self.add_reset_probas()

        # copy reset state values from current exploration agent (will reset upon returning to exploration)
        self.reset_near_values.append(self.reset_near_values[0].copy())
        self.reset_select_values.append(self.reset_select_values[0].copy())

    # TODO: notion of learning and exploration has become more and more muddled over time, currently
    #       learning mirrors exploration pretty heavily. See how much we can merge and resolve rather
    #       than repeating code
    # Perform learning for indicated operator until success or timeout
    # Args:
    #       time - number of seconds we have to perform training
    #       operator_str - current operator step in plan
    #       reset_every - number of 'episodes' to run before resetting to interesting state (better than 'start' in practice)
    def learn_for_time(self, time_to_train, operator_str, reset_every=1):
        start = time.time()
        if not USING_TOURNAMENT_MANAGER:
            print(Fore.BLUE + "Entering learning for operator {}".format(operator_str))
        else:
            print(Fore.BLUE + "Entering learning for operator {} with time {}".format(operator_str, time_to_train))
        try:
            agent_ind = self.operator_names.index(operator_str)
            self.control_agent_ind = agent_ind
            self.control_agent = self.learning_agents[self.control_agent_ind]
            # Need to do this in case we achieve the goal during initial reset
            self.success_funcs[agent_ind] = self.create_success_funcs[agent_ind](self.env.observation(),
                                                                                 self.env.get_info())
        except ValueError:
            print(Fore.RED + "Error: passing control to nonexisting operator {}".format(operator_str))
            return

        if not USING_TOURNAMENT_MANAGER and time_to_train < 30:
            print(Fore.YELLOW + 'Entering learning with less than 30 secs remaining, ending execution to prevent going over time')
            return False

        ep_start_time = time.time()

        # If moveTo check if we can motion plan immediately (if orig failure was planning)
        move_near_id, select_id = None, None
        if operator_str.startswith('moveTo'):
            self.control_agent_ind = agent_ind
            self.control_agent = self.moveToAgent

        # If moveTo, check if we can plan straight to goal
        # If we fail, still try to move_near something in block below
        if self.can_motion_plan and operator_str.startswith('moveTo'):
            plan_success, move_success, _info = self.move_near(operator_str.split()[1])
            if self.found_relevant_during_reset:
                self.found_relevant_during_reset = False
                return True
        # Going to always reset to interesting state before learning (should be biased to go to
        #  what works, so if we're already there it'll stay)
        # Dont reset if we just performed a moveTo, doesn't consider object right in front of it
        #   (because path plan for that object is empty I belive - TODO, fix)
        # If step is > 0, not moveTO, otherwise could have ditched moveTo so check for that
        if self.planner.current_subplan_step <= 1 or operator_str.startswith('moveTo'):
            move_near_id, select_id = self.reset_to_interesting_state(first_reset=True)
            if self.found_relevant_during_reset:
                self.found_relevant_during_reset = False
                return True

        longest_ep_time = 0
        # Every 10 train episodes attempt to execute policy with eps=0
        # Evaluate as is without better action selection is hardly ever better
        evaluate_every = 10
        ep_num = 1
        # don't want this to carry over
        self.last_reset_pos = None

        #Go until game over for whatever reason (keep track of time and step cost personally if not using tournament manager)
        while ((not USING_TOURNAMENT_MANAGER) and time_to_train - (time.time() - start) > longest_ep_time + 20) or (
                USING_TOURNAMENT_MANAGER and not self.env.game_over):

            #Reincorporated relevant_outcome for reset update in learning
            # TODO: separating update outcome and return outcome due to differences in handlinglogic, unify notions
            #TODO: update episode length based off of success rate and time spent learning (better agent -> less resets)
            if ep_num % evaluate_every == 0:
                success, rel_outcome_update = self.run_learning_episode(50, agent_ind, evaluate=True)
            else:
                success, rel_outcome_update = self.run_learning_episode(50, agent_ind)

            # Relevant outcome in learning will still include move forward or move back CP, need to differentiate
            # Success is done now, not necessarily success (could have moved back a res cp)
            if success:
                if self.last_res_cp < self.planner.current_res_cp:
                    relevant_outcome = 5
                else:
                    relevant_outcome = 3
            else:
                relevant_outcome = 0

            # self.update_reset_probas(move_near_id, select_id, relevant_outcome)
            self.update_reset_probas(move_near_id, select_id, rel_outcome_update)

            # FutureTODO: If we have extra time, we would ideally like to keep learning to achieve the subgoal
            #           even if we have already done so. (e.g. reset to different start state and run another episode)
            #   For now if we manage to achieve subgoal, return control to rest of agent
            #   Also have to keep in mind whether or not irreversible actions make this impossible or not (usually will be)
            if success:
                if relevant_outcome != 0:
                    print(Fore.GREEN + "Achieved subgoal for operator {} with time left {}".format(operator_str,
                                                                                                   time_to_train - (
                                                                                                               time.time() - start)))
                    return True
                else:
                    print(Fore.YELLOW + "Dropped back a cp in learning of operator {} with time left {}".format(
                        operator_str, time_to_train - (time.time() - start)))
                    return False
            elif self.novel_item_encountered:
                return False
            elif self.exceeded_step_cost:
                return False

            longest_ep_time = max(longest_ep_time, time.time() - ep_start_time)
            ep_num += 1
            ep_start_time = time.time()

            # Reset agent to some 'interesting' state between 'episodes'
            if ep_num % reset_every == 0:
                move_near_id, select_id = self.reset_to_interesting_state()
                if self.found_relevant_during_reset:
                    return True

                if self.exceeded_step_cost:
                    return False

        # Times up, did not manage to complete subgoal
        print(Fore.LIGHTRED_EX + "Did not achieve goal for operator {}".format(operator_str))
        return False

    # Run single 'episode' of learning
    def run_learning_episode(self, ep_t_limit, agent_ind, evaluate=False):
        if self.operator_names[agent_ind].startswith('moveTo'):
            self.control_agent = self.moveToAgent
        else:
            self.control_agent = self.learning_agents[agent_ind]

        self.control_agent_ind = agent_ind
        ep_t = 0
        done = False
        operator_success = False
        obs = self.env.observation()
        info = self.env.get_info()

        self.success_funcs[agent_ind] = self.create_success_funcs[agent_ind](obs, info)

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
            obs, rew, success, info = self.step_env(orig_obs=obs, info=info, evaluate=evaluate, done=done)
            relevant_outcome = max(relevant_outcome, possible_outcomes[self.last_outcome])

            if self.novel_item_encountered:
                return False  , relevant_outcome
            elif self.exceeded_step_cost:
                return False  , relevant_outcome

            ep_t += 1
            # Need done to be sent in last timestep to agent in step_env
            if ep_t >= ep_t_limit - 2 and not self.placed_tap:
                done = True

            if (ep_t >= ep_t_limit - 1 or success) and not self.placed_tap:
                return success  , relevant_outcome

            if self.env.game_over:
                print(
                    Fore.YELLOW + "[run_learning_episode] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                return self.env.game_success  , relevant_outcome

    ############## Exploration Functions ##############
    # Explore until timeout or finding plannable state - use dynamics checker to sense novelties and reach out to planner
    # Rather than resetting to start state between exploration episodes, reset to interesting states, modifying
    #   probabilities based on result
    def explore_for_time(self, time_to_train, reset_every=1):
        start = time.time()
        longest_ep_time = 0
        ep_num = 0

        if (not USING_TOURNAMENT_MANAGER) and time_to_train < 30:
            print(
                Fore.YELLOW + 'Entering learning with less than 30 secs remaining, ending execution to prevent going over time')
            return False, None

        # Should we just start exploring at failed step?
        # move_near_id, select_id = self.reset_to_interesting_state()
        move_near_id, select_id = None, None
        ep_start_time = time.time()

        #Go until game over for whatever reason (keep track of time and step cost personally if not using tournament manager)
        while ((not USING_TOURNAMENT_MANAGER) and time_to_train - (time.time() - start) > longest_ep_time + 20) \
                or (USING_TOURNAMENT_MANAGER and not self.env.game_over):

            found_plannable, relevant_outcome = self.run_exploration_episode(50)

            if self.exceeded_step_cost:
                return False, None

            # break on finding novel item type, need to reset learners and buffers
            if relevant_outcome == 6:
                return False, relevant_outcome

            # change if reset_every isn't 1 (really should be)
            self.update_reset_probas(move_near_id, select_id, relevant_outcome)

            if relevant_outcome == 5:
                return False, relevant_outcome

            # Once finding plannable state, convert findings to goal and switch to learning
            # Diff notion of found_plannable, could be closer to goal but not technically plannable
            if found_plannable:
                return True, relevant_outcome

            longest_ep_time = max(longest_ep_time, time.time() - ep_start_time)
            ep_num += 1

            # Reset agent to some 'interesting' state between 'episodes' (near obj, holding obj, ...)
            if ep_num % reset_every == 0:
                move_near_id, select_id = self.reset_to_interesting_state()
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

                if self.exceeded_step_cost:
                    return False, None

            ep_start_time = time.time()

        # Times up, did not manage to complete subgoal
        print(Fore.YELLOW + "Did not find plannable state")
        return False, relevant_outcome

    # Run single episode using exploration agent
    def run_exploration_episode(self, ep_t_limit):
        self.control_agent = self.exploration_agent
        self.control_agent_ind = None

        # assert control agent is exploration agent
        ep_t = 0
        done = False
        obs = self.env.observation()
        info = self.env.get_info()

        # outcomes: irrelevant, detrimental, beneficial, recovered, plannable
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

            if self.exceeded_step_cost:
                return False, None

            relevant_outcome = max(relevant_outcome, possible_outcomes[self.last_outcome])

            # break on encountering novel item type - need to reset learners and buffers
            if relevant_outcome == 6:
                return False, relevant_outcome

            ep_t += 1
            # Need done to be sent in last timestep to agent in step_env
            if ep_t >= ep_t_limit - 2 and not self.placed_tap:
                done = True

            if (ep_t >= ep_t_limit - 1 or found_plannable) and not self.placed_tap:
                return found_plannable, relevant_outcome

            if self.env.game_over:
                print(
                    Fore.YELLOW + "[run_exploration_episode] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                return self.env.game_success, None

    # Create agent to be used for exploration
    def create_exploration_agent(self):
        if self.exploration_agent is not None:
            print(Fore.CYAN + 'Returning to exploration after exploration already occured for previous operator, reusing old exploration agent')
        else:
            default_param_dict['eps_lambda'] = -math.log(0.01) / 4000.
            agent = DQNLambda_Agent(self.seed, self.env, scope='exploration', session=self.session,
                                    **default_param_dict)
            self.exploration_agent = agent

        # Either add first time or overwrite existing exploration reset probas
        self.add_reset_probas(exploration=True)
        self.control_agent = self.exploration_agent

    # TODO: Learn these probabilities in a smarter way, it's in a simpler space than the actual task
    #   so we don't need a whole new FA learner for it, we want it to be quicker, but could still use
    #   a smarter method than this
    # Reset probabilies for reset_to_interesting_state()
    #   Modified based on result of subsequent 'episodes'
    #   Vary slightly depending on goal
    def add_reset_probas(self, exploration=False):
        # Initializing probabilites of selecting reset_to_interesting_state details
        # Take softmax of valid option values at the current step, modify value based on response
        novel_items = self.env.novel_items.copy()
        novel_item_ids = np.array([self.env.mdp_items_id[novel_item] for novel_item in novel_items]).astype(int)

        # First N == object, N+1 to 2N == entities
        reset_near_values = np.zeros(self.env.num_types * 2) + 5
        # entities have increased value - always novel so always want to go to when they appear at first
        reset_near_values[self.env.num_types:] += 5
        # novel objects have increased value
        reset_near_values[novel_item_ids] += 5
        reset_near_values[novel_item_ids + self.env.num_types] += 5
        # Should only allow going near bedrock or air after much vain exploration (or do we want to do never?)
        reset_near_values[self.env.mdp_items_id['minecraft:bedrock']] = 0.1
        # Moving to air is essentially the same as random reset
        reset_near_values[self.env.mdp_items_id['minecraft:air']] = 1

        if self.failure_info['operator'].name.split()[0] == 'moveTo':
            # attempt to go as near as possible to goal of moveTo with higher probability
            reset_near_values[self.env.mdp_items_id[self.failure_info['operator'].name.split()[1]]] += 10
        elif self.failure_info['operator'].name.split()[0] == 'break':
            # attempt to go as near as possible to goal of break with higher probability
            reset_near_values[self.env.mdp_items_id[self.failure_info['operator'].name.split()[1]]] += 5

        # Only novel items are distinguishable in select case
        reset_select_values = np.ones(self.env.num_types)
        reset_select_values[novel_item_ids] += 9

        # replace existing
        if exploration and len(self.reset_near_values) > 0:
            self.reset_near_values[0] = reset_near_values
            self.reset_select_values[0] = reset_select_values
        # create new
        else:
            # if len(self.reset_near_values) == 0:
            self.reset_near_values.append(reset_near_values)
            self.reset_select_values.append(reset_select_values)

    # Code to actually reset to the interesting state selected
    def reset_to_interesting_state(self, first_reset=False):
        self.found_relevant_during_reset = False
        self.resetting_state = True
        print(Fore.LIGHTCYAN_EX + 'resetting to interesting state')
        if self.mode == 'exploration':
            reset_ind = 0
        else:
            reset_ind = self.control_agent_ind + 1

        selected_item = None
        item = None

        # Make sure movement dynamics have not been altered such that motion planning no longer works
        if self.can_motion_plan:
            # Get all valid possible targets
            blocks_in_world = list(self.env.items_location.keys())
            blockids_in_world = np.array([self.env.mdp_items_id[block] for block in blocks_in_world])

            entities_in_world = list(self.env.entities_location.keys())
            entityids_in_world = np.array(
                [self.env.mdp_items_id[entity] + self.env.num_types for entity in entities_in_world])

            all_ids_in_world = list(np.concatenate((blockids_in_world, entityids_in_world)).astype(int))

            # Don't want to recreate astar agent every time if planning to a certain type fails
            # Create AStar agent for current env config if not supplied
            grid_size = 1.0
            robot_radius = 0.9

            # obstacle positions
            ox, oy = [], []
            for r in range(len(self.env.binary_map[0])):
                for c in range(len(self.env.binary_map[1])):
                    if self.env.binary_map[r][c] == 1:
                        ox.append(c)
                        oy.append(r)
            a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

            # Take softmax and sample goal to move near - if fail to move near an object (blocked), try another
            move_success = False
            while len(all_ids_in_world) > 0 and not move_success:
                move_near_values = self.reset_near_values[reset_ind][all_ids_in_world]
                move_near_probas = move_near_values / sum(move_near_values)

                move_near_id = np.random.choice(all_ids_in_world, p=move_near_probas)
                if move_near_id < self.env.num_types:
                    entity = False
                    item = self.env.all_items[move_near_id]
                else:
                    entity = True
                    item = self.env.all_items[move_near_id - self.env.num_types]

                plan_success, move_success, info = self.move_near(instance_type=item, entity=entity, a_star=a_star,
                                                                  nearest=first_reset)

                # exit on exceeded step cost
                if self.exceeded_step_cost:
                    self.resetting_state = False
                    return None, None

                if self.found_relevant_during_reset:
                    self.resetting_state = False
                    return None, None

                if self.env.game_over:
                    print(Fore.YELLOW + "[reset_to_interesting_state] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                    return None, None

                if not move_success:
                    if plan_success:
                        if not self.failed_last_motion_plan:
                            print(
                                Fore.LIGHTYELLOW_EX + 'Failed motion planning execution, but could be due to items popping up mid execution or rare bug in motion planner -> allowing another attempt')
                            self.failed_last_motion_plan = True
                        else:
                            print(Fore.YELLOW + 'Failed motion planning execution twice in a row, disabling')
                            self.can_motion_plan = False
                        move_near_id = None
                        break
                    else:
                        all_ids_in_world.remove(move_near_id)
                else:
                    self.failed_last_motion_plan = False

            # Modify target value based on response
            if not move_success:
                print(
                    Fore.YELLOW + 'Couldnt moveTo any existing object type using A*, what happened? No blocks left in world? Surrounded by obstacles?')
                move_near_id = None
                self.last_reset_pos = None
        else:
            if self.found_relevant_exp_state < 5:
                print(Fore.YELLOW + 'Cannot motion plan and we havent found many relevant state in exploration yet, sending sequence of MOVE commands to hopefully explore in a different location')
                obs = self.env.observation()
                info = self.env.get_info()
                for _ in range(np.random.randint(10)):
                    obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)
                obs, rew, done, info = self.step_env(self.env.actions_id['TURN 90'], obs, info)
                for _ in range(np.random.randint(10)):
                    obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)

                if self.found_relevant_during_reset:
                    self.resetting_state = False
                    return None, None

            move_near_id = None

        # don't allow select if exceeded step cost
        if self.exceeded_step_cost:
            return None, None
        elif self.env.game_over:
            return None, None

        # Select item
        items_in_inv = self.env.inventory_quantity_dict.keys()
        if '' in items_in_inv:
            items_in_inv.remove('')
        if len(items_in_inv) > 0:
            itemids_in_inv = np.array([self.env.mdp_items_id[item] for item in items_in_inv])
            select_values = self.reset_select_values[reset_ind][itemids_in_inv]
            select_probas = np.exp(select_values) / sum(np.exp(select_values))
            select_id = np.random.choice(itemids_in_inv, p=select_probas)
            selected_item = self.env.all_items[select_id]
            self.select_item(selected_item)
        else:
            select_id = None

        self.resetting_state = False
        print(Fore.LIGHTCYAN_EX + 'near {}, holding {}'.format(item, selected_item))

        return move_near_id, select_id

    # TODO: overhaul how this decision making is done - don't want to use whole big FA agent since
    #   the space should be much simpler, but could use a simpler RL agent which is slightly more
    #   informed to learn what to reset to based on res_cp and last reset.
    # Update reset probas after an episode of exploring or learning
    def update_reset_probas(self, move_near_id, select_id, relevant_outcome):
        print('updating reset probas')
        if self.mode == 'exploration':
            reset_ind = 0
        else:
            reset_ind = self.control_agent_ind + 1

        # set value cap for plannable state higher than beneficial state (will decrement values on others, so irrelevant)
        value_cap = 50

        # Found nothing
        if relevant_outcome == 0:
            value_change = -1
        # Found detrimental dynamics novelty
        elif relevant_outcome == 1 or relevant_outcome == 6:
            value_change = -5
        # Found beneficial dynamics novelty
        elif relevant_outcome == 2:
            value_change = 2
            value_cap = 15
        # plannable or recovered or cp step
        else:
            value_change = 10

        if move_near_id is not None:
            if self.reset_near_values[reset_ind][move_near_id] < value_cap:
                self.reset_near_values[reset_ind][move_near_id] = max(1, min(value_cap,
                                                                             self.reset_near_values[reset_ind][
                                                                                 move_near_id] + value_change))
        if select_id is not None:
            if self.reset_select_values[reset_ind][select_id] < value_cap:
                self.reset_select_values[reset_ind][select_id] = max(1, min(value_cap,
                                                                            self.reset_select_values[reset_ind][
                                                                                select_id] + value_change))

        # update auxiliary moveTo learner as well if exploring moveTo failure
        if self.mode == 'exploration' and self.failure_info['operator'].name.split()[0] == 'moveTo':
            learner_ind = self.operator_names.index(self.failure_info['operator'].name) + 1
            if move_near_id is not None:
                if self.reset_near_values[learner_ind][move_near_id] < value_cap:
                    self.reset_near_values[learner_ind][move_near_id] = max(1, min(value_cap,
                                                                                   self.reset_near_values[learner_ind][
                                                                                       move_near_id] + value_change))
            if select_id is not None:
                if self.reset_select_values[learner_ind][select_id] < value_cap:
                    self.reset_select_values[learner_ind][select_id] = max(1, min(value_cap,
                                                                                  self.reset_select_values[learner_ind][
                                                                                      select_id] + value_change))

    ############## Util Functions ##############

    def select_item(self, item_to_select=None):
        # Randomly chose object to select (exploration)
        if item_to_select is None:
            interesting_items = self.env.novel_items.copy()

            # First try selecting novel item
            while len(interesting_items) > 0:
                interesting_item = interesting_items[np.random.randint(len(interesting_items))]
                if interesting_item in self.env.inventory_quantity_dict:
                    self.step_env(self.env.actions_id['SELECT_ITEM {}'.format(interesting_item)],
                                  store_transition=False)
                    return interesting_item
                else:
                    del interesting_item_locations[ind]

            # Select random item otherwise
            if len(self.env.inventory_quantity_dict) > 0:
                item = np.random.choice(list(self.env.inventory_quantity_dict.keys()))
                self.step_env(self.env.actions_id['SELECT_ITEM {}'.format(item)], store_transition=False)
                return item
            # Can't select anything, don't have anything
            else:
                return None
        # Choose specific item to select (learning)
        else:
            if item_to_select in self.env.inventory_quantity_dict:
                self.step_env(self.env.actions_id['SELECT_ITEM {}'.format(item_to_select)], store_transition=False)
                return item_to_select
            else:
                print(
                    Fore.YELLOW + 'Cannot reset to start state holding object {}, it is no longer present in the inventory'.format(
                        item_to_select))
                return None

    # TODO: most of these conditionals are never used, and the functionality for those are just wrong. Remove them or update
    # Move agent either to goal pose, near some random instance of an object type, or near random instance of random type
    # Prioritizes novel entities -> novel objects -> entities -> objects
    def move_near(self, instance_type=None, entity=False, goal_pose=None, relcoord=None, a_star=None, nearest=True):
        # Always plan from agent's current location
        sx = self.env.player['pos'][0]
        sy = self.env.player['pos'][2]

        # Create AStar agent for current env config if not supplied
        if a_star is None:
            grid_size = 1.0
            robot_radius = 0.9

            # obstacle positions
            ox, oy = [], []
            for r in range(len(self.env.binary_map[0])):
                for c in range(len(self.env.binary_map[1])):
                    if self.env.binary_map[r][c] == 1:
                        ox.append(c)
                        oy.append(r)
            a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

        # always first plan towards goal_location if explicitly supplied
        if goal_pose is not None:
            if relcoord is None:
                relcoord = (0, 0)
            # orientation
            rx, ry = a_star.planning(sx, sy, goal_pose[0] + relcoord[0], goal_pose[1] + relcoord[1])
            ro = goal_pose[2]
            info = {'moveType': 'goalPose',
                    'relcoord': relcoord,
                    'orientation': ro
                    }
            if len(rx) > 1:
                self.moveToUsingPlan(sx, sy, rx, ry, ro)
                move_success = (int(rx[0]) == self.env.player['pos'][0]) and (int(ry[0]) == self.env.player['pos'][2])
                return True, move_success, info
            else:
                return False, False, info
        # TODO: believe this is the only conditional actually used atm
        # Otherwise go to specific type of object if supplied
        elif instance_type is not None:
            if entity:
                if nearest:
                    plan_success, move_success, info = self.plan_and_go_to_nearest([instance_type],
                                                                                   self.env.entities_location,
                                                                                   a_star, sx, sy)
                else:
                    plan_success, move_success, info = self.plan_and_go_to_random([instance_type],
                                                                                  self.env.entities_location,
                                                                                  a_star, sx, sy)
            else:
                if nearest:
                    plan_success, move_success, info = self.plan_and_go_to_nearest([instance_type],
                                                                                   self.env.items_location,
                                                                                   a_star, sx, sy)
                else:
                    plan_success, move_success, info = self.plan_and_go_to_random([instance_type],
                                                                                  self.env.items_location,
                                                                                  a_star, sx, sy)
            info['entity'] = entity
            info['moveType'] = 'instanceType'
            return plan_success, move_success, info
        # Otherwise choose instance type and relcoord
        else:
            # if instance_type is None:
            # Options of object to go to:
            #  0. novel entities
            #  1. novel blocks
            #  2. entities
            #  3. blocks
            # Bias probability to plan towards objects in these categories going down the list
            # Are entities or novel blocks more interesting? Both should never be the case pre-novelty

            # 60% chance to immediately try going to novel object
            # If not, there's still a chance when randomly choosing between all blocks
            if len(self.env.novel_items) > 0:
                if np.random.random() < 0.6:
                    plan_success, move_success, info = self.plan_and_go_to_random(self.env.novel_items,
                                                                                  self.env.items_location, a_star, sx,
                                                                                  sy)
                    if plan_success:
                        info['moveType'] = 'randomType'
                        info['entity'] = False
                        return plan_success, move_success, info

            # Try going to any entities first if any exist - going to an entity should remove it, so should be safe to always try first
            interesting_entities = list(self.env.entities_location.keys())
            if len(interesting_entities) > 0:
                plan_success, move_success, info = self.plan_and_go_to_random(interesting_entities,
                                                                              self.env.entities_location, a_star, sx,
                                                                              sy)
                if plan_success:
                    info['moveType'] = 'randomType'
                    info['entity'] = True
                    return plan_success, move_success, info

            # Then try going to random block type
            interesting_blocks = list(self.env.items_location.keys())
            # Don't care about bedrock (unless we want to check explicitly if there's bedrock not on the outer limits)
            interesting_blocks.remove('minecraft:bedrock')
            interesting_blocks.remove('minecraft:air')
            plan_success, move_success, info = self.plan_and_go_to_random(interesting_blocks, self.env.items_location,
                                                                          a_star, sx, sy)
            info['moveType'] = 'randomType'
            info['entity'] = False
            return plan_success, move_success, info

    # TODO: clean this and plan_to_random, for first instance of reset we want to do nearest, if that
    #   doesn't work then do random
    def plan_and_go_to_nearest(self, interesting_items, items_location, a_star, sx, sy):
        # Then sample interesting blocks and go to them
        while len(interesting_items) != 0:
            # randomly sample item key of set to navigate towards (should mostly be len 1)
            item_ind = np.random.randint(len(interesting_items))
            interesting_item = interesting_items[item_ind]
            try:
                interesting_item_locations = items_location[interesting_item].copy()
            except:
                del interesting_items[item_ind]
                continue

            # If few enough items, just iterate through and order all in terms of distance
            if len(interesting_item_locations) <= 10:
                interesting_item_dists = []
                for i in range(len(interesting_item_locations)):
                    interesting_instance = interesting_item_locations[i]
                    locs = interesting_instance.split(',')
                    dist = (sx - int(locs[0])) ** 2 + (sy - int(locs[2])) ** 2
                    interesting_item_dists.append(dist)
                while len(interesting_item_locations) != 0:
                    # randomly sample instance of item key to navigate towards
                    # ind = np.random.randint(len(interesting_item_locations))
                    # take nearest remaining instance
                    ind = np.argmin(interesting_item_dists)
                    interesting_instance = interesting_item_locations[ind]
                    locs = interesting_instance.split(',')
                    gx = int(locs[0])
                    gy = int(locs[2])
                    # Can't actually go into the item, so randomly sample point next to it to go to
                    relcoord = np.random.randint(4)
                    rx, ry = [], []

                    # FutureTODO: clean up this check
                    if (self.mode == 'exploration' and self.resetting_state and self.failure_info is not None and
                        self.failure_info['operator'].name.split()[0] == 'moveTo' and
                        self.failure_info['operator'].name.split()[1] == interesting_item) or \
                            (self.mode == 'learning' and self.resetting_state and
                             self.operator_names[self.control_agent_ind].split()[0] == 'moveTo' and
                             self.operator_names[self.control_agent_ind].split()[1] == interesting_item):
                        dists = [1, 2, 3]
                    else:
                        dists = [1]
                    for dist in dists:
                        num_attempts = 0
                        # otherwise object is unreachable
                        while len(rx) < 2 and num_attempts < 4:
                            if relcoord == 0:
                                relx, rely = 1 * dist, 0
                                ro = 'WEST'
                            elif relcoord == 1:
                                relx, rely = -1 * dist, 0
                                ro = 'EAST'
                            elif relcoord == 2:
                                relx, rely = 0, 1 * dist
                                ro = 'NORTH'
                            elif relcoord == 3:
                                relx, rely = 0, -1 * dist
                                ro = 'SOUTH'
                            rx, ry = a_star.planning(sx, sy, gx + relx, gy + rely)
                            relcoord = (relcoord + 1) % 4
                            num_attempts += 1
                        if len(rx) > 1:
                            break

                    # Found plan
                    if len(rx) > 1:
                        self.last_reset_pos = (gx, gy)
                        self.moveToUsingPlan(sx, sy, rx, ry, ro)
                        move_success = (int(rx[0]) == self.env.player['pos'][0]) and (
                                int(ry[0]) == self.env.player['pos'][2])
                        info = {'instance_type': interesting_item,
                                'relcoords': (relx, rely),
                                'orientation': ro,
                                'end_pos': (rx[0], ry[0])
                                }
                        # Couldn't move next to but moved near
                        if dist > 1:
                            return False, move_success, info
                        else:
                            return True, move_success, info
                    # Unreachable, delete location and keep trying
                    else:
                        del interesting_item_locations[ind]
                        del interesting_item_dists[ind]

                interesting_items.remove(interesting_item)

            # #otherwise search out from agent and try one by one (don't want to get stuck on case where they spawn
            # # a bunch of instances
            else:
                print("TODO: implement spiral search for nearest goal instance, too many instances, picking random")
                success, move_success, info = self.plan_and_go_to_random([interesting_item], items_location, a_star, sx,
                                                                         sy)
                if success:
                    return success, move_success, info
                interesting_items.remove(interesting_item)
        # Did not find plan for any object
        info = {'instance_types': interesting_items,
                # 'relcoords': (relx,rely),
                # 'orientation': ro
                }
        return False, False, info

    # Goes to random instance of random item in interesting_items list (if possible)
    def plan_and_go_to_random(self, interesting_items, items_location, a_star, sx, sy):
        while len(interesting_items) != 0:
            # randomly sample item key to navigate towards
            item_ind = np.random.randint(len(interesting_items))
            interesting_item = interesting_items[item_ind]
            try:
                interesting_item_locations = items_location[interesting_item].copy()
            except:
                del interesting_items[item_ind]
                continue

            # randomly sample instance of item key to navigate towards
            while len(interesting_item_locations) != 0:
                ind = np.random.randint(len(interesting_item_locations))
                interesting_instance = interesting_item_locations[ind]
                locs = interesting_instance.split(',')
                gx = int(locs[0])
                gy = int(locs[2])
                # Can't actually go into the item, so randomly sample point next to it to go to
                # Check if relcoord to item is vacant or reachable, otherwise we're wasting an opportunity
                relcoord = np.random.randint(4)
                # start with sampled relcoord, then try iterating over other possibilities
                rx, ry = [], []

                # FutureTODO: clean up this check
                if (self.mode == 'exploration' and self.resetting_state and self.failure_info is not None and
                    self.failure_info['operator'].name.split()[0] == 'moveTo' and
                    self.failure_info['operator'].name.split()[1] == interesting_item) or \
                        (self.mode == 'learning' and self.resetting_state and
                         self.operator_names[self.control_agent_ind].split()[0] == 'moveTo' and
                         self.operator_names[self.control_agent_ind].split()[1] == interesting_item):
                    dists = [1, 2, 3]
                else:
                    dists = [1]
                for dist in dists:
                    num_attempts = 0
                    # otherwise object is unreachable
                    while len(rx) < 2 and num_attempts < 4:
                        if relcoord == 0:
                            relx, rely = 1 * dist, 0
                            ro = 'WEST'
                        elif relcoord == 1:
                            relx, rely = -1 * dist, 0
                            ro = 'EAST'
                        elif relcoord == 2:
                            relx, rely = 0, 1 * dist
                            ro = 'NORTH'
                        elif relcoord == 3:
                            relx, rely = 0, -1 * dist
                            ro = 'SOUTH'
                        rx, ry = a_star.planning(sx, sy, gx + relx, gy + rely)
                        relcoord = (relcoord + 1) % 4
                        num_attempts += 1
                    if len(rx) > 1:
                        break

                # Found plan
                if len(rx) > 1:
                    self.last_reset_pos = (gx, gy)
                    self.moveToUsingPlan(sx, sy, rx, ry, ro)
                    move_success = (int(rx[0]) == self.env.player['pos'][0]) and (
                                int(ry[0]) == self.env.player['pos'][2])
                    info = {'instance_type': interesting_item,
                            'relcoords': (relx, rely),
                            'orientation': ro,
                            'end_pos': (rx[0], ry[0])
                            }
                    # Couldn't move next to but moved near
                    if dist > 1:
                        return False, move_success, info
                    else:
                        return True, move_success, info
                # Unreachable, delete location and keep trying
                else:
                    del interesting_item_locations[ind]
            interesting_items.remove(interesting_item)

        # Did not find plan for any object
        info = {'instance_types': interesting_items,
                # 'relcoords': (relx,rely),
                # 'orientation': ro
                }
        return False, False, info

    # Given motion plan, execute steps and store trajectory
    def moveToUsingPlan(self, sx, sy, rxs, rys, ro):
        self.motion_planning = True
        # sx, sy: start pos
        # rx, ry: subsequent locations to moveTo
        # rx, ry are backwards, iterate in reverse
        obs = self.env.observation()
        info = self.env.get_info()
        for i in range(len(rxs) - 1):

            # abort if exceeded step cost
            if self.exceeded_step_cost:
                self.motion_planning = False
                return None, None
            elif self.env.game_over:
                self.motion_planning = False
                return None, None

            orientation = self.env.player['facing']
            # First location is same as current location, skip
            ind = len(rxs) - i - 2
            rx = rxs[ind]
            ry = rys[ind]

            # MOVE_RIGHT
            if sx == rx - 1:
                if orientation != 'EAST':
                    obs, rew, done, info = self.rotate_agent(orientation, 'EAST', obs, info)
                sx += 1
            # MOVE_LEFT
            elif sx == rx + 1:
                if orientation != 'WEST':
                    obs, rew, done, info = self.rotate_agent(orientation, 'WEST', obs, info)
                sx -= 1
            # MOVE_NORTH
            elif sy == ry - 1:
                if orientation != 'SOUTH':
                    obs, rew, done, info = self.rotate_agent(orientation, 'SOUTH', obs, info)
                sy += 1
            # MOVE_SOUTH
            elif sy == ry + 1:
                if orientation != 'NORTH':
                    obs, rew, done, info = self.rotate_agent(orientation, 'NORTH', obs, info)
                sy -= 1
            else:
                print("error in path plan")
                self.motion_planning = False
                return sx, sy
            obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)
        orientation = self.env.player['facing']
        if orientation != ro:
            self.rotate_agent(orientation, ro, obs, info)
        self.motion_planning = False
        return sx, sy

    # Rotate agent to face correct direction and store transitions
    def rotate_agent(self, start_o, goal_o, obs=None, info=None):
        dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
        start_val = dir_vals[start_o]
        goal_val = dir_vals[goal_o]
        num_rots = goal_val - start_val
        if num_rots == 0:
            return None, None, None, None
        elif num_rots == 1 or num_rots == -3:
            return self.step_env(self.env.actions_id['TURN 90'], obs, info)
        elif num_rots == 2 or num_rots == -2:
            obs, rew, done, info = self.step_env(self.env.actions_id['TURN 90'], obs, info)
            return self.step_env(self.env.actions_id['TURN 90'], obs, info)
        elif num_rots == 3 or num_rots == -1:
            return self.step_env(self.env.actions_id['TURN -90'], obs, info)

    # TODO: write code to convert TRADE planner response to actual operator plan steps to execute
    #Use this function to generate plans towards subgoals using TRADE planner
    def trade_plan_to_goal(self, goal='[fluent_geq inventory wooden_pogo_stick 1]', invalidated_actions=None):
        print(Fore.LIGHTYELLOW_EX + 'Not using TRADE planner atm')
        # if invalidated_actions is None:
        # invalidated_actions = self.planner.failed_operators
        # print(goal, invalidated_actions)
        return None
        #
        # #Use map between RL and TRADE invalidated_actions -> RL_TO_TRADE_INVALIDATED_OPERATOR_MAP
        # #ISSUE - not same notion of operators,
        # #   e.g. for break would have to update type of object to unbreakable, not invalidate breakBlock
        # self.sock.send(str.encode('CHECKPLAN'))

        # #self.sock.send(str.encode('tryToPlan {}'.format(goal)))
        # # self.sock.send(str.encode('tryToPlan ({}, {})'.format(goal, invalidated_actions)))

        # # wait for response
        # plan = recv_socket_data(self.sock).decode('UTF-8')
        # print(plan)
        # if plan == '':
        #     return None
        # else:
        #     return plan

    # Step the environment, store the transition, and update the appropriate agents
    # To be used rather than env.step() to prevent rewriting the update, storage, etc logic each time
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

        if store_transition is None:
            store_transition = self.encountered_novelty

        # Notion of success in step here is different in this version
        # success: found_plannable in exploration, success_func->True in learning
        success = False

        if orig_obs is None:
            orig_obs = self.env.observation()
        if info is None:
            info = self.env.get_info()

        if self.last_res_cp is None:
            res_cp = self.planner.get_current_resource_checkpoint(info)
            self.last_res_cp = res_cp

        if self.mode == 'learning' and self.control_agent_ind is None:
            print(Fore.RED + 'ERROR: must have control agent ind set to step_env during learning')

        # If we are learning or recovering moveTO, we want to reformat obs into moveTo specific (Goal location has its
        #   own relcoords and special item_id)
        if self.mode == 'learning' and self.operator_names[self.control_agent_ind].split()[0] == 'moveTo':
            obs = self.reformat_moveTo_obs(orig_obs.copy(), self.operator_names[self.control_agent_ind].split()[1])
        else:
            obs = orig_obs.copy()

        if self.control_agent is not None and store_transition:
            self.control_agent.store_obs(obs)

        permissible = True
        reason = None

        ## Choose to action ##
        if action is None:
            # Always informed random action until we've found reward, then weave in exploration agent
            if self.mode == 'exploration':
                if self.found_relevant_exp_state == 0:
                    action = self.informed_random_action(info)
                else:
                    # Do eps-greedy with greedy case being manually defined inform_random_action func
                    if np.random.random() < self.exploration_agent.epsilon:
                        action = self.informed_random_action(info)
                        permissible = True
                    else:
                        action = self.exploration_agent.get_action(obs, info, 0)
                        permissible, reason = self.check_permissible(action, info)

                    # Don't allow impermissible (irreversible) actions pre-novelty too early in exploration
                    if not permissible:
                        # don't consider placed_tap and extracted_rubber as impermissible actions in the same way
                        if reason == 'placed_tap' or reason == 'extracted_rubber' or reason == 'broke_tap':
                            if self.planner.cps_attempts_without_progress[self.planner.current_res_cp] < 4:
                                print('Impermissible action due to tap/extract sequence on step {} with action {}'.format(reason, action))
                                action = self.informed_random_action(info)
                                print('choosing {} instead'.format(action))
                                permissible = True
                        else:
                            # Only allow impermissible actions if we're still exploring from same state after 5 trials
                            if not self.impermissible_performed and self.planner.cps_attempts_without_progress[self.planner.current_res_cp] > 3:
                                print(Fore.YELLOW + 'Action chosen by agent is not permissible in pre-novelty scenario (makes task unsolvable) - allowing for exploration purposes but giving negative reward')
                                self.impermissible_performed = True
                                self.impermissible_reason = reason
                                print(Fore.YELLOW + self.impermissible_reason)
                            # Don't allow multiple impermissible actions - don't think one novelty would require multiple of these
                            else:
                                if self.planner.cps_attempts_without_progress[self.planner.current_res_cp] > 3:
                                    # Want this to be the case?
                                    print(Fore.YELLOW + 'Agent attempting to perform second, different type of impermissible action in exploration - not allowing')
                                    print(Fore.YELLOW + reason, self.impermissible_reason)
                                action = self.informed_random_action(info)
                                permissible = True

            # In learning mode
            else:
                # eps_greedy same as case 2 above, but NEVER allow impermissible (updated notion of impermissible)
                epsilon = 0 if evaluate else self.control_agent.epsilon

                if np.random.random() < epsilon:
                    action = self.informed_random_action(info)
                    permissible = True
                else:
                    action = self.control_agent.get_action(obs, info, 0)
                    permissible, reason = self.check_permissible(action, info)

                if not permissible:
                    print(Fore.YELLOW + 'Learning agent chose impermissible action {} due to reason {} - not allowing it do so'.format(self.env.all_actions[action], reason))

                    # Give dummy effect to learner with negative reward to discourage this action
                    # Otherwise since it's never allowed the agent will continuously try to use it
                    #Don't want to penalize things during placed_tap
                    if self.control_agent is not None and store_transition:
                        # What's an appropriate negative reward to give?
                        # Needs to be worse than that imposed by step_cost
                        self.control_agent.store_effect(action, -100, False)
                        self.control_agent.timesteps_trained += 1
                        self.control_agent.check_update()

                        # store original obs back into agent again for actual effect update
                        self.control_agent.store_obs(obs)

                    action = self.informed_random_action(info)
                    permissible = True

        #Execution trick: If we select the same action in a row without the observation changing,
        #   choose random instead with some probability (possible we may need to just try something
        #   multiple times so don't explicitly forbid)
        if self.env.all_actions[action] == self.last_action and np.all(orig_obs == self.last_obs):
            if np.random.random() < 0.5:
                action = self.informed_random_action(info)
                permissible = True

        ## Send action ##
        obs2, _r, _d, info_ = self.env.step(action)
        self.last_action = self.env.all_actions[action]
        self.last_obs = orig_obs.copy()

        # if game_over and success, store transition, else quit
        if self.env.game_over and not self.env.game_success:
            print(Fore.YELLOW + "[step_env] Env has indicated that the game is over and not successful, stopping execution and resetting to start next trial")
            return orig_obs, 0, True, None

        new_res_cp = self.planner.get_current_resource_checkpoint(info_)
        self.last_res_cp = new_res_cp

        if not USING_TOURNAMENT_MANAGER and self.env.accumulated_step_cost > MAX_STEP_COST:
            print(Fore.YELLOW + 'Close to exceeding the step cost limit for a trial, need to shut down everything we are doing and reset for the next trial')
            self.exceeded_step_cost = True
            return None, None, False, None

        #Check outcome of action for 'novelties' or progress
        outcome, material_diffs = self.dynamics_checker.check_for_novelty(info, self.env.all_actions[action], info_)
        self.last_outcome = outcome

        # Store transition into buffer to prepopulate subsequent learning policies
        # Want to store premodified obs if we're learning moveTo
        if store_transition and self.buffer_ind < self.buffer_capacity:
            # Changing details for buffer reward comp purposes
            if action == 'PLACE_TREE_TAP':
                info['permissible'] = 'placed_tap'
            else:
                if not permissible:
                    info['permissible'] = reason
                else:
                    info['permissible'] = None
            info['new_res_cp'] = new_res_cp
            if self.mode == 'exploration':
                info['operator_name'] = self.failure_info['operator'].name
            else:
                info['operator_name'] = self.operator_names[self.control_agent_ind]
            self.store_transition(orig_obs, action, done, info, info_)

        # This shouldn't be the case, but still need to reset if encountering a novel item during learning
        if outcome == 'novel_item':
            if self.mode == 'learning':
                print(Fore.LIGHTYELLOW_EX + 'Encountered novel item during learning - this means that the current operator doesnt require this item. TODO: Separate observation/action spaces for prior and subsequent operators?')
            self.novel_item_encountered = True
            return None, None, False, None

        ## Respond to action ##
        # In exploration mode, want to check if novelty occurred, and if so whether it was helpful or not
        if self.mode == 'exploration':
            if outcome != 'irrelevant':
                self.found_relevant_exp_state += 1

            # Care first if we've found a plannable state or recovered the exact effects of a lost operator
            if outcome == 'plannable' or outcome == 'recovered':
                # If we're exploring towards moveTo, we're already learning moveTo on the side
                # If we ever explore towards a different operator, we don't want to reward moveTo
                if outcome == 'recovered' and self.failure_info is not None and self.failure_info[
                    'operator'].name.startswith('moveTo'):
                    rew = -1
                else:
                    rew = 1000
                success = True
                # if impermissible action occured at any point during this episode, but we found a plannable state, no longer impermissibe
                if self.impermissible_performed:
                    print(
                        Fore.GREEN + 'Impermissible action {} with reason {} was performed this episode, but we found a plannable state. No longer setting as impermissible'.format(action, self.impermissible_reason))
                    try:
                        self.impermissible_actions.remove(self.impermissible_reason)
                    except:
                        pass
            # Falling back a resource checkpoint is (likely) very detrimental
            # Getting to res_cp which we've had trouble progressing past is also negatively rewarded
            # TODO How do we know if part of cycle or not? - would eventually be caught but not ideal
            elif (new_res_cp < self.planner.current_res_cp or new_res_cp in self.planner.detrimental_res_cps) \
                    and not (self.last_action == 'PLACE_TREE_TAP' or self.placed_tap):
                outcome = 'lost_progress'
                self.last_outcome = outcome
                rew = -500
                # New notion of 'success' is really just 'done'
                success = True
            # Just as important as recovered is if we've jumped forward more than one cp (should never be the case prenovelty)
            #   or one in a different way
            elif new_res_cp > self.planner.current_res_cp:
                if self.planner.current_res_cp in self.planner.detrimental_res_cps:
                    print(Fore.LIGHTGREEN_EX + 'Found way to progress past previously considered detrimental res cp, removing from detrimental set')
                    self.planner.detrimental_res_cps.remove(self.planner.current_res_cp)
                outcome = 'plannable'
                self.last_outcome == 'plannable'
                rew = 1000
                success = True
            elif outcome == 'detrimental':
                rew = -100
            elif outcome == 'irrelevant':
                # If we perform an impermissible action for exploration purposes but don't get a novel outcome,
                #  we have to negatively reward it
                # We're giving a chance to perform an impermissible action and find a plannable state, but don't
                #  want to throw away too many trials by allowing the agent to keep choosing to do so
                if permissible:
                    if self.env.all_actions[action].split()[0] == 'CRAFT' or self.env.all_actions[action].split()[
                        0] == 'SELECT_ITEM' or self.env.all_actions[action].startswith('PLACE'):
                        rew = -50
                    else:
                        rew = -1
                else:
                    rew = -1000
            elif outcome == 'beneficial':
                rew = 200
                #TODO: impermissible actions with 'beneficial' outcomes?
                #Notion of beneficial is too loose without actual planner so think it should be restricted to plannable

            # recovered_pending currently only under 1 case -> broke block but haven't picked up entity
            # Want to moveTo the block and pick it up
            elif outcome == 'recovered_pending':
                # if last timestep, extend episode to include moveTo trajectory
                done = False
                # regardless of case, we will be giving a beneficial outcome, so store effect now
                outcome == 'beneficial'
                rew = 200
                rew_ = rew - round(info_['last_step_cost'] * STEPCOST_PENALTY)

                if self.control_agent is not None and store_transition:
                    self.control_agent.store_effect(action, rew_, success or done)
                    self.control_agent.timesteps_trained += 1
                    self.control_agent.check_update()

                    if done or success:
                        self.control_agent.eps_trained += 1
                        self.control_agent.update_epsilon()

                # Need to make last step of this trajectory == 'recovered' -> done in dynamics_checker
                plan_success, move_success, _info = self.move_near('minecraft:log', entity=True)

                if plan_success and move_success:
                    print(Fore.GREEN + 'Recovered break minecraft:log, just hadnt picked up entity yet')
                    self.last_outcome = 'recovered'
                else:
                    print(
                        Fore.CYAN + 'Thought we had potentially recovered break log operator, but couldnt plan to log entity')
                    return None, None, False, None
                self.dynamics_checker.pending_entity_pickup = False

                # effect has already been stored previously - just signaling to outer func to stop execution
                return None, None, True, None
            else:
                print('ERROR, unknown outcome')
                quit()

            # If we are exploring but have lost moveTo, learn moveTo operator on the side in case we recover goal
            if self.failure_info['operator'].name.split()[0] == 'moveTo' and store_transition:
                obs_ = self.reformat_moveTo_obs(obs.copy(), self.failure_info['operator'].name.split()[1])
                self.moveToAgent.store_obs(obs_)
                succ = success or self.success_funcs[-1](obs2, info_)
                rew_ = 1000 if succ else -1
                rew_ -= round(info_['last_step_cost'] * STEPCOST_PENALTY)
                self.moveToAgent.store_effect(action, rew_, succ or done)
                self.moveToAgent.timesteps_trained += 1
                self.moveToAgent.check_update()

                if done or succ:
                    self.moveToAgent.eps_trained += 1
                    self.moveToAgent.update_epsilon()

            rew -= round(info_['last_step_cost'] * STEPCOST_PENALTY)

        # In learning mode we simply want to satisfy the goal effect set
        elif self.mode == 'learning':
            planning_success = False

            # Again, not same notion of success
            if outcome == 'plannable' or outcome == 'recovered' or new_res_cp != self.planner.current_res_cp:
                planning_success = True
                if (new_res_cp < self.planner.current_res_cp or new_res_cp in self.planner.detrimental_res_cps) and (self.last_action == 'PLACE_TREE_TAP' or self.placed_tap):
                    planning_success = False

            # Check if we've satisfied the operator's effect set
            success = self.success_funcs[self.control_agent_ind](obs2, info_)
            # If we satisfy original operator's effect set, we have recovered
            if success:
                outcome == 'recovered'
                self.last_outcome = 'recovered'
                planning_success = True

            # TODO: still intermittently check to replan using TRADE in learning
            # #If we're on the last timestep of a learning episode for novelty_bridge
            # if done and not (planning_success or success) and self.operator_names[self.control_agent_ind].startswith('novelty_bridge'):
            #     self.bridge_attempts_without_replan += 1
            #     if self.bridge_attempts_without_replan % 5 == 0 and self.can_trade_plan:
            #         self.bridge_attempts_without_replan = 0
            #         print(Fore.CYAN + "learning novelty_bridge operator - checking if we've reached a plannable state intermittently since effect set is very specific")
            #         plan = self.trade_plan_to_goal()
            #         if plan is not None:
            #             plannable = True
            #     if plannable:
            #         planning_success = True

            # Plannable states outside of our effect set should also count as successes
            if planning_success and not success:
                print(Fore.CYAN + "Found plannable state that does not fall into goal effect set for our operator")
                success = planning_success
            if success:
                if new_res_cp < self.planner.current_res_cp:  # and not self.placed_tap and not self.last_action == 'PLACE_TREE_TAP':#and self.impermissible_reason is not 'placed_tap':
                    print(Fore.YELLOW + "Moved back a resource checkpoint in the original task, negatively rewarding")
                    rew = -500
                elif new_res_cp in self.planner.detrimental_res_cps:  # and not self.placed_tap and not self.last_action == 'PLACE_TREE_TAP':#and self.impermissible_reason is not 'placed_tap':
                    print(Fore.YELLOW + "Progressed to a resource checkpoint we are having trouble moving past, negatively rewarding")
                    rew = -500
                else:
                    if self.planner.current_res_cp in self.planner.detrimental_res_cps:
                        print(Fore.LIGHTGREEN_EX + 'Found way to progress past previously considered detrimental res cp, removing from detrimental set')
                        self.planner.detrimental_res_cps.remove(self.planner.current_res_cp)
                    print(Fore.GREEN + "Making percieved progress in the task, res cp {} to {}, rewarding".format(self.planner.current_res_cp, new_res_cp))
                    rew = 1000
            elif outcome == 'beneficial' and not RESTRICT_BENEFICIAL:
                rew = 200
            else:
                if self.env.all_actions[action].split()[0] == 'CRAFT' or self.env.all_actions[action].split()[0] == 'SELECT_ITEM' or self.env.all_actions[action].startswith('PLACE'):
                    rew = -50
                else:
                    rew = -1
            rew -= round(info_['last_step_cost'] * STEPCOST_PENALTY)

        info = info_

        if self.control_agent is not None and store_transition:
            self.control_agent.store_effect(action, rew, success or done)
            self.control_agent.timesteps_trained += 1
            self.control_agent.check_update()

            if done or success:
                self.control_agent.eps_trained += 1
                self.control_agent.update_epsilon()

        ## If we're currently exploring (or learning) a way to moveTo an object due to the object being blocked,
        ##   check if we've unknowingly opened up a path so we can propogate rewards appropriately
        if not success and not self.motion_planning and self.can_motion_plan:
            # In learning
            if self.mode == 'learning' and self.operator_names[self.control_agent_ind].split()[0] == 'moveTo':
                # Two cases: spawned new instance of goal block, or some instance of a block disappeared and could have opened a path
                if material_diffs['world'][
                    self.env.mdp_items_id[self.operator_names[self.control_agent_ind].split()[1]]] > 0 or np.any(
                        material_diffs['world'] < 0):
                    plan_success, move_success, _info = self.move_near(
                        self.operator_names[self.control_agent_ind].split()[1])
                    # self.last_res_cp = self.planner.get_current_resource_checkpoint(_info)
                    if (plan_success and move_success) or self.found_relevant_during_reset:
                        self.found_relevant_during_reset = False
                        print(
                            Fore.LIGHTGREEN_EX + 'Opened up path to goal moveTo object without realizing, path planned to object and gave reward')
                        print(plan_success, move_success)
                        return None, None, True, None
            # In exploration
            elif self.failure_reason == 0 and self.failure_info['operator'].name.split()[
                0] == 'moveTo' and self.mode == 'exploration':
                # Two cases: spawned new instance of goal block, or some instance of a block disappeared and could have opened a path
                if material_diffs['world'][
                    self.env.mdp_items_id[self.failure_info['operator'].name.split()[1]]] > 0 or np.any(
                        material_diffs['world'] < 0):
                    plan_success, move_success, _info = self.move_near(self.failure_info['operator'].name.split()[1])
                    # self.last_res_cp = self.planner.get_current_resource_checkpoint(_info)
                    if (plan_success and move_success) or self.found_relevant_during_reset:
                        # want to indicate that we found it is possible to 'recover' moveTo functionality
                        self.last_outcome = 'recovered'
                        self.found_relevant_during_reset = False
                        print(
                            Fore.LIGHTGREEN_EX + 'Opened up path to goal moveTo object without realizing, path planned to object and gave reward')
                        print(plan_success, move_success)
                        return None, None, True, None

        #Indicate to stop execution during reset or motion planning
        if self.resetting_state and success:
            print(Fore.LIGHTYELLOW_EX + 'Either moved forwards or backwards during reset_to_interesting_state, stopping reset execution to take note of effect')
            self.found_relevant_during_reset = True
        elif self.motion_planning and success:
            print(Fore.LIGHTYELLOW_EX + 'Either moved forwards or backwards during motion_planning, stopping reset execution to take note of effect')
            self.found_relevant_during_reset = True
        return obs2, rew, success, info

    #Currently - actions that make the task unsolvable in the pre-novelty scenario should generally be
    #   forbidden, but it is possible that they are the solution due to some blocking novelty
    #We would like to very carefully apply them and keep track of results, TODO ideally near the end of trial time
    def check_permissible(self, action, info):
        # Check permissible
        #   no -> informed random
        #   yes -> check permissible
        action = self.env.all_actions[action]

        # If we've performed place_tree_tap, we want to immediately attempt extract and then break
        # it to pick it back up so we don't accidentally leave a tap in the world and fail on a later step
        #TODO: ideally check if there was a novelty on this step, if so, we don't want to keep this loop
        if self.last_action == 'PLACE_TREE_TAP':
            # If we try to place a tap and it doesn't work, don't want to try to extract and etc
            # Extractrubber's step cost is extremely high
            if info['block_in_front']['name'] == 'polycraft:tree_tap':
                self.placed_tap = True
                return action == 'EXTRACT_RUBBER', 'placed_tap'
        elif self.last_action == 'EXTRACT_RUBBER' and self.placed_tap:
            return action == 'BREAK_BLOCK', 'extracted_rubber'
        elif self.placed_tap and self.last_action == 'BREAK_BLOCK':
            return action == 'MOVE w', 'broke_tap'
        elif self.placed_tap and self.last_action == 'MOVE w':
            self.placed_tap = False
            if action == 'PLACE_TREE_TAP':
                # Ideally this would be a different reason, 'resetting'
                return False, 'placed_tap'

        # Can't break last tree if we don't have rubber yet
        # If lost way to get rubber, or found additional way to get rubber, then we cannot make this assumption
        if get_world_quant(info, 'minecraft:log') == 1 and get_inv_quant(info,'polycraft:sack_polyisoprene_pellets') < 1 and get_entity_quant(info, 'polycraft:sack_polyisoprene_pellets') < 1 and action == 'BREAK_BLOCK' and info['block_in_front']['name'] == 'minecraft:log' and 'break_last_tree' in self.impermissible_actions:
            return False, 'break_last_tree'
        # If recipe path changed, this needs to change. If lost way to craft, want to find craft
        elif action.startswith('CRAFT'):
            # Cannot unnecessarily craft sticks
            if action.split()[1] == 'minecraft:stick' and 'craft_unnecessary_stick' in self.impermissible_actions:
                # If there's no tree tap in the world at all, it means we havent passed the craft_tap CP
                #   can't craft too many sticks or won't have enough planks (only getting 2 logs first)
                if get_inv_quant(info, 'polycraft:tree_tap') < 1 and get_world_quant(info,'polycraft:tree_tap') < 1 and get_entity_quant(info, 'polycraft:tree_tap') < 1:
                    # Still need at least 1 stick if we got it some other way
                    return get_inv_quant(info, 'minecraft:stick') <= 1, 'craft_unnecessary_stick'
                else:
                    # Strict - cannot be holding more than 4 sticks
                    return get_inv_quant(info, 'minecraft:stick') <= 4, 'craft_unnecessary_stick'
                # Exact - calculate amt of resources in world, inv, etc etc

            # Cannot unnecessarily craft tree taps
            elif action.split()[1] == 'polycraft:tree_tap' and 'craft_unnecessary_tap' in self.impermissible_actions:
                # Strict - cannot be more than 1 tree tap in the world
                return get_inv_quant(info, 'polycraft:tree_tap') + get_world_quant(info,'polycraft:tree_tap') + get_entity_quant(info, 'polycraft:tree_tap') < 1, 'craft_unnecessary_tap'
                # Exact - calulate amt of resources in world, inv, etc etc

            # Else always allowed to craft pogostick and planks (bc planks are general purpose)

            # IF NOVEL RECIPE - allow craft once and make sure the above things don't change
            #CURRENTLY PLANNER SHOULD ALWAYS DO THIS BEFORE HANDING OFF TO RL - technically should disable this always
            elif action.split()[1] != 'minecraft:planks' and action.split()[1] != 'polycraft:wooden_pogo_stick':
                if get_inv_quant(info, action.split()[1]) > 0 or get_entity_quant(info, action.split()[1]) > 0 or get_world_quant(info, action.split()[1]) > 0:
                    return False, 'craft_new_recipe'

        return True, None

    # informed random action based on failed goal and last reset state
    def informed_random_action(self, info):
        # FutureTODO: reason about past successful trajectories
        action_pool = []
        action_values = []
        if self.last_reset_pos is not None:
            x, y = self.last_reset_pos
        for action in range(self.control_agent.n_actions):
            action_str = self.env.all_actions[action]
            if self.check_permissible(action, info)[0]:
                # #Don't allow craft actions if we don't have the recipe's components (or not in front of crafting table)
                if action_str.split()[0] == 'CRAFT':
                    item = action_str.split()[1]
                    recipe = self.env.ingredients_quantity_dict[item][0]
                    have_components = True
                    for component in recipe:
                        if get_inv_quant(info, item) < recipe[component]:
                            have_components = False
                            break

                    if have_components:
                        craft_table_needed = self.env.crafting_table_needed_dict[item][0]
                        if craft_table_needed and info['block_in_front']['name'] != 'minecraft:crafting_table':
                            continue
                        else:
                            if item == 'minecraft:planks' or item == 'minecraft:stick':
                                proba = 1
                            # Would likely be able to plan to the goal at the point of the pogo_stick craft, so
                            #  if we're even in this case we don't want to bias it too much because it's possible
                            #  that it's no longer actually possible for whatever reason
                            elif item == 'polycraft:tree_tap' or item == 'polycraft:wooden_pogo_stick':
                                proba = 2
                            # New item
                            # Never permissible for now, forcing trade agent to craft recipe
                            # Highly encourage crafting new items - but only once
                            else:
                                proba = 5
                            action_pool.append(action)
                            action_values.append(proba)

                # Only allow select item if we have it in the inventory and don't have it currently selected
                elif action_str.split()[0] == 'SELECT_ITEM':
                    if get_inv_quant(info, action_str.split()[1]) >= 1 and info['selected_item'] != action_str.split()[1]:
                        action_pool.append(action)
                        # reset will handle selects for the most part so decrease probabilities
                        action_values.append(0.25)

                # assert block in front is not air
                elif action_str == 'BREAK_BLOCK':
                    if info['block_in_front']['name'] == 'minecraft:air' or info['block_in_front']['name'] == 'minecraft:bedrock':
                        action_values.append(0.1)
                        action_pool.append(action)
                    else:
                        # TODO: encourage more if block in front is goal item?
                        action_values.append(2)
                        action_pool.append(action)
                elif action_str == 'PLACE_TREE_TAP':
                    # Can't encourage this too much because we enforce extract_rubber to follow, which is extremely expensive
                    # TODO: would be really best to handle extractRubber failure separately -
                    #   keep count of what we have tried to tap and how many times
                    if get_inv_quant(info, 'polycraft:tree_tap') > 0 and info['block_in_front']['name'] == 'minecraft:air':
                        if self.failure_info is not None and self.failure_info['operator'].name == 'extractRubber':
                            # Allow experimenting with tapping different object if extractRubber fails
                            if self.env.check_for_further_validity(any=True):
                                action_values.append(5.0)
                            # Possible tapping a tree after some condition, or from a different angle couldstill work
                            elif self.env.check_for_further_validity():
                                action_values.append(0.5)
                            # Will tapping nothing ever be helpful?
                            else:
                                action_values.append(0.05)
                            action_pool.append(action)
                        # Otherwise if we have a tap we would plan to extract rubber first and foremost
                        elif self.env.check_for_further_validity():
                            # action_values.append(5)
                            action_values.append(0.5)
                            action_pool.append(action)
                    continue
                elif action_str == 'PLACE_CRAFTING_TABLE':
                    if get_inv_quant(info, 'minecraft:crafting_table') >= 1 and info['block_in_front']['name'] == 'minecraft:air':
                        action_pool.append(action)
                        action_values.append(1)
                # assert block in front is tree tap
                elif action_str == 'EXTRACT_RUBBER':
                    if info['block_in_front']['name'] == 'polycraft:tree_tap':
                        # EXTRACTRUBBER IS SUPER EXPENSIVE, don't encourage
                        # only allow extractrubber if we're looking for a way to get rubber
                        # Either on that step in exploration or learning
                        if (self.mode == 'exploration' and self.failure_info is not None and self.failure_info['operator'].name == 'extractRubber') or \
                                (self.mode == 'learning' and self.operator_names[self.control_agent_ind] == 'extractRubber'):
                            action_pool.append(action)
                            action_values.append(5)
                        # believe this is necessary when we are forcing extract rubber and only the case then
                        elif self.placed_tap:
                            action_pool.append(action)
                            action_values.append(0.001)
                    # TODO: remove, should never be the case but want to ensure preventing crash
                    elif self.placed_tap:
                        action_pool.append(action)
                        action_values.append(0.001)
                else:
                    if action_str.split()[0] == 'MOVE' or action_str.split()[0] == 'TURN':
                        continue
                    action_pool.append(action)
                    action_values.append(1)

        # Treat movement separately, want to bias exploration to remain near the reset block
        if not self.placed_tap:
            if self.last_reset_pos is None:
                if info['block_in_front']['name'] == 'minecraft:air':
                    action_pool.append(self.env.actions_id['MOVE w'])
                    action_values.append(1)
                action_pool.append(self.env.actions_id['TURN 90'])
                action_pool.append(self.env.actions_id['TURN -90'])
                action_values.append(1)
                action_values.append(1)
            else:
                move_val = 1
                left_val = 1
                right_val = 1
                playerx = self.env.player['pos'][0]
                playery = self.env.player['pos'][2]
                playero = self.env.player['facing']
                distx = playerx - x
                disty = playery - y
                if playero == 'EAST':
                    if distx > 0:
                        move_val = 1 / (2 ** distx + 1)
                        if disty > 0:
                            left_val = 1
                            right_val = 0.5
                        elif disty < 0:
                            left_val = 0.5
                            right_val = 1
                    elif distx < 0:
                        move_val = -1 * (distx - 1)
                        if disty > 0:
                            left_val = 0.5
                            right_val = 1
                        elif disty < 0:
                            left_val = 1
                            right_val = 0.5
                elif playero == 'WEST':
                    if distx > 0:
                        move_val = 1 * (distx + 1)
                        if disty > 0:
                            left_val = 0.5
                            right_val = 1
                        elif disty < 0:
                            left_val = 1
                            right_val = 0.5
                    elif distx < 0:
                        move_val = 1 / (2 ** -(distx - 1))
                        if disty > 0:
                            left_val = 1
                            right_val = 0.5
                        elif disty < 0:
                            left_val = 0.5
                            right_val = 1
                elif playero == 'NORTH':
                    if disty > 0:
                        move_val = 1 * (disty + 1)
                        if distx > 0:
                            left_val = 0.5
                            right_val = 1.0
                        elif distx < 0:
                            left_val = 1.0
                            right_val = 0.5
                    elif disty < 0:
                        move_val = 1 / (2 ** -(disty - 1))
                        if distx > 0:
                            left_val = 1.0
                            right_val = 0.5
                        elif distx < 0:
                            left_val = 0.5
                            right_val = 1.0
                elif playero == 'SOUTH':
                    if disty > 0:
                        move_val = 1 / (2 ** (disty + 1))
                        if distx > 0:
                            left_val = 1.0
                            right_val = 0.5
                        elif distx < 0:
                            left_val = 0.5
                            right_val = 1.0
                    elif disty < 0:
                        move_val = -1 * (disty - 1)
                        if distx > 0:
                            left_val = 0.5
                            right_val = 1.0
                        elif distx < 0:
                            left_val = 1.0
                            right_val = 0.5
                if info['block_in_front']['name'] == 'minecraft:air':
                    action_pool.append(self.env.actions_id['MOVE w'])
                    action_values.append(move_val)
                action_pool.append(self.env.actions_id['TURN 90'])
                action_pool.append(self.env.actions_id['TURN -90'])
                action_values.append(left_val)
                action_values.append(right_val)
        # If we're in place-extract-break-move w operator loop, add move w on last step
        elif self.last_action == 'BREAK_BLOCK':
            action_pool.append(self.env.actions_id['MOVE w'])
            action_values.append(1)

        action_probas = action_values / np.sum(action_values)
        out_action = np.random.choice(action_pool, p=action_probas)
        return out_action

    # Reformat obs to make goal location(s) it's own id with relcoords (goal is to have general moveTo not caring about
    #  instance is has to go near)
    # Options - 1. goal location is specific location 2. goal location is any instance we want to moveTo
    # Issue with 1. In learning, we don't have an instance in mind to go to, just want to go to any
    # Issue with 2. Will likely learn to moveTo nearest instance, so we can't give a specific coord if we wanted that functionality
    # Go for 2 for now, because implementation is easier (and goal is really to go any tree)
    def reformat_moveTo_obs(self, obs, goal_type):
        agent_map_ind = (self.env.agent_view_size + 1) ** 2
        orientation_ind = agent_map_ind + 1

        # Requirements for #2
        # Edit type_id for any instance of goal_type in local_view
        goal_type_id = self.env.mdp_items_id[goal_type]
        for i in range(agent_map_ind):
            if obs[i] == goal_type_id:
                # Added entities in local map with id += len(ids)
                obs[i] = len(self.env.mdp_items_id) * 2

        # Zero out relcoords for goal_type, append to end
        relcoord_ind = orientation_ind + self.env.interesting_items_ids[goal_type] * 2
        orig_relcoords = obs[relcoord_ind:relcoord_ind + 2].copy()
        obs[relcoord_ind:relcoord_ind + 2] = [0, 0]
        obs = np.concatenate((obs, orig_relcoords))

        return obs


if __name__ == '__main__':
    # Set up socket on our side (if using trade)
    if CONNECT_TO_TRADE:
        sock_agent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_agent.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_agent.bind((HOST, TRADE_PORT))
        sock_agent.listen()
        # Will stall on this line until TRADE connects to the socket
        print(Fore.BLUE + 'Waiting for TRADE agent to accept socket connection')
        conn_agent, addr = sock_agent.accept()
        trade_sock = conn_agent
    else:
        trade_sock = None

    # Set up env
    if args['env'] == 'polycraft':
        if USING_TOURNAMENT_MANAGER and CONNECT_TO_TRADE:
            env = PolycraftMDP(True, trade_sock)
        elif USING_TOURNAMENT_MANAGER:
            env = PolycraftMDP(CONNECT_TO_TRADE, host=HOST, port=ENV_PORT)
        else:
            env = PolycraftMDP(CONNECT_TO_TRADE, host=HOST, port=ENV_PORT, task=args['reset_command'])
    elif args['env'] == 'novelgridworld':
        ##TRADE + NG works
        ##Socket_env_polycraft.py standalone seems redundant then
        ##NG standalone is much quicker and should also work

        if CONNECT_TO_TRADE:
            env = PolycraftMDP(True, trade_sock, using_ng=True)
        else:
            env = gym.make('NovelGridworld-Pogostick-v2')
            # env = inject_novelty(env, 'fencerestriction', 'hard', 'oak')
            env = inject_novelty(env, 'keywall', 'hard')
            env.reset()
            env = GridworldMDP(env, CONNECT_TO_TRADE, render=False)

    else:
        print('Unknown env \'{}\' supplied, must be one of polycraft or novelgridworld'.format(args['env']))
        quit()

    NoveltyRecoveryAgent(env, trade_sock)
