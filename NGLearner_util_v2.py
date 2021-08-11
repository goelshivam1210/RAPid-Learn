import numpy as np
from collections import defaultdict

#Util funcs used by learner copied from simulate_tournament.py
#EW: TODO - clean file and make interface for funcs with default implementation that is domain independent
#           (informed_random, check_permissble, reset_to_interesting,...)

default_param_dict = {
    'return_est_method': 'pengs-median',
    'replay_capacity': 10000,
    'history_len': 1,
    'discount': 0.99,
    'cache_size': 5000,
    'block_size': 100,
    'priority': 0,
    'learning_rate': 0.001,
    'prepopulate': 250,
    'max_epsilon': 0.2,
    'min_epsilon': 0.05,
    'eps_lambda': None,
    'batch_size': 32,
    'max_timesteps': 50000,
    # 'session': None,
    'hidden_size': 32,
    'update_freq': 50, #1,
}

LOG_STR="tree_log"
PLANK_STR="plank"
STICK_STR="stick"
CT_STR="crafting_table"
TAP_STR="tree_tap"
RUBBER_STR="rubber"
POGO_STR="pogo_stick"
WALL_STR="wall"
AIR_STR="air"
FORWARD_STR="Forward"
LEFT_STR="Left"
RIGHT_STR="Right"
BREAK_STR="Break"
# PLACE_TAP_STR="Place_tree_tap"
EXTRACT_STR="Extract_rubber"
CRAFT_STR="CRAFT_"
SELECT_STR="Select_"
APPROACH_STR="approach"

#EW: Not sure if want to use this, can remove
#SG: Would like to remove it for future cases.
def check_permissible(self, action, info):
    action = self.env.all_actions[action]

    # if self.last_action == PLACE_TAP_STR:
    #     if info['block_in_front']['name'] == TAP_STR:
    #         self.placed_tap = True
    #         return action == EXTRACT_STR, 'placed_tap'
    if self.last_action == EXTRACT_STR: #and self.placed_tap:
        return action == BREAK_STR, 'extracted_rubber'
    elif self.placed_tap and self.last_action == BREAK_STR:
        return action == FORWARD_STR, 'broke_tap'
    elif self.placed_tap and self.last_action == FORWARD_STR:
        self.placed_tap = False
        if action == PLACE_TAP_STR:
            return False, 'placed_tap'

    if get_world_quant(info, LOG_STR) == 1 and get_inv_quant(info,RUBBER_STR) < 1 and get_entity_quant(info, RUBBER_STR) < 1 and action == BREAK_STR and info['block_in_front']['name'] == LOG_STR and 'break_last_tree' in self.impermissible_actions:
        return False, 'break_last_tree'
    elif action.startswith(CRAFT_STR):
        if action[len(CRAFT_STR):] == STICK_STR and 'craft_unnecessary_stick' in self.impermissible_actions:
            if get_inv_quant(info, TAP_STR) < 1 and get_world_quant(info,TAP_STR) < 1 and get_entity_quant(info, TAP_STR) < 1:
                return get_inv_quant(info, STICK_STR) <= 1, 'craft_unnecessary_stick'
            else:
                return get_inv_quant(info, STICK_STR) <= 4, 'craft_unnecessary_stick'

        elif action[len(CRAFT_STR):] == TAP_STR and 'craft_unnecessary_tap' in self.impermissible_actions:
            return get_inv_quant(info, TAP_STR) + get_world_quant(info,TAP_STR) + get_entity_quant(info, TAP_STR) < 1, 'craft_unnecessary_tap'

        elif action[len(CRAFT_STR)] != PLANK_STR and action[len(CRAFT_STR)] != POGO_STR:
            if get_inv_quant(info, action[len(CRAFT_STR):]) > 0 or get_entity_quant(info,action[len(CRAFT_STR):]) > 0 or get_world_quant(info, action[len(CRAFT_STR):]) > 0:
                return False, 'craft_new_recipe'

    return True, None


# informed random action based on failed goal and last reset state
#TODO: pass in agent to use class vars
# SG: We are not calling this now.
def informed_random_action(self, info):
    # FutureTODO: reason about past successful trajectories
    action_pool = []
    action_values = []
    if self.last_reset_pos is not None:
        x, y = self.last_reset_pos
    for action in range(self.learning_agent.n_actions):
        action_str = self.env.all_actions[action]
        if check_permissible(self, action, info)[0]:
            # #Don't allow craft actions if we don't have the recipe's components (or not in front of crafting table)
            if action_str.split('_')[0] == 'CRAFT':
                item = action_str[6:]
                recipe = self.env.ingredients_quantity_dict[item][0]
                have_components = True
                for component in recipe:
                    if get_inv_quant(info, item) < recipe[component]:
                        have_components = False
                        break

                if have_components:
                    craft_table_needed = self.env.crafting_table_needed_dict[item][0]
                    if craft_table_needed and info['block_in_front']['name'] != CT_STR:
                        continue
                    else:
                        if item == PLANK_STR or item == STICK_STR:
                            proba = 1
                        # Would likely be able to plan to the goal at the point of the pogo_stick craft, so
                        #  if we're even in this case we don't want to bias it too much because it's possible
                        #  that it's no longer actually possible for whatever reason
                        elif item == TAP_STR or item == POGO_STR:
                            proba = 2
                        # New item
                        # Never permissible for now, forcing trade agent to craft recipe
                        # Highly encourage crafting new items - but only once
                        else:
                            proba = 5
                        action_pool.append(action)
                        action_values.append(proba)

            # Only allow select item if we have it in the inventory and don't have it currently selected
            elif action_str.split('_')[0] == 'SELECT':
                if get_inv_quant(info, action_str[len(SELECT_STR):]) >= 1 and info['selected_item'] != action_str[len(SELECT_STR):]:
                    action_pool.append(action)
                    # reset will handle selects for the most part so decrease probabilities
                    action_values.append(0.25)

            # assert block in front is not air
            elif action_str == BREAK_STR:
                if info['block_in_front']['name'] == AIR_STR or info['block_in_front'][
                    'name'] == WALL_STR:
                    action_values.append(0.1)
                    action_pool.append(action)
                else:
                    # TODO: encourage more if block in front is goal item?
                    action_values.append(2)
                    action_pool.append(action)
            # SG: Commented the portion out since we are not using place_tree_tap action now.
            # elif action_str == PLACE_TAP_STR:
            #     # Can't encourage this too much because we enforce extract_rubber to follow, which is extremely expensive
            #     # TODO: would be really best to handle extractRubber failure separately -
            #     #   keep count of what we have tried to tap and how many times
            #     if get_inv_quant(info, TAP_STR) > 0 and info['block_in_front']['name'] == AIR_STR:
            #         if self.failed_action == EXTRACT_STR:
            #             # Allow experimenting with tapping different object if extractRubber fails
            #             if self.env.check_for_further_validity(any=True):
            #                 action_values.append(5.0)
            #             # Possible tapping a tree after some condition, or from a different angle couldstill work
            #             elif self.env.check_for_further_validity():
            #                 action_values.append(0.5)
            #             # Will tapping nothing ever be helpful?
            #             else:
            #                 action_values.append(0.05)
            #             action_pool.append(action)
            #         # Otherwise if we have a tap we would plan to extract rubber first and foremost
            #         elif self.env.check_for_further_validity():
            #             # action_values.append(5)
            #             action_values.append(0.5)
            #             action_pool.append(action)
            #     continue
            elif action_str == 'PLACE_CRAFTING_TABLE':
                if get_inv_quant(info, CT_STR) >= 1 and info['block_in_front'][
                    'name'] == AIR_STR:
                    action_pool.append(action)
                    action_values.append(1)
            # assert block in front is tree tap
            elif action_str == EXTRACT_STR:
                if info['block_in_front']['name'] == TAP_STR:
                    # EXTRACTRUBBER IS SUPER EXPENSIVE, don't encourage
                    # only allow extractrubber if we're looking for a way to get rubber
                    # Either on that step in exploration or learning
                    if (self.mode == 'exploration' and self.failed_action == EXTRACT_STR) or \
                            (self.mode == 'learning' and self.failed_action == EXTRACT_STR):
                        action_pool.append(action)
                        action_values.append(5)
                        # SG: Removing this since we are not using place_tree_tap action anymore.
                    # # believe this is necessary when we are forcing extract rubber and only the case then
                    # elif self.placed_tap:
                    #     action_pool.append(action)
                    #     action_values.append(0.001)
                # TODO: remove, should never be the case but want to ensure preventing crash
                # elif self.placed_tap:
                #     action_pool.append(action)
                #     action_values.append(0.001)
            else:
                if action_str in [FORWARD_STR, LEFT_STR, RIGHT_STR]:
                    continue
                action_pool.append(action)
                action_values.append(1)


    if self.last_action == BREAK_STR:
        action_pool.append(self.env.actions_id[FORWARD_STR])
        action_values.append(1)

    action_probas = action_values / np.sum(action_values)
    out_action = np.random.choice(action_pool, p=action_probas)
    return out_action




# Code to actually reset to the interesting state selected
def reset_to_interesting_state(self, first_reset=False):
    self.found_relevant_during_reset = False
    self.resetting_state = True
    print(Fore.LIGHTCYAN_EX + 'resetting to interesting state')

    selected_item = None
    item = None

    # Make sure movement dynamics have not been altered such that motion planning no longer works
    if self.can_motion_plan:
        # Get all valid possible targets
        blocks_in_world = list(self.env.items_location.keys())
        blockids_in_world = np.array([self.env.mdp_items_id[block] for block in blocks_in_world])

        entities_in_world = list(self.env.entities_location.keys())
        entityids_in_world = np.array([self.env.mdp_items_id[entity] + self.env.num_types for entity in entities_in_world])

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
            move_near_values = self.reset_near_values[all_ids_in_world]
            move_near_probas = move_near_values / sum(move_near_values)

            move_near_id = np.random.choice(all_ids_in_world, p=move_near_probas)
            if move_near_id < self.env.num_types:
                entity = False
                item = self.env.all_items[move_near_id]
            else:
                entity = True
                item = self.env.all_items[move_near_id - self.env.num_types]

            plan_success, move_success, info = move_near(self, instance_type=item, entity=entity, a_star=a_star,
                                                              nearest=first_reset)

            #TODO: make sure this is updated such that only 'recovered' case interrupts execution
            if self.found_relevant_during_reset:
                self.resetting_state = False
                return None, None

            if self.env.game_over:
                print(Fore.YELLOW + "[reset_to_interesting_state] Env has indicated that the game is over, stopping execution and resetting to start next trial")
                return None, None

            if not move_success:
                if plan_success:
                    if not self.failed_last_motion_plan:
                        print(Fore.LIGHTYELLOW_EX + 'Failed motion planning execution, but could be due to items popping up mid execution or rare bug in motion planner -> allowing another attempt')
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
            print(Fore.YELLOW + 'Couldnt moveTo any existing object type using A*, what happened? No blocks left in world? Surrounded by obstacles?')
            move_near_id = None
            self.last_reset_pos = None
    else:
        #TODO: make sure this count is still actually used
        if self.found_relevant_exp_state < 5:
            print(Fore.YELLOW + 'Cannot motion plan and we havent found many relevant state in exploration yet, sending sequence of MOVE commands to hopefully explore in a different location')
            obs = self.env.observation()
            info = self.env.get_info()
            for _ in range(np.random.randint(10)):
                obs, rew, done, info = self.step_env(self.env.actions_id[FORWARD_STR], obs, info)
            obs, rew, done, info = self.step_env(self.env.actions_id[RIGHT_STR], obs, info)
            for _ in range(np.random.randint(10)):
                obs, rew, done, info = self.step_env(self.env.actions_id[FORWARD_STR], obs, info)

            if self.found_relevant_during_reset:
                self.resetting_state = False
                return None, None

        move_near_id = None

    # don't allow select if game over
    if self.env.game_over:
        return None, None

    # Select item
    items_in_inv = self.env.inventory_quantity_dict.keys()
    if '' in items_in_inv:
        items_in_inv.remove('')
    if len(items_in_inv) > 0:
        itemids_in_inv = np.array([self.env.mdp_items_id[item] for item in items_in_inv])
        select_values = self.reset_select_values[itemids_in_inv]
        select_probas = np.exp(select_values) / sum(np.exp(select_values))
        select_id = np.random.choice(itemids_in_inv, p=select_probas)
        selected_item = self.env.all_items[select_id]
        select_item(self, selected_item)
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
        if self.reset_near_values[move_near_id] < value_cap:
            self.reset_near_values[move_near_id] = max(1, min(value_cap,self.reset_near_values[move_near_id] + value_change))
    if select_id is not None:
        if self.reset_select_values[select_id] < value_cap:
            self.reset_select_values[select_id] = max(1, min(value_cap,self.reset_select_values[select_id] + value_change))

def select_item(self, item_to_select=None):
    # Randomly chose object to select (exploration)
    if item_to_select is None:
        interesting_items = self.env.novel_items.copy()

        # First try selecting novel item
        while len(interesting_items) > 0:
            interesting_item = interesting_items[np.random.randint(len(interesting_items))]
            if interesting_item in self.env.inventory_quantity_dict:
                self.step_env(self.env.actions_id['SELECT_{}'.format(interesting_item)],store_transition=False)
                return interesting_item
            else:
                del interesting_item_locations[ind]

        # Select random item otherwise
        if len(self.env.inventory_quantity_dict) > 0:
            item = np.random.choice(list(self.env.inventory_quantity_dict.keys()))
            self.step_env(self.env.actions_id['SELECT_{}'.format(item)], store_transition=False)
            return item
        # Can't select anything, don't have anything
        else:
            return None
    # Choose specific item to select (learning)
    else:
        if item_to_select in self.env.inventory_quantity_dict:
            self.step_env(self.env.actions_id['SELECT_{}'.format(item_to_select)], store_transition=False)
            return item_to_select
        else:
            print(
                Fore.YELLOW + 'Cannot reset to start state holding object {}, it is no longer present in the inventory'.format(
                    item_to_select))
            return None

#Given instance_type, entity, a_star, and nearest
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

    # TODO: believe this is the only conditional actually used atm
    # Otherwise go to specific type of object if supplied
    elif instance_type is not None:
        if entity:
            if nearest:
                plan_success, move_success, info = plan_and_go_to_nearest(self, [instance_type],
                                                                               self.env.entities_location,
                                                                               a_star, sx, sy)
            else:
                plan_success, move_success, info = plan_and_go_to_random(self, [instance_type],
                                                                              self.env.entities_location,
                                                                              a_star, sx, sy)
        else:
            if nearest:
                plan_success, move_success, info = plan_and_go_to_nearest(self, [instance_type],
                                                                               self.env.items_location,
                                                                               a_star, sx, sy)
            else:
                plan_success, move_success, info = plan_and_go_to_random(self, [instance_type],
                                                                              self.env.items_location,
                                                                              a_star, sx, sy)
        info['entity'] = entity
        info['moveType'] = 'instanceType'
        return plan_success, move_success, info

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

                if (self.resetting_state and self.failed_action.startswith(APPROACH_STR) and self.failed_action.split()[-1] == interesting_item):
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
                    moveToUsingPlan(self, sx, sy, rx, ry, ro)
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
            success, move_success, info = plan_and_go_to_random(self, [interesting_item], items_location, a_star, sx,
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
            if (self.resetting_state and self.failed_action.split()[0] == APPROACH_STR and self.failed_action.split()[-1] == interesting_item):
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
                moveToUsingPlan(self, sx, sy, rx, ry, ro)
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

        if self.env.game_over:
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
                obs, rew, done, info = rotate_agent(self, orientation, 'EAST', obs, info)
            sx += 1
        # MOVE_LEFT
        elif sx == rx + 1:
            if orientation != 'WEST':
                obs, rew, done, info = rotate_agent(self, orientation, 'WEST', obs, info)
            sx -= 1
        # MOVE_NORTH
        elif sy == ry - 1:
            if orientation != 'SOUTH':
                obs, rew, done, info = rotate_agent(self, orientation, 'SOUTH', obs, info)
            sy += 1
        # MOVE_SOUTH
        elif sy == ry + 1:
            if orientation != 'NORTH':
                obs, rew, done, info = rotate_agent(self, orientation, 'NORTH', obs, info)
            sy -= 1
        else:
            print("error in path plan")
            self.motion_planning = False
            return sx, sy
        obs, rew, done, info = self.step_env(self.env.actions_id[FORWARD_STR], obs, info)
    orientation = self.env.player['facing']
    if orientation != ro:
        rotate_agent(self, orientation, ro, obs, info)
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
        return self.step_env(self.env.actions_id[RIGHT_STR], obs, info)
    elif num_rots == 2 or num_rots == -2:
        obs, rew, done, info = self.step_env(self.env.actions_id[RIGHT_STR], obs, info)
        return self.step_env(self.env.actions_id[RIGHT_STR], obs, info)
    elif num_rots == 3 or num_rots == -1:
        return self.step_env(self.env.actions_id[LEFT_STR], obs, info)


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
    reset_near_values[self.env.mdp_items_id[WALL_STR]] = 0.1
    # Moving to air is essentially the same as random reset
    reset_near_values[self.env.mdp_items_id[AIR_STR]] = 1

    if self.failed_action.split()[0] == APPROACH_STR:
        # attempt to go as near as possible to goal of moveTo with higher probability
        reset_near_values[self.env.mdp_items_id[self.failed_action.split()[-1]]] += 10
    elif self.failed_action.split()[0] == BREAK_STR:
        # attempt to go as near as possible to goal of break with higher probability
        reset_near_values[self.env.mdp_items_id[self.failed_action.split()[-1]]] += 5

    # Only novel items are distinguishable in select case
    reset_select_values = np.ones(self.env.num_types)
    reset_select_values[novel_item_ids] += 9

    self.reset_near_values = reset_near_values
    self.reset_select_values = reset_select_values

########################## detectors.py code ######################################

# for colored print statements
from colorama import Fore, Style, init
init(autoreset=True)

#Util functions
def get_entity_quant(info, item_str):
    try:
        amt = len(info['entities_locs'][item_str])
    except:
        amt = 0
    return amt

def get_world_quant(info, item_str):
    try:
        amt = len(info['items_locs'][item_str])
    except:
        amt = 0
    return amt

def get_inv_quant(info, item_str):
    try:
        amt = info['inv_quant_dict'][item_str]
    except:
        amt = 0
    return amt


## Convert predicate set to function for RL to check success in MDP space
# TODO: implement relative <,>,== etc. rather than just increase/decrease N
def get_create_success_func_from_predicate(predicate_str):
    create_success_func = None
    if predicate_str.split()[0] == 'holding':
        def create_success_func(init_obs, init_info):
            def success_func(obs, info):
                return info['selected_item'] == predicate_str.split()[1]
            return success_func
    elif predicate_str.split()[0] == 'facing':
        def create_success_func(init_obs, init_info):
            def success_func(obs, info):
                return info['block_in_front']['name'] == predicate_str.split()[1]
            return success_func
    # Numeric fluents - need to support <,<=,==,>=,>,decrease,increase
    elif predicate_str.split()[0] == 'increase':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    world_increase = get_world_quant(info, predicate_str.split()[2]) == get_world_quant(init_info, predicate_str.split()[2]) + int(predicate_str.split()[3])
                    # return len(info['items_locs'][predicate_str.split()[1]]) == len(init_info['items_locs'][predicate_str.split()[1]]) + int(predicate_str.split()[2])
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    inv_increase = get_inv_quant(info, predicate_str.split()[2]) == get_inv_quant(init_info, predicate_str.split()[2]) + int(predicate_str.split()[3])
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] + int(predicate_str.split()[2])
                    return inv_increase
                return success_func
        #For the 'gather log' subgoal, we don't exactly care at the end if we have 2 logs vs 1,
        # Having 1 log and 4 planks is permissible but wouldn't be caught by func above
        elif predicate_str.split()[1] == 'inventory_log':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    desired_increase = int(predicate_str.split()[2])
                    log_increase = get_inv_quant(info, LOG_STR) - get_inv_quant(init_info,LOG_STR)
                    plank_increase = get_inv_quant(info, PLANK_STR) - get_inv_quant(init_info,PLANK_STR)
                    stick_increase = get_inv_quant(info, STICK_STR) - get_inv_quant(init_info,STICK_STR)
                    return log_increase+plank_increase/4+stick_increase/8 >= desired_increase
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] + int(predicate_str.split()[2])
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'decrease':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) == get_world_quant(init_info, predicate_str.split()[2]) - int(predicate_str.split()[3])
                    # return len(info['items_locs'][predicate_str.split()[1]]) == len(init_info['items_locs'][predicate_str.split()[1]]) - int(predicate_str.split()[2])
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    inv_decrease = get_inv_quant(info, predicate_str.split()[2]) == get_inv_quant(init_info, predicate_str.split()[2]) - int(predicate_str.split()[3])
                    # inv_decrease = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] - int(predicate_str.split()[2])
                    return inv_decrease
                return success_func
    elif predicate_str.split()[0] == 'eq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) == predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) == predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) == predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'leq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) <=  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) <=  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) <=  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] <=  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'l':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) <  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) <  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) <  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] <  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'geq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) >=  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) >=  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) >=  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] >=  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'g':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) >  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) >  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) >  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] >  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    if create_success_func is None:
        print(Fore.RED+"Error, unknown predicate {} supplied".format(predicate_str))
    return create_success_func

def get_create_success_func_from_predicate_set(predicate_list):
    create_success_funcs_set = []
    for predicate_str in predicate_list:
        create_success_funcs_set.append(get_create_success_func_from_predicate(predicate_str))
    def create_success_func(init_obs, init_info):
        def success_func(obs, info):
            # return np.all([fun(init_obs, init_info)(obs, info) for fun in create_success_funcs_set])
            ss = [fun(init_obs, init_info)(obs, info) for fun in create_success_funcs_set]
            return np.all(ss)
        return success_func
    return create_success_func

## Convert prenovelty operator to function for RL to check success in MDP space
# TODO: need general increase/decrease any or inc/dec morethan/lessthan N
# Can't really enforce that operators do the same things to achieve the goal, using effect set is better than this
def get_create_success_func_from_failed_operator(operator_str):
    # e.g. 'approach {}'.format(LOG_STR) - goal is block_in_front == obj
    if operator_str.split()[0] == APPROACH_STR:
        return get_create_success_func_from_predicate('facing {}'.format(operator_str.split()[1]))
    # to the RL agent approach and pickup should have the same policy though
    if operator_str.split()[0] == 'pickup':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'decrease world {} 1'.format(operator_str.split()[1])])
    #e.g. break minecraft:log - real goal is inv increase obs, block in front air
    elif operator_str.split()[0] == BREAK_STR:
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'facing {}'.format(AIR_STR), 'decrease world {} 1'.format(operator_str.split()[1])])
    elif operator_str.split()[0] == 'place':
        return get_create_success_func_from_predicate_set(['decrease inventory {} 1'.format(operator_str.split()[1]), 'facing {}'.format(operator_str.split()[1])])
    # Don't have notion of tapped_log currently
    elif operator_str == EXTRACT_STR:
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(RUBBER_STR)])
    # For all crafting actions goal is to increase
    # Do we need to include notion of decrease as well? - to prevent losing too much?
    #   Do we just make it so the decrease has to be <= the original? Just losing more is not okay
    elif operator_str == 'Craft_plank':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(PLANK_STR)])
    elif operator_str == 'Craft_stick':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(STICK_STR)])
    elif operator_str == 'Craft_tree_tap':
        return get_create_success_func_from_predicate_set(['increase inventory {}} 1'.format(TAP_STR)])
    elif operator_str == 'Craft_pogo_stick':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(POGO_STR)])
    else:
        print(Fore.RED+"ERROR: cant create effect set from unknown operator {}".format(operator_str))


# 

####################### NG standalone env/interface for learner ###############################

# import basic libs
from collections import OrderedDict
import copy
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class NovelgridworldInterface:
    # metadata = {'render.modes': ['human']}
    """
    Goal: Craft 1 pogo_stick
    State: map, agent_location, agent_facing_id, inventory_items_quantity
    Action: {FORWARD_STR: 0, 'Left': 1, 'Right': 2, BREAK_STR: 3, PLACE_TAP_STR: 4, EXTRACT_STR: 5,
            Craft action for each recipe, Select action for each item except unbreakable items}
    """

    def __init__(self, env, render_bool=False):
        self.env = env #Pogostick env
        self.game_over = False

        ## Don't want to do socket connection if unnecessary
        # # Given attributes:
        self.render_bool = render_bool

        # Processed SENSE_RECIPES attributes:
        self.recipes = {}
        #TODO: compute this along with changes to recipe rep
        self.crafting_table_needed_dict = {PLANK_STR: [False],
                                           STICK_STR: [False],
                                           TAP_STR: [True],
                                           POGO_STR: [True]}
        # self.crafting_table_needed_dict = {}
        self.ingredients_quantity_dict = {}

        # SENSE_ALL attributes:
        self.block_in_front = {}  # name
        self.inventory_quantity_dict = {}
        self.player = {}  # pos, facing, yaw, pitch
        self.destination_pos = []
        self.entities = {}
        self.map_to_plot = []

        # Processed SENSE_ALL attributes:
        self.entities_location = {}  # contains entity's name and its locations in env.
        self.map_origin = [0, 0, 0]
        self.map_size = [0, 0, 0]  # x, z, y
        self.x_max, self.z_max, self.y_max = self.map_size
        self.items_location = {}  # contains item's name and its locations in env.
        self.items_id = {}  # contains name and ID of items in env., starts from 1
        self.map_to_plot = []  # contains item ID of items in env.
        self.binary_map = []  # contains 0 for AIR_STR, otherwise 1

        # Constants specific to PAL
        self.move_commands = ['SMOOTH_MOVE', 'MOVE', 'SMOOTH_TURN', 'TURN', 'SMOOTH_TILT']
        self.run_SENSE_ALL_and_update()
        self.run_SENSE_RECIPES_and_update()


    def send_command(self, command,  reset_from_failed_state = False, env_instance = None):

        if command.startswith == 'RESET' or command == 'RESET':
            # print ("inside NG_learner_util_send_command \n items_quantity = {}\n  items_inventory_quantity = {}\n".format(items_quantity, items_inventory_quantity))
            # print ("self.env = {}".format(self.env))
            self.env.reset( reset_from_failed_state = reset_from_failed_state, env_instance = env_instance)
            return  {'command_result' : {'command': 'RESET',
                                        'argument': None,
                                        'result': 'SUCCESS',
                                        'message': '',
                                        'stepCost': 0.0}}
        elif command.startswith('SENSE_RECIPES'):
            self.run_SENSE_RECIPES_and_update()
            return {'command_result': {'command': 'SENSE_RECIPES',
                                'argument': None,
                                'result': 'SUCCESS',
                                'message': '',
                                'stepCost': 1200.0}}

        elif command.startswith('SENSE_ALL'):
            self.run_SENSE_ALL_and_update()
            return {'command_result': {'command': 'SENSE_ALL',
                                'argument': None,
                                'result': 'SUCCESS',
                                'message': '',
                                'stepCost': 114.0}}
        elif command.startswith('SENSE_INVENTORY'):
            return {'command_result': self.run_SENSE_INVENTORY_and_update()}
        elif command.startswith('SENSE_LOCATIONS'):
            return {'command_result': self.run_SENSE_LOCATIONS_and_update()}
        elif command.startswith('SENSE_SCREEN'):
            return {'command_result': {'command': command,
                                       'argument': None,
                                       'result': 'FAIL',
                                       'message': 'SENSE_SCREEN not supported for gridworldenv',
                                       'stepCost': 6}}
        else:
            if command == FORWARD_STR:
                _, _r, _d, info = self.env.step(self.env.actions_id[FORWARD_STR])
            elif command == LEFT_STR:
                _, _r, _d, info = self.env.step(self.env.actions_id['Left'])
            elif command == RIGHT_STR:
                _, _r, _d, info = self.env.step(self.env.actions_id['Right'])
            elif command == BREAK_STR:
                _, _r, _d, info = self.env.step(self.env.actions_id[BREAK_STR])
            # SG: Removing this since we are not using place_tree_tap action anymore.
            # elif command == PLACE_TAP_STR:
            #     _, _r, _d, info = self.env.step(self.env.actions_id[PLACE_TAP_STR])
            elif command == 'PLACE_CRAFTING_TABLE':
                _, _r, _d, info = self.env.step(self.env.actions_id['Place_crafting_table'])
            elif command == EXTRACT_STR:
                _, _r, _d, info = self.env.step(self.env.actions_id[EXTRACT_STR])
            elif command.split('_')[0] == 'SELECT':
                _, _r, _d, info = self.env.step(self.env.actions_id['Select_{}'.format(command[len(SELECT_STR):])])
            elif command.split('_')[0] == 'CRAFT':
                _, _r, _d, info = self.env.step(self.env.actions_id['Craft_{}'.format(command[len(CRAFT_STR):])])
            else:
                _, _r, _d, info = self.env.step(self.env.actions_id[command])
            return {'command_result': {'command': command,
                                       'argument': None,
                                       'result': 'SUCCESS' if info['result'] else 'FAIL',
                                       'message': info['message'],
                                       'stepCost': info['step_cost']}}

    ## Start interface functions to align with polycraft connection ##
    def execute_action(self, action):
        # I think in all cases we can just do run_command and sense_all
        a_command_result, sense_all_command_result = self.run_a_command_and_sense_all(action)
        return a_command_result, sense_all_command_result

    def run_a_command_and_sense_all(self, a_command, sleep_time=0, reset_from_failed_state = False, env_instance = None):
        """
        Run a command and sense all
        Extending to handle any action type
        """
        # print ("inside run_a_command_and_sense_all \n items_quantity = {}\n  items_inventory_quantity = {}\n".format(items_inventory_quantity, items_quantity))

        a_command_output = self.send_command(a_command,  reset_from_failed_state, env_instance)
        sense_all_command_result = self.run_SENSE_ALL_and_update(parameter='NONAV')  # Give all item's location

        if self.render_bool:
            self.env.render()

        return a_command_output['command_result'], sense_all_command_result

    def run_SENSE_ALL_and_update(self, parameter=None, set_agent_id=False):
        """
        set_agent_id: set to True only when visualize_env_2d() is used for plotting
                      because agent is represented by an ID 1 on the plot
        """
        if len(self.items_id) == 0:
            self.items_id = self.env.items_id
            self.generate_id_items()

        # These will all updated in step anyway right? So we don't actually have to do anything
        # Copy relevant variables from underlying env
        # Do we have to copy things here
        self.selected_item = self.env.selected_item
        try:
            self.selected_item_id = self.items_id[self.selected_item]
        except:
            # Equate holding air to holding nothing
            self.selected_item_id = 0
        self.block_in_front = {'name':self.env.block_in_front_str}
        self.inventory_quantity_dict = {}
        unfiltered_inventory_quantity_dict = self.env.inventory_items_quantity.copy()
        for item in unfiltered_inventory_quantity_dict:
            if unfiltered_inventory_quantity_dict[item] > 0:
                self.inventory_quantity_dict[item] = unfiltered_inventory_quantity_dict[item]
        self.player = {'pos': [self.env.agent_location[1], 0, self.env.agent_location[0]],
                       'facing': self.env.agent_facing_str,
                       'yaw': 0,
                       'pitch': 0}
        self.entities = self.env.entities.copy()
        self.items_location = {}
        # construct items_location dict from map
        env_map = self.env.map.copy()
        self.map_to_plot = np.zeros((env_map.shape[0], env_map.shape[1]))  # Y (row) is before X (column) in matrix
        for r in range(env_map.shape[0]):
            for c in range(env_map.shape[1]):
                item_name = self.all_items[env_map[r][c]]
                self.items_location.setdefault(item_name, [])
                # x,z,y to match with polycraft
                self.items_location[item_name].append('{},0,{}'.format(c,r))
                self.map_to_plot[r][c] = self.items_id[item_name]

        self.binary_map = np.where(self.map_to_plot == self.items_id[AIR_STR], 0, 1)

        sense_all_command_result = {'command': 'SENSE_ALL',
                                    'argument': parameter,
                                    'result': 'SUCCESS',
                                    'message': '',
                                    'stepCost': 114.0}
        return sense_all_command_result

    def run_SENSE_RECIPES_and_update(self):
        env_recipes = self.env.recipes.copy()
        self.recipes = env_recipes

        # Finding self.ingredients_quantity_dict
        for item_to_craft in env_recipes:
            self.ingredients_quantity_dict.setdefault(item_to_craft, [])

            a_recipe = env_recipes[item_to_craft]

            ingredients_quantity_dict_for_item_to_craft = {}

            for item in a_recipe['input']:
                ingredients_quantity_dict_for_item_to_craft.setdefault(item, 0)
                ingredients_quantity_dict_for_item_to_craft[item] = a_recipe['input'][item]

            self.ingredients_quantity_dict[item_to_craft].append(ingredients_quantity_dict_for_item_to_craft)

    def run_SENSE_INVENTORY_and_update(self, inventory_output_inventory=None):
        #Filter out items we don't actually have to align with polycraft
        self.inventory_quantity_dict = {}
        unfiltered_inventory_quantity_dict = self.env.inventory_items_quantity.copy()
        for item in unfiltered_inventory_quantity_dict:
            if unfiltered_inventory_quantity_dict[item] > 0:
                self.inventory_quantity_dict[item] = unfiltered_inventory_quantity_dict[item]
        self.selected_item = self.env.selected_item
        try:
            self.selected_item_id = self.items_id[self.selected_item]
        except:
            # Equate holding air to holding nothing
            self.selected_item_id = 0

        return {'command': 'SENSE_INVENTORY', 'argument': '', 'result': 'SUCCESS', 'message': '', 'stepCost': 60.0}

    def run_SENSE_LOCATIONS_and_update(self):
        # does not update blockInFront
        self.player = {'pos': [self.env.agent_location[0], 0, self.env.agent_location[1]],
                       'facing': self.env.agent_facing_str,
                       'yaw': 0,
                       'pitch': 0}
        return {'command': 'SENSE_LOCATIONS', 'argument': '', 'result': 'SUCCESS', 'message': '', 'stepCost': 6.0}

    def generate_id_items(self):
        self.all_items = [None for _ in range(len(self.items_id))]
        for item in self.items_id:
            self.all_items[self.items_id[item]] = item


# Env class to interface between polycraft socket connection and RL agents
class GridworldMDP(NovelgridworldInterface):

    def __init__(self, env, use_trade, render=False, agent_view_size=4, restrict_space=False):
        NovelgridworldInterface.__init__(self, env, render_bool=render)

        # local view size
        self.agent_view_size = agent_view_size

        self.observation_space = None
        self.action_space = None
        self.accumulated_step_cost = 0
        self.last_step_cost = 0
        self.novel_items = []
        self.all_items = []
        self.first_space_init = False
        self.mdp_items_id = {}
        self.restrict_space = restrict_space
        if not use_trade:
            print('generating obs_action')
            self.generate_obs_action_spaces()
            self.first_space_init = False

        # if render:
        #     # self.env.render()
        #     # self.env.render()

    def set_items_id(self, items):
        if AIR_STR in items:
            #This should always be 0
            self.mdp_items_id[AIR_STR] = 0
        for item in items:
            if item != AIR_STR:
                if AIR_STR in items:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id))
                else:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id)+1)

        self.all_items = [None for i in range(len(self.mdp_items_id))]
        for item in self.mdp_items_id:
            self.all_items[self.mdp_items_id[item]] = item


        # return items_id

    def generate_obs_action_spaces(self, new_items=[]):

        novel_items = []
        # Want complete list of items_ids, not just whats in map
        # Take everything that's currently in the map and part of crafting world
        # TODO: Need to do this smarter, if completed items pops up in the middle of a
        #           round but post 'novelty-detection', this will not pick it up
        # If we have knowledge of the new item from any point, pass it in the new_items arg
        item_list = []
        for item in self.ingredients_quantity_dict:
            item_list.append(item)
            for i, ingredients_quantity_dict in enumerate(self.ingredients_quantity_dict[item]):
                for ingredient in self.ingredients_quantity_dict[item][i]:
                    item_list.append(ingredient)

        #Anything passed in new_items that wasn't found elsewhere
        for item in new_items:
            if item not in item_list:
                item_list.append(item)

        item_list = set(list(item_list) + list(self.items_location.keys())  + list(self.inventory_quantity_dict.keys()))

        if '' in item_list:
            item_list.remove('')

        #indicate new items as novel
        for item in item_list:
            if item not in [AIR_STR, WALL_STR, CT_STR, LOG_STR,
                            PLANK_STR, STICK_STR, TAP_STR, POGO_STR,
                            RUBBER_STR]:
                novel_items.append(item)
                if self.first_space_init:
                    if item not in self.novel_items:
                        print('WARNNING - Novel item {} has been discovered since last MDP action/obs space init, observations prior to and after this point will be mismatched'.format(item))
                # self.novel_items.append(item)

        self.novel_items = novel_items

        self.set_items_id(item_list)

        #Need items_id to be aligned with MDP items id
        self.items_id = self.mdp_items_id.copy()

        self.mdp_inventory_items = list(self.mdp_items_id.keys())
        if AIR_STR in self.mdp_inventory_items:
            self.mdp_inventory_items.remove(AIR_STR)
        if WALL_STR in self.mdp_inventory_items:
            self.mdp_inventory_items.remove(WALL_STR)
        #remove pogostick?

        # Generate all actions from current state of env and set action space
        # TODO: make sure at this point whatever novel object is present in env to be included
        self.manip_actions =  [FORWARD_STR,
                               LEFT_STR,
                               RIGHT_STR,
                               BREAK_STR,
                            #    PLACE_TAP_STR,
                               EXTRACT_STR]

        # Add place_crafting_table to action list -> we can break it but not put it back currently
        # self.manip_actions.append('PLACE_CRAFTING_TABLE')

        # Should crafting table be a recipe?
        self.crafting_actions = ['CRAFT_' + item for item in self.env.recipes.keys()]

        # REMOVE CRAFT crafting_table - don't think this is present in tournament, but is in API
        if 'CRAFT_{}'.format(CT_STR) in self.crafting_actions:
            self.crafting_actions.remove('CRAFT_{}'.format(CT_STR))

        self.select_actions = ['SELECT_' + item for item in self.mdp_inventory_items]

        self.all_actions = self.manip_actions + self.crafting_actions + self.select_actions
        self.actions_id = {}
        for i in range(len(self.all_actions)):
            self.actions_id[self.all_actions[i]] = i
        # print(self.actions_id)
        if not self.restrict_space:
            self.action_space = spaces.Discrete(len(self.actions_id))
        else:
            self.action_space = spaces.Discrete(len(self.manip_actions))

        # TODO: compute max possible number of items given an env? or just set to arbitrary cap
        self.max_items = 20
        # Make observation_space
        agent_map_size = (self.agent_view_size + 1) ** 2
        low_agent_map = np.zeros(agent_map_size)
        high_agent_map = (len(self.mdp_items_id)+1) * np.ones(agent_map_size)
        low_orientation = np.array([0])
        high_orientation = np.array([4])
        y_max, x_max = self.map_to_plot.shape
        # How many rel_coords items are we going to use? All possible?
        self.interesting_items = OrderedDict(self.mdp_items_id.copy())
        for item in [AIR_STR, WALL_STR, PLANK_STR, STICK_STR, RUBBER_STR,POGO_STR]:
            try:
                del self.interesting_items[item]
            except:
                continue
        #for moveTo obs reformatting
        self.interesting_items_ids = {}
        for item in self.interesting_items:
            self.interesting_items_ids[item] = len(self.interesting_items_ids)

        low_rel_coords = np.array([[-x_max, y_max] for i in range(len(self.interesting_items))]).flatten()
        high_rel_coords = np.array([[x_max, y_max] for i in range(len(self.interesting_items))]).flatten()
        low_map = np.concatenate((low_agent_map, low_orientation, low_rel_coords))
        high_map = np.concatenate((high_agent_map, high_orientation, high_rel_coords))
        low = np.concatenate((low_map,[0],np.zeros(len(self.mdp_inventory_items))))
        high = np.concatenate((high_map, [len(self.mdp_items_id)+1], self.max_items*np.ones(len(self.mdp_inventory_items))))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Need to update map using updated items_id set
        sense_all_command_result = self.run_SENSE_ALL_and_update('NONAV')
        self.accumulated_step_cost += sense_all_command_result['stepCost']

        self.num_types = len(self.mdp_items_id)+1

        print('updated items, recipes, and actions in MDP')
        print('Items: ', self.mdp_items_id)
        print('Craft items: ', self.env.recipes.keys())
        print('Actions: ', self.actions_id)
        self.first_space_init = True

    def step(self, action_id):
        action = self.all_actions[action_id]

        # Need to map to action string?
        # we should get the state of the inventory here, or the state of the world from which we need 
        # to design a reward function. and then after executing the action we will compare and give reward.
        action_result, sense_all_result = self.execute_action(action)
        self.accumulated_step_cost += action_result['stepCost'] + sense_all_result['stepCost']
        self.last_step_cost = action_result['stepCost'] + sense_all_result['stepCost']

        obs = self.observation()

        return obs, None, None, self.get_info()

    def get_info(self):
        info = {'items_locs': self.items_location.copy(), \
                'entities_locs': self.entities_location, \
                'block_in_front': self.block_in_front.copy(), \
                'inv_quant_dict': self.inventory_quantity_dict.copy(), \
                'player': self.player, \
                'selected_item': self.selected_item,
                'total_step_cost': self.accumulated_step_cost,
                'last_step_cost': self.last_step_cost}

        return info
    def mdp_gridworld_reset(self, reset_from_failed_state = False, env_instance = None):
    # def mdp_gridworld_reset(self, map_size = None, items_quantity=None, items_inventory_quantity = None):
        # print("This is that should be called")
        # print ("inside learnerUTIL \n items_quantity = {}\n  items_inventory_quantity = {}\n".format(items_quantity, items_inventory_quantity))

        self.accumulated_step_cost = 0
        # if items_quantity is not None and items_inventory_quantity is not None and map_size is not None:
        if reset_from_failed_state:                
            a_command_result, sense_all_command_result = self.run_a_command_and_sense_all('RESET', reset_from_failed_state = reset_from_failed_state, env_instance = env_instance, sleep_time=5)
        else:
            a_command_result, sense_all_command_result = self.run_a_command_and_sense_all('RESET', sleep_time=5)
        self.accumulated_step_cost += a_command_result['stepCost'] + sense_all_command_result['stepCost']
        obs = self.observation()

        return obs

    def observation(self):
        # this function computes the complete state representation and returns it

        # generate the local view
        local_view = self.generate_local_view()

        # generate relative coordinates to nearest items of interesting types
        nearest_items = self.get_nearest_items()

        # generate inventory qunatities
        inventory = self.generate_inv_list()

        facing_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}[self.player["facing"]]
        state = np.concatenate((local_view.flatten(), [facing_id], nearest_items.flatten(), [self.selected_item_id], inventory))

        return state

    def generate_local_view(self):
        # this function returns the local view of the agent in the map
        # with a binary map, 0 for nothing and 1 for something

        # pad the map
        p_width = self.agent_view_size // 2 + 1
        padded_map = np.pad(self.map_to_plot,
                            pad_width=p_width,
                            mode='constant',
                            constant_values=0.0)

        # player's current position in the padded map
        self.player_pose_x = self.player['pos'][0] + p_width
        self.player_pose_y = self.player['pos'][2] + p_width

        # cut the local view
        local_view = padded_map[ \
                     self.player_pose_y - int(self.agent_view_size / 2):self.player_pose_y + int(
                         self.agent_view_size / 2) + 1, \
                     self.player_pose_x - int(self.agent_view_size / 2):self.player_pose_x + int(
                         self.agent_view_size / 2) + 1 \
                     ]

        # rotate the local view based on the agents orientation
        rots = 0
        if (self.player["facing"] == "NORTH"):
            pass
        elif (self.player["facing"] == "WEST"):
            rots -= 1
        elif (self.player["facing"] == "SOUTH"):
            rots -= 2
        elif (self.player["facing"] == "EAST"):
            rots += 1

        local_view = np.rot90(local_view, rots)

        return local_view

    def get_nearest_items(self):
        nearest_dists = [np.inf] * len(self.mdp_items_id)
        nearest_coords = np.zeros((len(self.mdp_items_id), 2))

        envmap = self.map_to_plot.copy()
        agent_x_ = self.player['pos'][0]
        agent_y_ = self.player['pos'][2]

        if (self.player["facing"] == "NORTH"):
            agent_y = agent_y_
            agent_x = agent_x_
        elif (self.player["facing"] == "SOUTH"):
            envmap = np.flipud(envmap)
            agent_y = envmap.shape[0]-1-agent_y_
            agent_x = agent_x_
        elif (self.player["facing"] == "EAST"):
            envmap = np.rot90(envmap, 1)
            agent_y = envmap.shape[1]-1-agent_x_
            agent_x = agent_y_
        elif (self.player["facing"] == "WEST"):
            envmap = np.rot90(envmap, 3)
            agent_y = agent_x_
            agent_x = envmap.shape[0]-1-agent_y_

        # nearest dist should be manhattan distance
        for i in range(envmap.shape[0]):
            for j in range(envmap.shape[1]):
                item_id = int(envmap[i][j])
                # if item_id in self.item_ids:
                if item_id in range(len(self.mdp_items_id)):
                    dist = np.abs(agent_x - j) + np.abs(agent_y - i)
                    # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
                    if dist < nearest_dists[item_id]:
                        nearest_dists[item_id] = dist
                        nearest_coords[item_id] = (i, j)
                        # if self.env.agent_facing_id == 1:
                        if (self.player["facing"] == "SOUTH"):
                            nearest_coords[item_id] = (agent_y - i, agent_x - j)
                        else:
                            nearest_coords[item_id] = (agent_y - i, j - agent_x)
        return nearest_coords[list(self.interesting_items.values())]

    def generate_inv_list(self):
        inv = []
        for item in self.mdp_inventory_items:
            if item in self.inventory_quantity_dict:
                inv.append(self.inventory_quantity_dict[item])
            else:
                inv.append(0)
        return inv

    def check_for_further_validity(self, any=False):
        # this checks for further validity of the action based on
        # the agent needs to be two blocks away from the log and facing it
        x = self.player['pos'][0]
        y = self.player['pos'][2]
        if self.player['facing'] == 'NORTH':
            y -= 1
        elif self.player['facing'] == 'SOUTH':
            y += 1
        elif self.player['facing'] == 'EAST':
            x += 1
        elif self.player['facing'] == 'WEST':
            x -= 1

        if any:
            if self.map_to_plot[y+1][x] != self.mdp_items_id[AIR_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[WALL_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y-1][x] != self.mdp_items_id[AIR_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[WALL_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y][x-1] != self.mdp_items_id[AIR_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[WALL_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y][x+1] != self.mdp_items_id[AIR_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[WALL_STR] and self.map_to_plot[y+1][x] != self.mdp_items_id[LOG_STR]:
                return True
            else:
                return
        else:
            if self.map_to_plot[y+1][x] == self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y-1][x] == self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y][x-1] == self.mdp_items_id[LOG_STR]:
                return True
            elif self.map_to_plot[y][x+1] == self.mdp_items_id[LOG_STR]:
                return True
            else:
                return False

show_animation = False

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                # print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]#,
                  # [-1, -1, math.sqrt(2)],
                  # [-1, 1, math.sqrt(2)],
                  # [1, -1, math.sqrt(2)],
                  # [1, 1, math.sqrt(2)]]

        return motion
