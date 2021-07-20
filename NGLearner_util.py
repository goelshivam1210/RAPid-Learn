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
    'max_epsilon': 0.8,
    'min_epsilon': 0.2,
    'eps_lambda': None,
    'batch_size': 32,
    'max_timesteps': 50000,
    # 'session': None,
    'hidden_size': 32,
    'update_freq': 50, #1,
}

#EW: Not sure if want to use this, can remove
def check_permissible(self, action, info):
    action = self.env.all_actions[action]

    if self.last_action == 'PLACE_TREE_TAP':
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
            return False, 'placed_tap'

    if get_world_quant(info, 'minecraft:log') == 1 and get_inv_quant(info,'polycraft:sack_polyisoprene_pellets') < 1 and get_entity_quant(info, 'polycraft:sack_polyisoprene_pellets') < 1 and action == 'BREAK_BLOCK' and info['block_in_front']['name'] == 'minecraft:log' and 'break_last_tree' in self.impermissible_actions:
        return False, 'break_last_tree'
    elif action.startswith('CRAFT'):
        if action.split()[1] == 'minecraft:stick' and 'craft_unnecessary_stick' in self.impermissible_actions:
            if get_inv_quant(info, 'polycraft:tree_tap') < 1 and get_world_quant(info,'polycraft:tree_tap') < 1 and get_entity_quant(info, 'polycraft:tree_tap') < 1:
                return get_inv_quant(info, 'minecraft:stick') <= 1, 'craft_unnecessary_stick'
            else:
                return get_inv_quant(info, 'minecraft:stick') <= 4, 'craft_unnecessary_stick'

        elif action.split()[1] == 'polycraft:tree_tap' and 'craft_unnecessary_tap' in self.impermissible_actions:
            return get_inv_quant(info, 'polycraft:tree_tap') + get_world_quant(info,'polycraft:tree_tap') + get_entity_quant(info, 'polycraft:tree_tap') < 1, 'craft_unnecessary_tap'

        elif action.split()[1] != 'minecraft:planks' and action.split()[1] != 'polycraft:wooden_pogo_stick':
            if get_inv_quant(info, action.split()[1]) > 0 or get_entity_quant(info,action.split()[1]) > 0 or get_world_quant(info, action.split()[1]) > 0:
                return False, 'craft_new_recipe'

    return True, None


# informed random action based on failed goal and last reset state
#TODO: pass in agent to use class vars
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
                if info['block_in_front']['name'] == 'minecraft:air' or info['block_in_front'][
                    'name'] == 'minecraft:bedrock':
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
                    if self.failed_action == 'extractRubber':
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
                if get_inv_quant(info, 'minecraft:crafting_table') >= 1 and info['block_in_front'][
                    'name'] == 'minecraft:air':
                    action_pool.append(action)
                    action_values.append(1)
            # assert block in front is tree tap
            elif action_str == 'EXTRACT_RUBBER':
                if info['block_in_front']['name'] == 'polycraft:tree_tap':
                    # EXTRACTRUBBER IS SUPER EXPENSIVE, don't encourage
                    # only allow extractrubber if we're looking for a way to get rubber
                    # Either on that step in exploration or learning
                    if (self.mode == 'exploration' and self.failed_action == 'extractRubber') or \
                            (self.mode == 'learning' and self.failed_action == 'extractRubber'):
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
                obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)
            obs, rew, done, info = self.step_env(self.env.actions_id['TURN 90'], obs, info)
            for _ in range(np.random.randint(10)):
                obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)

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
                self.step_env(self.env.actions_id['SELECT_ITEM {}'.format(interesting_item)],store_transition=False)
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

                if (self.resetting_state and self.failed_action.split()[0] == 'moveTo' and self.failed_action.split()[1] == interesting_item):
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
            if (self.resetting_state and self.failed_action.split()[0] == 'moveTo' and self.failed_action.split()[1] == interesting_item):
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
        obs, rew, done, info = self.step_env(self.env.actions_id['MOVE w'], obs, info)
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
        return self.step_env(self.env.actions_id['TURN 90'], obs, info)
    elif num_rots == 2 or num_rots == -2:
        obs, rew, done, info = self.step_env(self.env.actions_id['TURN 90'], obs, info)
        return self.step_env(self.env.actions_id['TURN 90'], obs, info)
    elif num_rots == 3 or num_rots == -1:
        return self.step_env(self.env.actions_id['TURN -90'], obs, info)


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

    if self.failed_action.split()[0] == 'moveTo':
        # attempt to go as near as possible to goal of moveTo with higher probability
        reset_near_values[self.env.mdp_items_id[self.failed_action.split()[1]]] += 10
    elif self.failed_action.split()[0] == 'break':
        # attempt to go as near as possible to goal of break with higher probability
        reset_near_values[self.env.mdp_items_id[self.failed_action.split()[1]]] += 5

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
    elif predicate_str.split()[0] == 'near':
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
                    log_increase = get_inv_quant(info, 'minecraft:log') - get_inv_quant(init_info,'minecraft:log')
                    plank_increase = get_inv_quant(info, 'minecraft:planks') - get_inv_quant(init_info,'minecraft:planks')
                    stick_increase = get_inv_quant(info, 'minecraft:stick') - get_inv_quant(init_info,'minecraft:stick')
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
    # e.g. 'approach minecraft:log' - goal is block_in_front == obj
    if operator_str.split()[0] == 'approach':
        return get_create_success_func_from_predicate('near {}'.format(operator_str.split()[1]))
    # to the RL agent approach and pickup should have the same policy though
    if operator_str.split()[0] == 'pickup':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'decrease world {} 1'.format(operator_str.split()[1])])
    #e.g. break minecraft:log - real goal is inv increase obs, block in front air
    elif operator_str.split()[0] == 'break':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'near minecraft:air', 'decrease world {} 1'.format(operator_str.split()[1])])
    elif operator_str.split()[0] == 'place':
        return get_create_success_func_from_predicate_set(['decrease inventory {} 1'.format(operator_str.split()[1]), 'near {}'.format(operator_str.split()[1])])
    # Don't have notion of tapped_log currently
    elif operator_str == 'extractrubber':
        return get_create_success_func_from_predicate_set(['increase inventory polycraft:sack_polyisoprene_pellets 1'])
    # For all crafting actions goal is to increase
    # Do we need to include notion of decrease as well? - to prevent losing too much?
    #   Do we just make it so the decrease has to be <= the original? Just losing more is not okay
    elif operator_str == 'craftplank':
        return get_create_success_func_from_predicate_set(['increase inventory minecraft:planks 1'])
    elif operator_str == 'craftstick':
        return get_create_success_func_from_predicate_set(['increase inventory minecraft:stick 1'])
    elif operator_str == 'crafttreetap':
        return get_create_success_func_from_predicate_set(['increase inventory polycraft:tree_tap 1'])
    elif operator_str == 'craftpogostick':
        return get_create_success_func_from_predicate_set(['increase inventory polycraft:wooden_pogo_stick 1'])
    else:
        print(Fore.RED+"ERROR: cant create effect set from unknown operator {}".format(operator_str))


# Naive dynamics checker to emulate functionality from TRADE planner, which indicates if an outcome in terms of
#   resources is novel to the agent
class PolycraftDynamicsChecker:
    # Initialize with existing notion of dynamics
    def __init__(self, env, recovery_agent):
        self.breakable_effects = {'minecraft:log': {'inventory': {'minecraft:log': 1},
                                                    'world' :    {'minecraft:log': -1},
                                                    'entity':    {}
                                                    },
                                  'minecraft:crafting_table': {'inventory':  {'minecraft:crafting_table': 1},
                                                                'world':     {'minecraft:crafting_table': -1},
                                                                'entity':    {}
                                                               },
                                  'polycraft:tree_tap': {'inventory': {'polycraft:tree_tap': 1},
                                                         'world':     {'polycraft:tree_tap': -1},
                                                         'entity':    {}
                                                         }
                                  }
        self.extractable_effects = {'polycraft:tree_tap': {'inventory': {'polycraft:sack_polyisoprene_pellets': 1},
                                                           'world':     {},
                                                           'entity':    {}}
                                    }
        self.lost_effects = []

        self.new_effects = []
        self.new_effect_counts = []

        # Have notion of failed subgoal or operator to reason about what's a useful effect
        #   and to know when to replan/what to replan to (even if just movement)
        # Special case 'planning' due to percieved quantity of items in world too low?
        self.failed_goal = None
        self.failed_step = None
        self.updated_dynamics = False

        self.env = env

        self.pending_entity_pickup = False
        self.trade_plan = None

        #Ideally don't want this - change
        self.recovery_agent = recovery_agent

    # Two notions: what failed, and what the underlying issue is due to that failure
    # Possible failures:
    #   unplannable from the start
    #   moveTo (either log or ct)
    #   breakLog
    #   extractrubber
    #   craftItem (this should really never be the case right?)
    # Possible issues:
    #   Need to get 'item' more efficiently
    #   Need to get access to item in order to moveTo
    #   Need to relearn policy for moveTo (or any operator) due to dynamics change
    def update_from_failure(self, failed_step, info = {}):
        # Is there any instance where it's unplannable, but not due to lack of resources?
        if failed_step == 'planning':
            # Possible cases:
                # not enough resources in env
                    # less trees than originally
                    # no crafting table and no recipe or too expensive
                    # some recipe(s) became more expensive
                # no recipe for pogo_stick or some required item
                    # need to find other way to obtain
            self.failed_step = 'planning'
            # Can we reason about and pass in whether we need e.g. more of a specific item?
            # Will always be the case, but don't know how to overcome it, so not really helpful
            self.failed_goal = info['desired_goal']
            #haven't found anything new because haven't done anything, failed on planning
            self.updated_dynamics = True
        elif failed_step.startswith('break'):
            if 'new_effects' in info:
                #add notion of lost effect in case we happen to recover it
                self.add_lost_effect(info['block_in_front']['name'], 'breakable')
                self.breakable_effects[failed_step.split()[1]] = info['new_effects']
                self.updated_dynamics = True
            else:
                print('Warn, need to provide new resource effect for modified break result, assuming nothing happened when action failed')
                self.add_lost_effect(failed_step.split()[1], 'breakable')
                self.breakable_effects[failed_step.split()[1]] = {'inventory': {}, 'world': {}, 'entity': {}}
                self.updated_dynamics = True
            self.failed_step = failed_step
            self.failed_goal = 'inventory minecraft:log'
        elif failed_step == 'extractRubber':
            if 'new_effects' in info:
                self.add_lost_effect(info['block_in_front']['name'], 'extractable')
                self.extractable_effects[failed_step.split()[1]] = info['new_effects']
                self.updated_dynamics = True
            else:
                print(Fore.YELLOW+'Warn, need to provide new resource effect for modified extract result, assuming nothing happened when action failed')
                self.add_lost_effect('polycraft:tree_tap', 'extractable')
                self.extractable_effects['polycraft:tree_tap'] = {'inventory': {}, 'world': {}, 'entity': {}}
                self.updated_dynamics = True
            self.failed_step = failed_step
            self.failed_goal = 'inventory polycraft:sack_polyisoprene_pellets'
        elif failed_step.startswith('moveTo'):
            if 'cause' in info:
                self.failed_step = failed_step
                # planning vs execution
                self.failed_goal = info['cause']
                self.updated_dynamics = True
            else:
                print(Fore.RED+"Error: must provide cause for moveTo failure - failure in planning vs execution requires different responses")
                quit()
        elif failed_step.startswith('craft') or failed_step.startswith('pickUp'):
            self.failed_step = failed_step
            self.failed_goal = 'inventory {}'.format(failed_step.split()[1])
            self.updated_dynamics = True
        else:
            print(Fore.RED+"Error: unknown failure step {}, exiting".format(failed_step))
            quit()
        print('in update from failure')
        print(self.failed_goal)
        print(self.failed_step)

    # Given subsequent observations, check if any relevant novelties have occured
    def check_for_novelty(self, info, a, info_2, exploring=False):
        if self.failed_step is None:
            print(Fore.RED+'ERROR: cant check for novelty before supplying dynamics object what caused execution failure')
        if get_world_quant(info, 'minecraft:bedrock') != get_world_quant(info_2, 'minecraft:bedrock'):
            print(Fore.LIGHTYELLOW_EX+'New bedrock blocks spawned, need to account for this')

        # FIRST CHECK IF NOVEL OBJECT APPEARED, IF SO REVERT TO EXPLORATION
        if len(self.env.items_id)+1 > self.env.num_types:
            print(Fore.LIGHTRED_EX+"WARN: Novel type detected mid-execution and RL agent has already been instantiated")
            print(Fore.CYAN+"In this case we want to revert to explicit exploration, for now updating state space, resetting all RL details, and continuing")
            return 'novel_item', None

        expected_world_diffs = np.zeros(len(self.env.mdp_items_id))
        expected_inv_diffs = np.zeros(len(self.env.mdp_items_id))
        actual_world_diffs = np.zeros(len(self.env.mdp_items_id))
        actual_inv_diffs = np.zeros(len(self.env.mdp_items_id))
        expected_entity_diffs = np.zeros(len(self.env.mdp_items_id))
        actual_entity_diffs = np.zeros(len(self.env.mdp_items_id))

        # EXPECTED OUTCOMES
        # First compute expected changes to the world and inventory based on state and action
        if a.startswith('CRAFT'):
            item_to_craft = a.split()[1]
            # TODO: handle multiple recipes
            recipe = self.env.recipes[item_to_craft][0]
            # We expect to lose ingredients and gain output
            if len(recipe['ingredients']) <= 4 or (len(recipe['ingredients']) > 4 and info['block_in_front'] != 'minecraft:crafting_table'):
                # Subtract ingredients
                for slot in recipe['ingredients']:
                    if recipe['ingredients'][slot] is not None:
                        expected_inv_diffs[self.env.mdp_items_id[recipe['ingredients'][slot]['Item']]] -= recipe['ingredients'][slot]['stackSize']
                #Add output item
                expected_inv_diffs[self.env.mdp_items_id[recipe['output_item']['Item']]] += recipe['output_item']['stackSize']
            #If we don't actually have the ingredients, we expect nothing to happen
            for item_id in range(len(expected_inv_diffs)):
                if expected_inv_diffs[item_id] < 0 and get_inv_quant(info, self.env.all_items[item_id]) < -expected_inv_diffs[item_id]:
                    expected_inv_diffs = np.zeros(len(self.env.mdp_items_id))
                    break
        # TODO: account for nondeterministic action outcomes, or conditional action outcomes explicitly
        elif a == 'BREAK_BLOCK':
            # update breakable block set if found to be unbreakable
            # include novel objects once we know is breakable but not helpful
            for block in self.breakable_effects: #['minecraft:log', 'minecraft:crafting_table', 'polycraft:tree_tap']:
                if info['block_in_front']['name'] == block:
                    for item in self.breakable_effects[block]['inventory']:
                        expected_inv_diffs[self.env.mdp_items_id[item]] += self.breakable_effects[block]['inventory'][item]
                    for item in self.breakable_effects[block]['world']:
                        expected_world_diffs[self.env.mdp_items_id[item]] += self.breakable_effects[block]['world'][item]
                    for item in self.breakable_effects[block]['entity']:
                        expected_entity_diffs[self.env.mdp_items_id[item]] += self.breakable_effects[block]['entity'][item]

        # TODO: need to encode if object that is being tapped can be something other than a tree
        #       - should be found in explixit exploration though
        elif a == 'EXTRACT_RUBBER':
            for block in self.extractable_effects:
                if info['block_in_front']['name'] == block and self.check_extract_rubber(info):
                    for item in self.extractable_effects['polycraft:tree_tap']['inventory']:
                        expected_inv_diffs[self.env.mdp_items_id[item]] += self.extractable_effects['polycraft:tree_tap']['inventory'][item]
                    for item in self.extractable_effects['polycraft:tree_tap']['world']:
                        expected_world_diffs[self.env.mdp_items_id[item]] += self.extractable_effects['polycraft:tree_tap']['world'][item]
                    for item in self.extractable_effects['polycraft:tree_tap']['entity']:
                        expected_entity_diffs[self.env.mdp_items_id[item]] += self.extractable_effects['polycraft:tree_tap']['entity'][item]

        elif a == 'PLACE_CRAFTING_TABLE':
            if info['block_in_front']['name'] == 'minecraft:air' and get_inv_quant(info, 'minecraft:crafting_table') > 0:
                expected_inv_diffs[self.env.mdp_items_id['minecraft:crafting_table']] -= 1
                expected_world_diffs[self.env.mdp_items_id['minecraft:crafting_table']] += 1

        elif a == 'PLACE_TREE_TAP':
            if info['block_in_front']['name'] == 'minecraft:air' and get_inv_quant(info, 'polycraft:tree_tap') > 0:
                expected_inv_diffs[self.env.mdp_items_id['polycraft:tree_tap']] -= 1
                expected_world_diffs[self.env.mdp_items_id['polycraft:tree_tap']] += 1

        #select should have no effect
        #movement player pos differences should be caught separately

        #Keep track whether we 1. need to update behavior based on detected change
        #                  and 2. want to update notion of an action effect to no longer consider it novel
        inv_as_expected, world_as_expected, entity_as_expected = True, True, True
        discrepency_world_diffs = np.zeros(len(self.env.mdp_items_id))
        discrepency_inv_diffs = np.zeros(len(self.env.mdp_items_id))
        discrepency_entity_diffs = np.zeros(len(self.env.mdp_items_id))
        negative_inv, positive_inv, negative_world, positive_world, negative_entity, positive_entity = False, False, False, False, False, False

        # ACTUAL OUTCOMES
        ## Changes to world (block or entity)
        ## Changes to inventory from step
        ## TODO: check if locations of items in world align?
        for item in self.env.mdp_items_id:
            if item == 'minecraft:air' or item == 'minecraft:bedrock':
                continue
            #### calculate actual inv diffs ####
            actual_inv_diff = get_inv_quant(info_2, item) - get_inv_quant(info, item)
            actual_inv_diffs[self.env.mdp_items_id[item]] += actual_inv_diff

            #### calculate actual world diffs ####
            actual_world_diff = get_world_quant(info_2, item) - get_world_quant(info, item)
            actual_world_diffs[self.env.mdp_items_id[item]] += actual_world_diff

            #### calculate actual entity diffs ####
            actual_entity_diff = get_entity_quant(info_2, item) - get_entity_quant(info, item)
            actual_entity_diffs[self.env.mdp_items_id[item]] += actual_entity_diff

            #calculate discrepency between expected and actual difs for item anywhere in domain
            discrepency_inv_diff = actual_inv_diffs[self.env.mdp_items_id[item]] - expected_inv_diffs[self.env.mdp_items_id[item]]
            discrepency_inv_diffs[self.env.mdp_items_id[item]] = discrepency_inv_diff

            discrepency_world_diff = actual_world_diffs[self.env.mdp_items_id[item]] - expected_world_diffs[self.env.mdp_items_id[item]]
            discrepency_world_diffs[self.env.mdp_items_id[item]] = discrepency_world_diff

            discrepency_entity_diff = actual_entity_diffs[self.env.mdp_items_id[item]] - expected_entity_diffs[self.env.mdp_items_id[item]]
            discrepency_entity_diffs[self.env.mdp_items_id[item]] = discrepency_entity_diff

            #### Calculate inv diffs ####
            if discrepency_inv_diff < 0:
                inv_as_expected = False
                negative_inv = True
                print(Fore.CYAN+'Got {} less of material {} in inventory than expecting - this shouldnt be the case, check if blocking novelty info has been incorporated'.format(-discrepency_inv_diff, item))
            elif discrepency_inv_diff > 0:
                inv_as_expected = False
                positive_inv = True
                print(Fore.CYAN+'Got {} more of material {} in inventory than expecting '.format(discrepency_inv_diff, item))

            #### Calculate block diffs ####
            if discrepency_world_diff < 0:
                world_as_expected = False
                negative_world = True
                print(Fore.CYAN+'{} less instances of block {} in the world than expecting'.format(-discrepency_world_diff, item))
            elif discrepency_world_diff > 0:
                world_as_expected = False
                positive_world = True
                print(Fore.CYAN+'{} more instances of block {} in the world than expecting'.format(discrepency_world_diff, item))

            #### Calculate entity diffs ####
            if discrepency_entity_diff < 0:
                entity_as_expected = False
                negative_entity = True
                print(Fore.CYAN+'{} less instances of entity {} than expecting - did we pick it up?'.format(-discrepency_entity_diff, item))
            elif discrepency_entity_diff > 0:
                entity_as_expected = False
                positive_entity = True
                print(Fore.CYAN+'{} more instances of entity {} spawned than expecting'.format(discrepency_entity_diff, item))

        # TODO: is this ever used anymore?
        # Upon first init call, we just want to update notion of our dynamics
        # Ideally this will be passed in from the planner in a separate call
        if not self.updated_dynamics:
            print(Fore.CYAN+'Updating dynamics based on transition supplied - this should only be called with failed transition information')
            self.update_knowledge_base(info, a, actual_inv_diffs, actual_world_diffs, actual_entity_diffs)
            self.updated_dynamics = True
            return

        #return info to novelty agent to react accordingly
        actual_diffs = {'inventory': actual_inv_diffs,
                        'world': actual_world_diffs,
                        'entity': actual_entity_diffs}

        #Special case - if exploring from failed moveTo, then we want to check if block_in_front is goal id
        if self.recovery_agent.mode == 'exploration' and self.recovery_agent.failed_action.split()[0] == 'moveTo':
            if info_2['block_in_front']['name'] == self.recovery_agent.failed_action.split()[1]:
                return 'recovered', actual_diffs

        #Done cycle through item differences, reason about results
        # Everything happened as expected, do nothing special
        if inv_as_expected and world_as_expected and entity_as_expected:
            return 'irrelevant', actual_diffs
        #TODO: is this recovered case irrelevant with new res_cp setup? If we've recovered the current
        #   operator then we will find that in check_success, otherwise we don't want to indicate
        #   success if its for a different step and doesn't progress us res_cp.
        else:
            #Will it ever be the case where we have lost functionality, and we cannot plan from the start?
            # First check if we recovered lost functionality
            if len(self.lost_effects) > 0:
                for lost_effect in self.lost_effects:
                    if np.all(lost_effect['effect']['inventory'] == actual_inv_diffs) and np.all(
                            lost_effect['effect']['world'] == actual_world_diffs) and np.all(
                            lost_effect['effect']['entity'] == actual_entity_diffs):
                        print(Fore.GREEN+'Recovered lost effect exactly')
                        if self.failed_step == 'planning':
                            print(Fore.YELLOW+'Recovered lost effect but failed on planning step so we cannot use original plan')
                            print(Fore.YELLOW+"Warning: This shouldn't ever be the case?")
                            break
                        # If we recover lost functionality we need to make sure it's on the same subplan step
                        # We want recovered when on that operator or the moveTo before that operator, otherwise don't indicate recovered
                        # elif lost_effect['type'] == 'breakable':
                        #     if 'break minecraft:log' not in self.recovery_agent.planner.resource_checkpoint_subplans[self.recovery_agent.planner.current_res_cp]:
                        #         break
                        # elif lost_effect['type'] == 'extractable':
                        #     if 'extractRubber' not in self.recovery_agent.planner.resource_checkpoint_subplans[self.recovery_agent.planner.current_res_cp]:
                        #         break
                        print(Fore.GREEN+'Should be able to recreate lost operator and fit into original plan to solve problem')
                        return 'recovered', actual_diffs


            print(Fore.CYAN+'Something novel happened performing action {}, reasoning about novel effects'.format(a))
            # TODO: expand reasoning to cover more cases, for now only want to prioritize basic ones
            #   Or ideally offload this to planner so we don't have to do any of this
            outcome = 'irrelevant'
            attempt_replan = False
            #Only inventory diffs changed
            if not inv_as_expected and world_as_expected and entity_as_expected:
                if positive_inv and negative_inv:
                    print(Fore.CYAN+'Gained and lost more of some inv items than expected - reason about whether net effect was good or not, for now assuming useful change')
                    outcome = 'beneficial'
                    attempt_replan = True
                elif positive_inv:
                    if a == 'MOVE w' or a.startswith('TURN'):
                        print(Fore.CYAN+'Unexpectedly gained item from movement action - likely entity wasnt picked up in sense all, indicating change as irrelevant but attempting replan just in case')
                        outcome == 'irrelevant'
                        attempt_replan = True
                    else:
                        print(Fore.CYAN+'Only gained more inv items than expected - indicating change as beneficial')
                        outcome = 'beneficial'
                        attempt_replan = True
                elif negative_inv:
                    if a == 'BREAK_BLOCK':
                        print(Fore.CYAN + 'Didnt get as many items from break action as expected - possible entity has not been caught in sense all, indicating change as irrelevant')
                        outcome = 'irrelevant'
                    else:
                        print(Fore.CYAN+'Only lost more inv items than expected - indicating change as detrimental')
                        outcome = 'detrimental'

            #Only blocks in world diffs changed
            elif inv_as_expected and not world_as_expected and entity_as_expected:
                if positive_world and negative_world:
                    print(Fore.CYAN+'Gained and lost more blocks in world than expected - indicating novel effect as beneficial')
                    outcome = 'beneficial'
                    attempt_replan = True
                elif positive_world:
                    print(Fore.CYAN+'Already known block unexpectedly spawned in world - indicating novel effect as beneficial')
                    outcome = 'beneficial'
                    attempt_replan = True
                elif negative_world:
                    print(Fore.CYAN+'Block was unexpectedly removed from world')
                    #Think it's better to just assume novelties are beneficial, e.g. if in fence case we press a button to make fences disappear,
                    #   we don't want the fences disappearing to be considered 'detrimental'
                    outcome = 'beneficial'
                    attempt_replan = True

            #Only entities in world diffs changed
            elif inv_as_expected and world_as_expected and not entity_as_expected:
                if positive_entity and negative_entity:
                    print(Fore.CYAN+'Entities of unexpected types have spawned and/or left the world - Should compare value of different entities')
                    outcome = 'beneficial'
                    attempt_replan = True
                elif positive_entity:
                    print(Fore.CYAN+'Entity unexpectedly spawned in world - assuming useful entity')
                    outcome = 'beneficial'
                    attempt_replan = True
                elif negative_entity:
                    print(Fore.CYAN+'Entity unexpectedly disappeared from world without gaining into inventory')
                    outcome = 'detrimental'

            #other common cases: breaking block for the first time -> + world, - world, + inv
            elif positive_inv and negative_world and not (negative_inv or positive_world) and entity_as_expected:
                if self.recovery_agent.mode == 'exploration':
                    print(Fore.CYAN+'Likely found way to break a block for the first time, setting as beneficial')
                outcome = 'beneficial'
                attempt_replan = True

            #Picking up entity
            elif positive_inv and negative_entity and not (negative_inv or positive_entity) and world_as_expected:
                #If we're waiting on results of moveTo log and pick up log entity -> return 'recovered'
                if self.pending_entity_pickup:
                    if actual_inv_diffs[self.env.items_id['minecraft:log']] > 0:
                        print(Fore.GREEN + "Picked up entity missed from breaking log, setting return as 'recovered'")
                        outcome = 'recovered'
                        self.pending_entity_pickup = False
                else:
                    if self.recovery_agent.mode == 'exploration':
                        print(Fore.CYAN+'Likely picked up an entity, should already know how to do this and will be prioritized in reset anyway, setting as irrelevant but checking replannable in case its new')
                    outcome = 'irrelevant'
                    attempt_replan = True

            #Failed to break block
            elif negative_inv and positive_world and not (positive_inv or negative_world) and entity_as_expected:
                if self.recovery_agent.mode == 'exploration':
                    print(Fore.CYAN + 'Likely failed to break a block, if not a blocking novelty, setting as irrelevant\n')
                outcome = 'irrelevant'

            #Broke block but entity not picked up
            elif negative_inv and positive_entity and not (positive_inv or negative_entity) and world_as_expected and a == 'BREAK_BLOCK':
                if self.recovery_agent.mode == 'exploration':
                    print(Fore.CYAN + 'Likely broke a block without picking up entity that dropped, setting as irrelevant\n')
                outcome = 'irrelevant'

            #Broke block without picking up entity, but wasn't expecting to break block
            elif negative_world and positive_entity and not (positive_world or negative_entity) and inv_as_expected and a == 'BREAK_BLOCK':
                if self.recovery_agent.mode == 'exploration' and not self.failed_step.startswith('break'):
                    print(Fore.CYAN + 'Likely broke a block without picking up entity that dropped, but wasnt expecting to recieve item, setting as beneficial')

                if exploring:
                    if len(self.lost_effects) > 0:
                        for lost_effect in self.lost_effects:
                            #In this case, if the broken block is the same, and the entity is what we were expecting to
                            # pick up, consider it recovering the old operator
                            if np.all(lost_effect['effect']['inventory'] == actual_entity_diffs) and np.all(
                                    lost_effect['effect']['world'] == actual_world_diffs):
                                print(Fore.GREEN + 'Believe we found a way to break a tree after being unable to do so')
                                print(Fore.GREEN + 'Sending signal to the agent to moveTo the log entity, if successful, recreating lost break log operator')
                                self.pending_entity_pickup = True
                                return 'recovered_pending', actual_diffs
                outcome = 'beneficial'

            #Check against lost operator effects - if recovered exactly, then can likely just use the original operator as a goal
            else:
                print(Fore.CYAN+'Unplanned action effect outcome resulting in pos_inv={}, neg_inv={}, pos_world={}, neg_world={},'
                      'pos_entity={}, neg_entity={}. Assuming positive effect'.format(positive_inv, negative_inv,
                                                                                      positive_world, negative_world,
                                                                                      positive_entity, negative_entity))
                outcome = 'beneficial'
                attempt_replan = True

        #If we're learning a specific operator, it means we've found a solution to it,
        #  so to keep things simple don't check to replan if learning something we know can work
        if self.recovery_agent.mode == 'exploration':
            if attempt_replan:
                #First check in new effects if this is something we've seen before
                #If replanning is expensive, we don't want to try to replan everytime we discover the same auxiliary
                #   (or nuisance) effect if it doesn't eventually lead to a plannable state
                new_effect_ind = None
                for i, new_effect in enumerate(self.new_effects):
                    #Have seen this effect before
                    if a == new_effect['type'] and  \
                      np.all(new_effect['effect']['inventory'] == actual_inv_diffs) and \
                      np.all(new_effect['effect']['world'] == actual_world_diffs) and \
                      np.all(new_effect['effect']['entity'] == actual_entity_diffs):
                        #How much do we discourage replanning and when?
                        #Initial attempt - try first 5 times, then based off 'plannable ratio'
                        #Also, if we find the same effect too often without finding a plannable state, change outcome
                        #   from beneficial to irrelevant
                        if new_effect['count'] >= 10:
                            print(Fore.CYAN+'Found same novel effect too often without finding plannable state, setting as irrelevant')
                            # TODO: Need to be smarter about this, probably relate to success rate. In hard cases
                            #   we don't want to through any notion of progress away
                            # outcome = 'irrelevant'
                            return outcome, actual_diffs

                        if new_effect['count'] >= 5:
                            unplannable_ratio = (new_effect['count'] - new_effect['plannable']) / new_effect['count']
                            #always give some chance to try to replan
                            if np.random.random() < min(0.9, unplannable_ratio):
                                print(Fore.CYAN+'Found same novel effect often without reaching plannable state -> not replanning')
                                return outcome, actual_diffs
                        #Move this higher?
                        #Else, try replan and increment count or plannable
                        new_effect_ind = i
                        self.new_effects[new_effect_ind]['count'] += 1
                        break

                if self.recovery_agent.can_trade_plan:
                    print(Fore.LIGHTGREEN_EX+'Found interesting enough novelty, reaching out to planner to attempt to replan')
                    plan = self.recovery_agent.trade_plan_to_goal()
                    if plan is not None:
                        plannable = True
                        self.trade_plan = plan
                    else:
                        plannable = False
                        self.trade_plan = None
                else:
                    plannable = False

                # #TODO test with actual connection to planner
                # self.planner_sock.send(str.encode('tryToPlan [inventory polycraft:wooden_pogo_stick 1]'))
                # response = recv_socket_data(self.planner_sock)
                # print(response)

                if plannable:
                    if new_effect_ind is not None:
                        self.new_effects[new_effect_ind]['plannable'] += 1
                    print(Fore.LIGHTGREEN_EX+'Found plannable state, overriding previous outcome and returning plannable')
                    return 'plannable', actual_diffs
                #Otherwise, add to new effects. If novelty leads to plannable state, keep notion of outcome as novel
                # so we check to replan each time it occurs.
                #For each time it occurs and doesn't lead to a novel state, we increment the count and eventually discourage
                # replanning
                else:
                    if new_effect_ind is None:
                        self.new_effects.append({'type': a,
                                                  'item': info['block_in_front']['name'],
                                                  'effect': actual_diffs,
                                                  'count': 0,
                                                  'plannable': 0})
        return outcome, actual_diffs


    #Taken from polycraft_interface - better way to do this?
    def check_extract_rubber(self, info):
            player_facing = self.env.player['facing']
            player_x = self.env.player['pos'][0]
            player_y = self.env.player['pos'][2]
            flag = False
            for i in range(get_world_quant(info, 'minecraft:log')):
                loc = list(map(int, str.split(info['items_locs']['minecraft:log'][i], ',')))
                # if facing any direction there are three cases:
                # A: two blocks away B: 2 diagonal blocks away cases
                if player_facing == 'NORTH':
                    if player_x == loc[0] and \
                            player_y == loc[2] + 2:
                        flag = True
                        break
                    elif player_x == loc[0] - 1 \
                            and player_y == loc[2] + 1:
                        flag = True
                        break
                    elif player_x == loc[0] + 1 and \
                            player_y == loc[2] + 1:
                        flag = True
                        break
                elif player_facing == 'SOUTH':
                    if player_x == loc[0] and \
                            player_y == loc[2] - 2:
                        flag = True
                        break
                    elif player_x == loc[0] - 1 and \
                            player_y == loc[2] - 1:
                        flag = True
                        break
                    elif player_x == loc[0] + 1 and \
                            player_y == loc[2] - 1:
                        flag = True
                        break
                elif player_facing == 'EAST':
                    if player_x == loc[0] - 2 and \
                            player_y == loc[2]:
                        flag = True
                        break
                    elif player_x == loc[0] - 1 and \
                            player_y == loc[2] - 1:
                        flag = True
                        break
                    elif player_x == loc[0] - 1 and \
                            player_y == loc[2] + 1:
                        flag = True
                        break
                elif player_facing == 'WEST':
                    if player_x == loc[0] + 2 and \
                            player_y == loc[2]:
                        flag = True
                        break
                    elif player_x == loc[0] + 1 and \
                            player_y == loc[2] - 1:
                        flag = True
                        break
                    elif player_x == loc[0] + 1 and \
                            player_y == loc[2] + 1:
                        flag = True
                        break
                else:
                    flag = False
            return flag

    #Is this still used?
    #TODO: differentiate on more than block_in_front or have multiple possible effects for a given action
    def update_knowledge_base(self, info, a, actual_inv_diffs, actual_world_diffs, actual_entity_diffs):
        #if updated effect is that nothing happens, simply remove entry
        if not (np.any(actual_inv_diffs) or np.any(actual_world_diffs) or np.any(actual_entity_diffs)):
            if a == 'BREAK_BLOCK':
                if info['block_in_front']['name'] in self.breakable_effects:
                    self.add_lost_effect(info['block_in_front']['name'], 'breakable')
                    del self.breakable_effects[info['block_in_front']['name']]
            elif a == 'EXTRACT_RUBBER':
                if info['block_in_front']['name'] in self.extractable_effects:
                    self.add_lost_effect(info['block_in_front']['name'], 'extractable')
                    del self.extractable_effects_effects[info['block_in_front']['name']]
            else:
                print(Fore.RED+'Failure occured on action {}, this should never be the case (has a recipe failed without us knowing?)'.format(a))
            return

        #TODO: need ability of multiple effects for same action. If old effect isn't blocked, we don't want to consider
        # it novel. Otherwise create new and update effect dict for action
        effect_dict = {'inventory': {},
                       'world': {},
                       'entity': {}}
        for i in range(len(actual_inv_diffs)):
            if actual_inv_diffs[i] != 0:
                effect_dict['inventory'][self.env.all_items[i]] = actual_inv_diffs[i]
            if actual_world_diffs[i] != 0:
                effect_dict['world'][self.env.all_items[i]] = actual_world_diffs[i]
            if actual_entity_diffs[i] != 0:
                effect_dict['entity'][self.env.all_items[i]] = actual_entity_diffs[i]

        if a == 'BREAK_BLOCK':
            self.breakable_effects[info['block_in_front']['name']] = effect_dict
        elif a == 'EXTRACT_RUBBER':
            self.extractable_effects_effects[info['block_in_front']['name']] = effect_dict
        else:
            print(Fore.RED+'Failure occured on action {}, this should never be the case (has a recipe failed without us knowing?)'.format(a))

    def add_lost_effect(self, item_in_front, action_type):
        lost_effect = {'inventory': np.zeros(len(self.env.mdp_items_id)),
                       'world': np.zeros(len(self.env.mdp_items_id)),
                       'entity': np.zeros(len(self.env.mdp_items_id))}
        if action_type == 'breakable':
            old_effect = self.breakable_effects[item_in_front]
        elif action_type == 'extractable':
            old_effect = self.extractable_effects[item_in_front]

        # Do we care if it's the same method or not, or only if its the same outcome?
        for item in old_effect['inventory']:
            lost_effect['inventory'][self.env.mdp_items_id[item]] += old_effect['inventory'][item]
        for item in old_effect['world']:
            lost_effect['world'][self.env.mdp_items_id[item]] += old_effect['world'][item]
        for item in old_effect['entity']:
            lost_effect['entity'][self.env.mdp_items_id[item]] += old_effect['entity'][item]

        self.lost_effects.append({'type': action_type,
                                  'item': item_in_front,
                                  'effect': lost_effect})


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
    Action: {'Forward': 0, 'Left': 1, 'Right': 2, 'Break': 3, 'Place_tree_tap': 4, 'Extract_rubber': 5,
            Craft action for each recipe, Select action for each item except unbreakable items}
    """

    def __init__(self, env, render_bool=False):
        self.env = env
        self.game_over = False

        ## Don't want to do socket connection if unnecessary
        # # Given attributes:
        # self.host = host
        # self.port = port
        # self.reset_command = reset_command
        self.render_bool = render_bool
        # self.save_json = save_json

        # Processed SENSE_RECIPES attributes:
        self.recipes = {}
        #TODO: compute this along with changes to recipe rep
        self.crafting_table_needed_dict = {'minecraft:planks': [False],
                                           'minecraft:stick': [False],
                                           'polycraft:tree_tap': [True],
                                           'polycraft:wooden_pogo_stick': [True]}
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
        self.binary_map = []  # contains 0 for 'minecraft:air', otherwise 1

        # Constants specific to PAL
        self.move_commands = ['SMOOTH_MOVE', 'MOVE', 'SMOOTH_TURN', 'TURN', 'SMOOTH_TILT']
        # self.move_commands.extend(['MOVE_NORTH', 'MOVE_SOUTH', 'MOVE_EAST', 'MOVE_WEST', 'LOOK_NORTH', 'LOOK_SOUTH',
        #                            'LOOK_EAST', 'LOOK_WEST'])  # RL agent nav. actions
        self.run_SENSE_ALL_and_update()
        self.run_SENSE_RECIPES_and_update()


    def send_command(self, command):
        if command.startswith == 'RESET' or command == 'RESET':
            self.env.reset()
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
            if command == 'MOVE w':
                _, _r, _d, info = self.env.step(self.env.actions_id['Forward'])
            elif command == 'TURN -90':
                _, _r, _d, info = self.env.step(self.env.actions_id['Left'])
            elif command == 'TURN 90':
                _, _r, _d, info = self.env.step(self.env.actions_id['Right'])
            elif command == 'BREAK_BLOCK':
                _, _r, _d, info = self.env.step(self.env.actions_id['Break'])
            elif command == 'PLACE_TREE_TAP':
                _, _r, _d, info = self.env.step(self.env.actions_id['Place_tree_tap'])
            elif command == 'PLACE_CRAFTING_TABLE':
                _, _r, _d, info = self.env.step(self.env.actions_id['Place_crafting_table'])
            elif command == 'EXTRACT_RUBBER':
                _, _r, _d, info = self.env.step(self.env.actions_id['Extract_rubber'])
            elif command.split()[0] == 'SELECT_ITEM':
                _, _r, _d, info = self.env.step(self.env.actions_id['Select_{}'.format(command.split()[1])])
            elif command.split()[0] == 'CRAFT':
                _, _r, _d, info = self.env.step(self.env.actions_id['Craft_{}'.format(command.split()[1])])
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

    def run_a_command_and_sense_all(self, a_command, sleep_time=0):
        """
        Run a command and sense all
        Extending to handle any action type
        """

        a_command_output = self.send_command(a_command)
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

        # These will all updated in step anyway right? So we don't actually have to do anything
        # Copy relevant variables from underlying env
        # Do we have to copy things here
        self.selected_item = self.env.selected_item
        try:
            self.selected_item_id = self.items_id[self.selected_item]
        except:
            # Equate holding air to holding nothing
            self.selected_item_id = 0
        # self.selected_item_id = self.items_id[self.selected_item]
        self.block_in_front = {'name':self.env.block_in_front_str}
        self.inventory_quantity_dict = {}
        unfiltered_inventory_quantity_dict = self.env.inventory_items_quantity.copy()
        for item in unfiltered_inventory_quantity_dict:
            if unfiltered_inventory_quantity_dict[item] > 0:
                self.inventory_quantity_dict[item] = unfiltered_inventory_quantity_dict[item]
        # self.inventory_quantity_dict = self.env.inventory_items_quantity.copy()
        # self.player = self.env.player
        # Don't think z should ever change
        # self.player = {'pos': [self.env.agent_location[0], 0, self.env.agent_location[1]],
        self.player = {'pos': [self.env.agent_location[1], 0, self.env.agent_location[0]],
                       'facing': self.env.agent_facing_str,
                       'yaw': 0,
                       'pitch': 0}
        self.entities = self.env.entities.copy()
        # self.map_to_plot = self.env.map.copy()
        self.items_location = {}
        # construct items_location dict from map
        # for r in range(self.map_to_plot.shape[0]):
        #     for c in range(self.map_to_plot.shape[1]):
        env_map = self.env.map.copy()
        self.map_to_plot = np.zeros((env_map.shape[0], env_map.shape[1]))  # Y (row) is before X (column) in matrix
        for r in range(env_map.shape[0]):
            for c in range(env_map.shape[1]):
                item_name = self.env.id_items[env_map[r][c]]
                self.items_location.setdefault(item_name, [])
                # self.items_location[item_name].append('{},0,{}'.format(r,c))
                # x,z,y to match with polycraft
                self.items_location[item_name].append('{},0,{}'.format(c,r))
                self.map_to_plot[r][c] = self.items_id[item_name]

        # self.binary_map = np.where(self.map_to_plot == self.items_id['minecraft:air'], 0, 1)
        # self.binary_map = np.where(self.map_to_plot == self.env.items_id['minecraft:air'], 0, 1)
        self.binary_map = np.where(self.map_to_plot == self.items_id['minecraft:air'], 0, 1)

        sense_all_command_result = {'command': 'SENSE_ALL',
                                    'argument': parameter,
                                    'result': 'SUCCESS',
                                    'message': '',
                                    'stepCost': 114.0}
        # self.entities_location
        # self.binary_map

        return sense_all_command_result

    def run_SENSE_RECIPES_and_update(self):
        self.recipes = self.env.polycraft_recipes.copy()
        # for item in self.recipes:
        #     self.recipes[item] = [self.recipes[item]]
        # env_recipes = self.env.env_recipes
        env_recipes = self.env.recipes.copy()

        # Finding self.ingredients_quantity_dict
        for item_to_craft in env_recipes:
            # print(item_to_craft)
            self.ingredients_quantity_dict.setdefault(item_to_craft, [])

            # for a_recipe in env_recipes[item_to_craft]:
            a_recipe = env_recipes[item_to_craft]

            # print(a_recipe)
            ingredients_quantity_dict_for_item_to_craft = {}

            for item in a_recipe['input']:
                # print(item)
                # print(a_recipe['input'][item])
                ingredients_quantity_dict_for_item_to_craft.setdefault(item, 0)
                ingredients_quantity_dict_for_item_to_craft[item] = a_recipe['input'][item]
            # print(ingredients_quantity_dict_for_item_to_craft)

            self.ingredients_quantity_dict[item_to_craft].append(ingredients_quantity_dict_for_item_to_craft)

    def run_SENSE_INVENTORY_and_update(self, inventory_output_inventory=None):
        # self.inventory_quantity_dict = self.env.inventory_items_quantity.copy()
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


# Env class to interface between polycraft socket connection and RL agents
class GridworldMDP(NovelgridworldInterface):

    def __init__(self, env, use_trade, render=False, agent_view_size=4, restrict_space=False):
        NovelgridworldInterface.__init__(self, env, render_bool=render)

        # local view size
        self.agent_view_size = agent_view_size
        # self.task = task

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

        if render:
            self.env.render()
            self.env.render()

    def set_items_id(self, items):
        # items_id = {}
        self.all_items = [None for i in range(len(items))]
        if 'minecraft:air' in items:
            #This should always be 0
            self.mdp_items_id['minecraft:air'] = 0
            # items_id['minecraft:air'] = 0
            # self.all_items.append('minecraft:air')
            self.all_items[0] = 'minecraft:air'
        # for item in sorted(items):
        for item in items:
            if item != 'minecraft:air':
                if 'minecraft:air' in items:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id))
                    # items_id[item] = len(items_id)
                else:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id)+1)
                    # items_id[item] = len(items_id) + 1
                # self.all_items.append(item)
                self.all_items[self.mdp_items_id[item]] = item

        # return items_id

    def generate_obs_action_spaces(self, new_items=[]):
        # #MDP items_id must be same as env items_id, but we want to include items that the env has not seen before
        # #Since items_id is updated as we go on, copying items_id should actually always be fine as is
        # self.all_items = []
        # self.mdp_items_id = self.items_id.copy()
        # for item in new_items:
        #     if item not in self.mdp_items_id:
        #         self.mdp_items_id[item] = len(items_id)
        # #Notion used to prioritize reset states
        # self.novel_items = []
        # for item in self.mdp_items_id:
        #     self.all_items.append(item)
        #     if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
        #                     'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
        #                     'polycraft:sack_polyisoprene_pellets']:
        #         self.novel_items.append(item)

        # self.novel_items = []
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
            if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
                            'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
                            'polycraft:sack_polyisoprene_pellets']:
                novel_items.append(item)
                if self.first_space_init:
                    if item not in self.novel_items:
                        print('WARNNING - Novel item {} has been discovered since last MDP action/obs space init, observations prior to and after this point will be mismatched'.format(item))
                # self.novel_items.append(item)

        self.novel_items = novel_items

        # self.mdp_items_id = self.set_items_id(item_list)
        self.set_items_id(item_list)

        #Need items_id to be aligned with MDP items id
        #TODO: ask Gyan if I can change this
        self.items_id = self.mdp_items_id.copy()
        # print('mdp items', self.mdp_items_id)
        # print('items', self.items_id)
        # print(self.all_items)

        self.mdp_inventory_items = list(self.mdp_items_id.keys())
        if 'minecraft:air' in self.mdp_inventory_items:
            self.mdp_inventory_items.remove('minecraft:air')
        if 'minecraft:bedrock' in self.mdp_inventory_items:
            self.mdp_inventory_items.remove('minecraft:bedrock')
        #remove pogostick?

        #Need some notion of entities in the obs space
        #I believe adding a new type for each id to differentiate between block and entity
        #  and extending relcoords to each of those is too much
        #Entities are also weird to put in the map because they can pop out on top of blocks
        #Solution - just add notion of how many entities of each id are in the world? So the agent
        #   can know when it caused an entity to pop up or pick one up. Downside is don't know how close
        #   to one we are, but picking them up should be handled in reset_to_interesting_state
        # self.mdp_entity_items = list(self.mdp_items_id.keys())
        # if 'minecraft:air' in self.mdp_inventory_items:
        #     self.mdp_inventory_items.remove('minecraft:air')
        # if 'minecraft:bedrock' in self.mdp_inventory_items:
        #     self.mdp_inventory_items.remove('minecraft:bedrock')
        #TODO: Best solution:
        # Only add notion of entity count if we've seen it post-novelty (+ blocks we know it's possible)
        # Extend action space to include pick_up actions for entities?
        #   move to and pick up entity if in map - *But what if can't motion plan? Just as confusing for the agent
        #   otherwise give greater neg reward


        # print(self.items_id)
        # print(self.recipes.keys())
        # print(self.inventory_items)

        # Generate all actions from current state of env and set action space
        # TODO: make sure at this point whatever novel object is present in env to be included
        self.manip_actions =  ['MOVE w',
                               'TURN -90',
                               'TURN 90',
                               'BREAK_BLOCK',
                               'PLACE_TREE_TAP',
                               'EXTRACT_RUBBER']

        # Add place_crafting_table to action list -> we can break it but not put it back currently
        self.manip_actions.append('PLACE_CRAFTING_TABLE')

        # Should crafting table be a recipe?
        self.crafting_actions = ['CRAFT ' + item for item in self.recipes.keys()]

        # REMOVE CRAFT crafting_table - don't think this is present in tournament, but is in API
        if 'CRAFT minecraft:crafting_table' in self.crafting_actions:
            self.crafting_actions.remove('CRAFT minecraft:crafting_table')

        # self.select_actions =  ['SELECT_ITEM polycraft:wooden_pogo_stick',
        #                         'SELECT_ITEM polycraft:tree_tap',
        #                         'SELECT_ITEM minecraft:planks',
        #                         'SELECT_ITEM minecraft:stick',
        #                         'SELECT_ITEM minecraft:crafting_table',
        #                         'SELECT_ITEM polycraft:sack_polyisoprene_pellets',
        #                         'SELECT_ITEM minecraft:log']
        self.select_actions = ['SELECT_ITEM ' + item for item in self.mdp_inventory_items]

        # TODO: planner has deselect action, but I don't see how you can deselect an item through API
        #   The initial selected_item is '', but there's no command I see to select nothing
        # And USE_HAND is different than break without object
        # print(self.execute_action('SELECT_ITEM '))
        # input('wait')

        # For testing purposes to assert that everything is working
        # self.manip_actions =  ['MOVE w',
        #                        'TURN -90',
        #                        'TURN 90']
        # self.crafting_actions = []
        # self.select_actions = []

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
        for item in ['minecraft:air', 'minecraft:bedrock', 'minecraft:planks', 'minecraft:stick', 'polycraft:sack_polyisoprene_pellets','polycraft:wooden_pogo_stick']:
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

        # print(self.get_nearest_items())

        #No notion of selected item anywhere? Assuming we are holding nothing to start
        #       NOT ALWAYS GOING TO BE TRUE
        # self.selected_item = 'minecraft:air'
        # self.selected_item_id = 0
        # self.selected_item_id = self.items_id[self.selected_item]

        # Need to update map using updated items_id set
        sense_all_command_result = self.run_SENSE_ALL_and_update('NONAV')
        self.accumulated_step_cost += sense_all_command_result['stepCost']

        self.num_types = len(self.mdp_items_id)+1

        print('updated items, recipes, and actions in MDP')
        print('Items: ', self.mdp_items_id)
        print('Craft items: ', self.recipes.keys())
        print('Actions: ', self.actions_id)
        self.first_space_init = True

    # # def set_items_id(self, items, novel_items):
    # #     items_id = {}
    # #     self.all_items = []
    # #     if 'minecraft:air' in items:
    # #         items_id['minecraft:air'] = 0
    # #         self.all_items.append('minecraft:air')
    # #     for item in sorted(items):
    # #         if item != 'minecraft:air':
    # #             if 'minecraft:air' in items:
    # #                 items_id[item] = len(items_id)
    # #             else:
    # #                 items_id[item] = len(items_id) + 1
    # #             self.all_items.append(item)
    # #
    # #     #To make sure ids align, believe interface sets novel items to len(ids)+1 in the order they appear
    # #     for item in novel_items:
    # #         self.all_items.append(item)
    # #         if 'minecraft:air' in items:
    # #             items_id[item] = len(items_id)
    # #         else:
    # #             items_id[item] = len(items_id) + 1
    # #
    # #     return items_id
    #
    # # Function to be called post-novelty to update observation and action space
    # #   to be used by RL agent
    # def generate_obs_action_spaces(self, new_items=[]):
    #     #MDP items_id must be same as env items_id, but we want to include items that the env has not seen before
    #     self.all_items = []
    #     self.mdp_items_id = self.env.items_id.copy()
    #     for item in new_items:
    #         if item not in self.mdp_items_id:
    #             self.mdp_items_id[item] = len(items_id)
    #     #Notion used to prioritize reset states
    #     self.novel_items = []
    #     for item in self.mdp_items_id:
    #         self.all_items.append(item)
    #         if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
    #                         'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
    #                         'polycraft:sack_polyisoprene_pellets']:
    #             self.novel_items.append(item)
    #
    #
    #
    #     # # self.novel_items = []
    #     # novel_items = []
    #     # # Want complete list of items_ids, not just whats in map
    #     # # Take everything that's currently in the map and part of crafting world
    #     # # TODO: Need to do this smarter, if completed items pops up in the middle of a
    #     # #           round but post 'novelty-detection', this will not pick it up
    #     # item_list = []
    #     # for item in self.ingredients_quantity_dict:
    #     #     item_list.append(item)
    #     #     for i, ingredients_quantity_dict in enumerate(self.ingredients_quantity_dict[item]):
    #     #         for ingredient in self.ingredients_quantity_dict[item][i]:
    #     #             item_list.append(ingredient)
    #     #
    #     # # Anything passed in new_items that wasn't found elsewhere
    #     # # for item in new_items:
    #     # #     if item not in item_list:
    #     # #         item_list.append(item)
    #     #
    #     # item_list = set(list(item_list) + list(self.items_location.keys()) + list(self.inventory_quantity_dict.keys()))
    #     #
    #     # for item in item_list:
    #     #     if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
    #     #                     'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
    #     #                     'polycraft:sack_polyisoprene_pellets'] and item not in new_items:
    #     #         # self.novel_items.append(item)
    #     #         novel_items.append(item)
    #     #         if self.first_space_init:
    #     #             if item not in self.novel_items:
    #     #                 print('WARNING - Novel item {} has been discovered since last MDP action/obs space init, observations prior to and after this point will be mismatched'.format(item))
    #     #
    #     # self.novel_items = novel_items + new_items
    #     #
    #     # print(item_list, new_items)
    #     # self.mdp_items_id = self.set_items_id(item_list, new_items)
    #     # print('mdp items', self.mdp_items_id)
    #     # print('env items', self.env.items_id)
    #
    #     self.inventory_items = list(self.mdp_items_id.keys())
    #     if 'minecraft:air' in self.inventory_items:
    #         self.inventory_items.remove('minecraft:air')
    #     if 'minecraft:bedrock' in self.inventory_items:
    #         self.inventory_items.remove('minecraft:bedrock')
    #     # print(self.items_id)
    #     # print(self.recipes.keys())
    #     # print(self.inventory_items)
    #
    #     # Generate all actions from current state of env and set action space
    #     # TODO: make sure at this point whatever novel object is present in env to be included
    #     self.manip_actions =  ['MOVE w',
    #                            'TURN -90',
    #                            'TURN 90',
    #                            'BREAK_BLOCK',
    #                            'PLACE_TREE_TAP',
    #                            'EXTRACT_RUBBER']
    #
    #     # Add place_crafting_table to action list -> we can break it but not put it back currently
    #     self.manip_actions.append('PLACE_CRAFTING_TABLE')
    #
    #     # Should crafting table be a recipe?
    #     self.crafting_actions = ['CRAFT ' + item for item in self.recipes.keys()]
    #
    #     # REMOVE CRAFT crafting_table - don't think this is present in tournament, but is in API
    #     if 'CRAFT minecraft:crafting_table' in self.crafting_actions:
    #         self.crafting_actions.remove('CRAFT minecraft:crafting_table')
    #
    #     # self.select_actions =  ['SELECT_ITEM polycraft:wooden_pogo_stick',
    #     #                         'SELECT_ITEM polycraft:tree_tap',
    #     #                         'SELECT_ITEM minecraft:planks',
    #     #                         'SELECT_ITEM minecraft:stick',
    #     #                         'SELECT_ITEM minecraft:crafting_table',
    #     #                         'SELECT_ITEM polycraft:sack_polyisoprene_pellets',
    #     #                         'SELECT_ITEM minecraft:log']
    #     self.select_actions = ['SELECT_ITEM ' + item for item in self.inventory_items]
    #
    #     # TODO: planner has deselect action, but I don't see how you can deselect an item through API
    #     #   The initial selected_item is '', but there's no command I see to select nothing
    #     # And USE_HAND is different than break without object
    #     # print(self.execute_action('SELECT_ITEM '))
    #     # input('wait')
    #
    #     # For testing purposes to assert that everything is working
    #     # self.manip_actions =  ['MOVE w',
    #     #                        'TURN -90',
    #     #                        'TURN 90']
    #     # self.crafting_actions = []
    #     # self.select_actions = []
    #
    #     self.all_actions = self.manip_actions + self.crafting_actions + self.select_actions
    #     self.actions_id = {}
    #     for i in range(len(self.all_actions)):
    #         self.actions_id[self.all_actions[i]] = i
    #     self.action_space = spaces.Discrete(len(self.actions_id))
    #
    #     # TODO: compute max possible number of items given an env? or just set to arbitrary cap
    #     self.max_items = 20
    #     # Make observation_space
    #     agent_map_size = (self.agent_view_size + 1) ** 2
    #     low_agent_map = np.zeros(agent_map_size)
    #     high_agent_map = (len(self.mdp_items_id)+1) * np.ones(agent_map_size)
    #     low_orientation = np.array([0])
    #     high_orientation = np.array([4])
    #     y_max, x_max = self.map_to_plot.shape
    #     # How many rel_coords items are we going to use? All possible?
    #     self.interesting_items = self.mdp_items_id.copy()
    #     del self.interesting_items['minecraft:air']
    #     low_rel_coords = np.array([[-x_max, y_max] for i in range(len(self.interesting_items))]).flatten()
    #     high_rel_coords = np.array([[x_max, y_max] for i in range(len(self.interesting_items))]).flatten()
    #     low_map = np.concatenate((low_agent_map, low_orientation, low_rel_coords))
    #     high_map = np.concatenate((high_agent_map, high_orientation, high_rel_coords))
    #     low = np.concatenate((low_map,[0],np.zeros(len(self.inventory_items))))
    #     high = np.concatenate((high_map, [len(self.mdp_items_id)+1], self.max_items*np.ones(len(self.inventory_items))))
    #     self.observation_space = spaces.Box(low, high, dtype=np.float32)
    #
    #     # TODO: what is the best way to do this
    #     # Need to update map using updated mdp_items_id set
    #     sense_all_command_result = self.run_SENSE_ALL_and_update('NONAV')
    #     self.accumulated_step_cost += sense_all_command_result['stepCost']
    #
    #     self.num_types = len(self.mdp_items_id)+1
    #
    #     print('updated items, recipes, and actions in MDP')
    #     print('Items: ', self.mdp_items_id)
    #     print('Craft items: ', self.recipes.keys())
    #     print('Actions: ', self.actions_id)
    #     self.first_space_init = True


    def step(self, action_id):
        # action_id = int(input('action'))
        action = self.all_actions[action_id]

        # Need to map to action string?
        action_result, sense_all_result = self.execute_action(action)
        self.accumulated_step_cost += action_result['stepCost'] + sense_all_result['stepCost']
        self.last_step_cost = action_result['stepCost'] + sense_all_result['stepCost']

        #Should be caught in dynamics_agent check
        # if len(self.items_id) >= self.num_types:
        #     print('gridworldMDP, novel item added to items_id')
        # if len(self.env.items_id) >= self.num_types:
        #     print('gridworldMDP, novel item added to underlying items_id')

        obs = self.observation()

        return obs, None, None, self.get_info()

    def get_info(self):
        # info = {'bin_map': self.binary_map, 'map_limits': (self.x_max, self.y_max), \
        info = {'items_locs': self.items_location.copy(), \
                'entities_locs': self.entities_location, \
                'block_in_front': self.block_in_front.copy(), \
                'inv_quant_dict': self.inventory_quantity_dict.copy(), \
                # 'inv_quant_list': self.inventory_list, \
                # 'ingred_quant_dict': self.ingredients_quantity_dict, \
                # 'player': self.player, 'local_view_size': self.local_view_size, \
                'player': self.player, \
                'selected_item': self.selected_item,
                'total_step_cost': self.accumulated_step_cost,
                'last_step_cost': self.last_step_cost}

        return info

    def reset(self, task=None):
        # if task is None:
        #     task = self.task
        self.accumulated_step_cost = 0
        # self.reset()
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
        # print(state)

        # state.astype(np.float).ravel()

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

        # print(local_view)

        return local_view
        # """
        # Slice map with 0 padding based on agent_view_size
        #
        # :return: local view of the agent
        # """
        #
        # envmap = self.map_to_plot
        # agent_facing = self.player['facing']
        # desired_center = (self.player['pos'][0], self.player['pos'][2])
        #
        # extend = [int(self.agent_view_size/2), int(self.agent_view_size/2)]  # row and column
        # pad_value = 0
        #
        # extend = np.asarray(extend)
        # map_ext_shp = envmap.shape + 2 * np.array(extend)
        # map_ext = np.full(map_ext_shp, pad_value)
        # insert_idx = [slice(i, -i) for i in extend]
        # map_ext[tuple(insert_idx)] = envmap
        # region_idx = [slice(i, j) for i, j in zip(desired_center, extend * 2 + 1 + desired_center)]
        # area = map_ext[tuple(region_idx)]
        #
        # if agent_facing == 'NORTH':
        #     out = area
        # elif agent_facing == 'SOUTH':
        #     out = np.flip(area)
        # elif agent_facing == 'EAST':
        #     out = np.rot90(area, 1)
        # elif agent_facing == 'WEST':
        #     out = np.rot90(area, 3)
        # else:
        #     print("unknown agent facing id: ", agent_facing_id)
        #     quit()
        #
        # return out.flatten()

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

        # nearest_dists = [np.inf] * len(self.items_id)
        # nearest_coords = np.zeros((len(self.items_id), 2))
        #
        # envmap = self.map_to_plot.copy()
        # agent_x_ = self.player['pos'][0]
        # agent_y_ = self.player['pos'][2]
        #
        # if (self.player["facing"] == "NORTH"):
        #     agent_y = agent_x_
        #     agent_x = agent_y_
        # elif (self.player["facing"] == "SOUTH"):
        #     envmap = np.flipud(envmap)
        #     agent_y = envmap.shape[0]-1-agent_x_
        #     agent_x = agent_y_
        # elif (self.player["facing"] == "EAST"):
        #     envmap = np.rot90(envmap, 1)
        #     agent_y = envmap.shape[1]-1-agent_y_
        #     agent_x = agent_x_
        # elif (self.player["facing"] == "WEST"):
        #     envmap = np.rot90(envmap, 3)
        #     agent_y = agent_y_
        #     agent_x = envmap.shape[0]-1-agent_x_
        #
        # # nearest dist should be manhattan distance
        # for i in range(envmap.shape[0]):
        #     for j in range(envmap.shape[1]):
        #         item_id = int(envmap[i][j])
        #         # if item_id in self.item_ids:
        #         if item_id in range(len(self.items_id)):
        #             dist = np.abs(agent_x - j) + np.abs(agent_y - i)
        #             # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
        #             if dist < nearest_dists[item_id]:
        #                 nearest_dists[item_id] = dist
        #                 nearest_coords[item_id] = (i, j)
        #                 # if self.env.agent_facing_id == 1:
        #                 if (self.player["facing"] == "SOUTH"):
        #                     nearest_coords[item_id] = (agent_y - i, agent_x - j)
        #                 else:
        #                     nearest_coords[item_id] = (agent_y - i, j - agent_x)
        #
        # return nearest_coords[list(self.interesting_items.values())]


    def generate_inv_list(self):
        # print(self.inventory_list)
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
            if self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y-1][x] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x-1] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x+1] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            else:
                return
        else:
            if self.map_to_plot[y+1][x] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y-1][x] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x-1] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x+1] == self.mdp_items_id['minecraft:log']:
                return True
            else:
                return False


# Gridworld env to match with updated RL polycraft interface and allow
#   for commands like SENSE_ALL etc without simulating socket connection


##############################################################

# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
# import basic libs
import socket
import time
import ast
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import copy
import sys
from IPython import embed
from gym import spaces
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append('../utils')
from polycraft_tufts.utils.utils import get_active_windows, get_paths, sleep_and_display

NG_PC_COMMANDS = {'MOVE w': 'SMOOTH_MOVE W',
                  'TURN -90': 'SMOOTH_TURN -90',
                  'TURN 90': 'SMOOTH_TURN 90',
                  'BREAK_BLOCK': 'BREAK_BLOCK',
                  'PLACE_TREE_TAP': 'PLACE_BLOCK polycraft:tree_tap',
                  'PLACE_CRAFTING_TABLE': 'PLACE_BLOCK minecraft:crafting_table',
                  'EXTRACT_RUBBER': 'EXTRACT_RUBBER'}


class PolycraftInterface:
    def __init__(self, use_trade, trade_socket, host=None, port=None, reset_command=None, render_bool=False, save_json=False, using_ng=False):
        if use_trade and trade_socket is None:
            print('[PolycraftInterface] ERROR, use_trade is set to true but no socket connection to trade has been supplied')
            quit()

        # Given attributes:
        self.host = host
        self.port = port
        self.reset_command = reset_command
        self.render_bool = render_bool
        self.save_json = save_json
        self.game_over = False
        self.game_success = False
        self.use_trade = use_trade

        # Processed SENSE_RECIPES attributes:
        self.recipes = {}
        self.crafting_table_needed_dict = {}
        self.ingredients_quantity_dict = {}

        # SENSE_ALL attributes:
        self.block_in_front = {}  # name
        self.inventory_quantity_dict = {}
        self.player = {}  # pos, facing, yaw, pitch
        self.destination_pos = []
        self.entities = {}
        self.map = {}

        # Processed SENSE_ALL attributes:
        self.entities_location = {}  # contains entity's name and its locations in env.
        self.map_origin = [0, 0, 0]
        self.map_size = [0, 0, 0]  # x, z, y
        self.x_max, self.z_max, self.y_max = self.map_size
        self.items_location = {}  # contains item's name and its locations in env.
        self.items_id = {}  # contains name and ID of items in env., starts from 1
        self.map_to_plot = []  # contains item ID of items in env.
        self.binary_map = []  # contains 0 for 'minecraft:air', otherwise 1
        self.using_ng = using_ng

        # Constants specific to PAL
        self.move_commands = ['SMOOTH_MOVE', 'MOVE', 'SMOOTH_TURN', 'TURN', 'SMOOTH_TILT']
        self.move_commands.extend(['MOVE_NORTH', 'MOVE_SOUTH', 'MOVE_EAST', 'MOVE_WEST', 'LOOK_NORTH', 'LOOK_SOUTH',
                                   'LOOK_EAST', 'LOOK_WEST'])  # RL agent nav. actions

        if self.save_json:
            task = self.reset_command.split(" ")[2].split('/')[-1]
            csv_filename = task + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

            _, POLYCRAFT_TUFTS_PATH, _ = get_paths(os.getcwd(), 'PAL')
            os.makedirs(POLYCRAFT_TUFTS_PATH + os.sep + "datasets", exist_ok=True)
            self.csv_filepath = POLYCRAFT_TUFTS_PATH + os.sep + "datasets" + os.sep + csv_filename
            with open(self.csv_filepath, 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Command", "JSON", "player", "block_in_front", "x_max", "y_max", "items_location",
                                 "binary_map", "inventory_quantity_dict", "ingredients_quantity_dict", "command_result"]
                                )

        if not use_trade:
            """
            If Polycraft is running, simply connects the socket.
            If Polycraft is not running, start Polycraft, run START command.
            """
            windows = get_active_windows()
            #This is technically the conditional on USING_TOURNAMENT_MANAGER
            if 'Minecraft 1.8.9' in windows:
                print('Minecraft is running')
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                if self.reset_command is not None:
                    self.run_a_command_and_update_map(self.reset_command, sleep_time=2)  # takes time to reset
                else:
                    self.run_SENSE_ALL_and_update('NONAV')
            else:
                print('Minecraft is not running, starting the game, connecting the socket and sending START command. '
                      'Be patient :)')
                current_path, _, POLYCRAFT_PATH = get_paths(os.getcwd(), 'PAL')
                os.chdir(POLYCRAFT_PATH)
                if platform.system() == 'Linux':
                    # process = subprocess.Popen('xterm -hold -e sudo ./gradlew runClient &', shell=True)
                    process = subprocess.Popen('konsole --hold -e sudo ./gradlew runClient &', shell=True)
                    sleep_and_display(70)  # polycraft takes ~ 0:60 to load
                elif platform.system() == 'Windows':
                    process = subprocess.Popen('start /wait LaunchPolycraft.bat', shell=True)
                    sleep_and_display(120)  # polycraft takes ~ 2:00 to load
                os.chdir(current_path)
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.send_command_and_save_json('START')
                self.run_a_command_and_update_map(self.reset_command, sleep_time=10)  # takes time to reset
        else:
            self.sock = trade_socket
        #     self.run_SENSE_ALL_and_update('NONAV')

        # Update: self.recipes, self.crafting_table_needed_dict, self.ingredients_quantity_dict
        # self.run_SENSE_RECIPES_and_update()

        # self.run_SENSE_ALL_and_update()


    def send_command(self, command):
        """
        Send command to the env.
        """

        # Block code necessary to run with socket_env_polycraft.py
        if self.using_ng:
            if command in NG_PC_COMMANDS:
                command = NG_PC_COMMANDS[command]

        # Wait before SENSE_ALL NONAV to get updated info.
        # if command == "SENSE_ALL NONAV":
        #     sleep_and_display(0.5)
        # print('Sending command: {}'.format(command))
        self.sock.send(str.encode(command + '\n'))
        # print("Command: ", command)
        BUFF_SIZE = 4096  # 4 KiB
        data = b''
        while True:
            time.sleep(0.00001)
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break

        output=json.loads(data)
        if data.decode('UTF-8') == 'null\n':
            print('recieved back null after send command to TRADE, resending last command {}'.format(command))
            return self.send_command(command)

        if output['gameOver'] == True:
            if output['goal']['goalAchieved']:
                self.game_success = True
            self.game_over = True


        return output

    def write_to_csv(self, command, output):

        if self.save_json:
            if 'command_result' not in output:
                output['command_result'] = ""
            with open(self.csv_filepath, 'a') as f:  # append to the file created
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([command, output, self.player, self.block_in_front, self.x_max, self.y_max,
                                 self.items_location, self.binary_map, self.inventory_quantity_dict,
                                 self.ingredients_quantity_dict, output['command_result']])

    def send_command_and_save_json(self, command, sleep_time=0):
        """
        Use it for commands that do not update processed attributes in __init__ like START
        """

        output = self.send_command(command)
        if sleep_time:
            sleep_and_display(sleep_time)
        self.write_to_csv(command, output)

    # Methods to process SENSE commands

    def run_SENSE_RECIPES_and_update(self):
        """
        Generate recipes, crafting_table_needed_dict, ingredients_quantity_dict
        Example of recipes:
        {'polycraft:wooden_pogo_stick':
          [{'ingredients': {'0': {'Item': 'minecraft:stick', 'stackSize': 1, 'slot': 0},
                            '1': {'Item': 'minecraft:stick', 'stackSize': 1, 'slot': 1},
                            ...},
            'output_item': {'Item': 'polycraft:wooden_pogo_stick', 'stackSize': 1, 'slot': 9}},
           {'ingredients': {'0': {'Item': 'minecraft:planks', 'stackSize': 1, 'slot': 0},
                             '1': {'Item': 'polycraft:bag_polyisoprene_pellets', 'stackSize': 1, 'slot': 1},
                            ...},
            'output_item': {'Item': 'polycraft:wooden_pogo_stick', 'stackSize': 1, 'slot': 9}}
          ]
        """

        self.recipes = {}
        self.crafting_table_needed_dict = {}
        self.ingredients_quantity_dict = {}

        command = "SENSE_RECIPES"
        reset_output = self.send_command(command)  # send command

        for recipes_dict in reset_output["recipes"]:
            ingredients = {}
            # In Minecraft recipes are always represented by 9 slots
            for slot_no in range(9):
                # Handling items that requires only one item to craft
                if (slot_no == 0) and (len(recipes_dict['inputs']) == 1):
                    if recipes_dict['inputs'][slot_no]['slot'] == -1:
                        ingredients[str(slot_no)] = recipes_dict['inputs'][slot_no]
                        for slot_no2 in range(slot_no + 1, 4):
                            ingredients[str(slot_no2)] = None
                        break

                for an_item in recipes_dict['inputs']:
                    if an_item['slot'] == slot_no:
                        ingredients[str(slot_no)] = an_item
                        break
                # for blank slots, fill with None
                if str(slot_no) not in ingredients:
                    ingredients[str(slot_no)] = None
            # To create the same item, there can be different recipes, storing each in a list's element
            self.recipes.setdefault(recipes_dict["outputs"][0]['Item'], [])
            self.recipes[recipes_dict["outputs"][0]['Item']].append({'ingredients': ingredients,
                                                                     'output_item': recipes_dict['outputs'][0]})

        # Finding which item needs crafting table
        for item_to_craft in self.recipes:
            self.crafting_table_needed_dict.setdefault(item_to_craft, [])
            for a_recipe in self.recipes[item_to_craft]:
                craft_command = "CRAFT " + str(1) + " "
                for a_ingredient in range(len(a_recipe['ingredients'])):
                    if a_recipe['ingredients'][str(a_ingredient)] is not None:
                        craft_command += a_recipe['ingredients'][str(a_ingredient)]['Item'] + " "
                    else:
                        craft_command += "0 "
                craft_command = craft_command[:-1]
                craft_command_list = np.array(craft_command.split(' '))
                if len(craft_command_list) == 11:
                    crafting_table_needed = False
                    for i in [4, 7, 8, 9, 10]:
                        if craft_command_list[i] != '0':
                            crafting_table_needed = True
                    if not crafting_table_needed:
                        craft_command = " ".join(craft_command_list[np.array([0, 1, 2, 3, 5, 6])])
                craft_command_list = np.array(craft_command.split(' '))
                if len(craft_command_list) == 11:
                    self.crafting_table_needed_dict[item_to_craft].append(True)
                else:
                    self.crafting_table_needed_dict[item_to_craft].append(False)

        # Finding self.ingredients_quantity_dict
        for item_to_craft in self.recipes:
            self.ingredients_quantity_dict.setdefault(item_to_craft, [])

            for a_recipe in self.recipes[item_to_craft]:
                ingredients_quantity_dict_for_item_to_craft = {}

                for slot in a_recipe['ingredients']:
                    if a_recipe['ingredients'][slot] != None:
                        ingredients_quantity_dict_for_item_to_craft.setdefault(a_recipe['ingredients'][slot]['Item'], 0)
                        ingredients_quantity_dict_for_item_to_craft[a_recipe['ingredients'][slot]['Item']] += 1

                self.ingredients_quantity_dict[item_to_craft].append(ingredients_quantity_dict_for_item_to_craft)

        self.write_to_csv(command, reset_output)  # Keep in the end as it saves processed attributes

    def run_SENSE_INVENTORY_and_update(self, inventory_output_inventory=None):

        if not isinstance(inventory_output_inventory, dict):
            inventory_output = self.send_command('SENSE_INVENTORY')  # send command
            sense_inventory_command_result = inventory_output['command_result']
            inventory_output_inventory = inventory_output['inventory']  # goal, command_result
        else:
            sense_inventory_command_result = {}

        self.inventory_quantity_dict = {}
        for slot in inventory_output_inventory:
            if inventory_output_inventory[slot]['item'] != '' and slot != 'selectedItem':
                if inventory_output_inventory[slot]['item'] in self.inventory_quantity_dict:
                    self.inventory_quantity_dict[inventory_output_inventory[slot]['item']] += inventory_output_inventory[slot]['count']
                else:
                    self.inventory_quantity_dict[inventory_output_inventory[slot]['item']] = inventory_output_inventory[slot]['count']

        if not isinstance(inventory_output_inventory, dict):
            self.write_to_csv('SENSE_INVENTORY', inventory_output)  # Keep in the end as it saves processed attributes

        return sense_inventory_command_result

    def run_SENSE_LOCATIONS_and_update(self):
        # does not update blockInFront

        sense_location_output = self.send_command('SENSE_LOCATIONS')  # send command
        sense_location_output_command_result = sense_location_output['command_result']
        self.player = sense_location_output['player']  # contains player's pos, facing, yaw, pitch
        self.write_to_csv('SENSE_LOCATIONS', sense_location_output)

        return sense_location_output_command_result

    def run_SENSE_ALL_and_update(self, parameter=None, set_agent_id=False, sense_recipes=False):
        """
        set_agent_id: set to True only when visualize_env_2d() is used for plotting
                      because agent is represented by an ID 1 on the plot
        """
        # Get env. info.
        parameter = 'NONAV'
        if parameter == None:
            command = "SENSE_ALL"
            sense_all_output = self.send_command(command)
        else:
            command = "SENSE_ALL " + parameter
            sense_all_output = self.send_command(command)

        self.block_in_front = sense_all_output['blockInFront']  # name
        self.run_SENSE_INVENTORY_and_update(
            sense_all_output['inventory'])  # key is slot no. in inventory. Update: self.inventory_quantity_dict
        self.player = sense_all_output['player']  # contains player's pos, facing, yaw, pitch
        self.destination_pos = sense_all_output['destinationPos']
        self.entities = sense_all_output['entities']
        self.map = sense_all_output['map']
        sense_all_command_result = sense_all_output['command_result']
        if sense_recipes:
            self.run_SENSE_RECIPES_and_update()
        # print(self.entities)

        if parameter == None:
            self.entities_location = {}  # contains entity's name and its locations in env.
            for location, an_entity in self.entities.items():
                self.entities_location.setdefault(an_entity['item'], [])
                x = int(an_entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[0].split('=')[1])
                z = int(an_entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[1].split('=')[1])
                y = int(an_entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[2].split('=')[1])

                self.entities_location[an_entity['item']].append('{},{},{}'.format(x,z,y))
                # self.entities_location[an_entity['item']].append(location)

            self.map_size = self.map['size']
            self.x_max, self.z_max, self.y_max = self.map_size
            self.map_origin = self.map['origin']

        elif parameter == 'NONAV':
            # Finding x_max, y_max and items_id from SENSE_ALL NONAV
            # Storing all the locations of items in items_location to be used in run_TP_TO_and_update_map
            self.x_max, self.y_max = 0, 0
            items_list = []

            self.items_location = {}  # contains item's name and its locations in env.
            for xzy, item in self.map.items():
                items_list.append(item['name'])
                self.items_location.setdefault(item['name'], [])
                self.items_location[item['name']].append(xzy)

                x, y = int(xzy.split(',')[0]), int(xzy.split(',')[2])
                if x > self.x_max:
                    self.x_max = x
                if y > self.y_max:
                    self.y_max = y

            self.x_max += 1
            self.y_max += 1

            # for item in self.ingredients_quantity_dict:
            #     items_list.append(item)
            #     for i, ingredients_quantity_dict in enumerate(self.ingredients_quantity_dict[item]):
            #         for ingredient in self.ingredients_quantity_dict[item][i]:
            #             items_list.append(ingredient)

            # items_list = set(items_list)

            # Assigning an id for each item in env.
            if set_agent_id:
                self.items_id.setdefault('agent', len(self.items_id) + 1)

            for item in sorted(set(items_list)):
                self.items_id.setdefault(item, len(self.items_id) + 1)

            if self.num_types == 0:
                self.num_types = len(self.items_id)+1

            # Filling a 2D list to plot as map
            self.map_to_plot = np.zeros((self.y_max, self.x_max))  # Y (row) is before X (column) in matrix

            for xzy, item in self.map.items():
                x, y = int(xzy.split(',')[0]), int(xzy.split(',')[2])
                self.map_to_plot[y][x] = self.items_id[item['name']]

            #don't want binary map to include entities
            self.binary_map = np.where(self.map_to_plot == self.items_id['minecraft:air'], 0, 1)

            self.entities_location = {}  # contains entity's name and its locations in env.
            for location, entity in self.entities.items():
                self.items_id.setdefault(entity['item'], len(self.items_id) + 1)
                self.entities_location.setdefault(entity['item'], [])
                # self.entities_location[entity['item']].append(location)

                x = int(entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[0].split('=')[1])
                z = int(entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[1].split('=')[1])
                y = int(entity['Pos'].split('BlockPos')[1].replace(' ', '')[1:-1].split(',')[2].split('=')[1])

                self.entities_location[entity['item']].append('{},{},{}'.format(x,z,y))

                #separate set of ids for blocks and entities in map
                self.map_to_plot[y][x] = self.items_id[entity['item']]+self.num_types

            if set_agent_id:
                x, z, y = self.player['pos']
                self.map_to_plot[y][x] = self.items_id['agent']

            # self.binary_map = np.where(self.map_to_plot == self.items_id['minecraft:air'], 0, 1)

        # EW: Wasn't storing selected item anywhere, setting here from senseall
        self.selected_item = sense_all_output['inventory']['selectedItem']['item']
        try:
            self.selected_item_id = self.items_id[self.selected_item]
        except:
            # Equate holding air to holding nothing
            self.selected_item_id = 0

        self.write_to_csv(command, sense_all_output)  # Keep in the end as it saves processed attributes

        return sense_all_command_result

    # Methods that interact or change the environment

    def run_TP_TO_and_update_map(self, location):
        """
        Teleport to given location and update its command_result
        """

        command = 'TP_TO ' + location
        output = self.send_command(command)
        self.write_to_csv(command, output)

        sleep_and_display(1)  # takes time to teleport

        tp_to_command_result = output['command_result']
        if self.render_bool:
            sense_all_command_result = self.visualize_env_2d()
        else:
            sense_all_command_result = self.run_SENSE_ALL_and_update(parameter='NONAV')  # Give all item's location

        return tp_to_command_result, sense_all_command_result

    def run_MOVE_and_update_location(self, move_command):
        """
        Run a move command and update agent's location
        Use it for commands that changes agent's location in the env. like:
        'MOVE_FORWARD','MOVE_NORTH','MOVE_SOUTH','MOVE_EAST','MOVE_WEST'
        """

        move_output = self.send_command(move_command)
        self.write_to_csv(move_command, move_output)

        if self.render_bool:
            sense__command_result = self.visualize_env_2d(sense_location=True)
        else:
            # sense__command_result = self.run_SENSE_LOCATIONS_and_update() # does not update blockInFront
            sense__command_result = self.run_SENSE_ALL_and_update()

        return move_output['command_result'], sense__command_result

    def run_CRAFT_and_update_inventory(self, item_to_craft):
        """
        CRAFT given item by finding its recipe from the RESET command
        If RESET does not has its recipe, just pass the complete "CRAFT" command with required parameters
        """

        # print("Crafting: ", item_to_craft)
        # To create the same item, there can be different recipes, stored in different elements of self.recipes list
        # Trying each recipe until SUCCESS
        if item_to_craft in self.recipes:

            for a_recipe in self.recipes[item_to_craft]:

                command = "CRAFT " + str(1) + " "  # Always craft 1 recipe
                for a_ingredient in range(len(a_recipe['ingredients'])):
                    if a_recipe['ingredients'][str(a_ingredient)] != None:
                        command += a_recipe['ingredients'][str(a_ingredient)]['Item'] + " "
                    else:
                        command += "0 "
                command = command[:-1]

                """
                Items that needs 4 items to crafted, may not need to be near crafting table,
                so passing only 4 items in slots: 0, 1, 3, 4
                """
                craft_command_list = np.array(command.split(' '))
                if len(craft_command_list) == 11:
                    crafting_table_needed = False
                    for i in [4, 7, 8, 9, 10]:
                        if craft_command_list[i] != '0':
                            crafting_table_needed = True
                    if not crafting_table_needed:
                        command = " ".join(craft_command_list[np.array([0, 1, 2, 3, 5, 6])])

                output = self.send_command(command)
                craft_command_result = output['command_result']
                if craft_command_result['result'] == 'SUCCESS':
                    self.write_to_csv(command, output)
                    return craft_command_result, self.run_SENSE_INVENTORY_and_update()  # update the inventory
                # else:
                    # print("The recipe did not worked, trying another recipe or exiting ...")
                    # print("Command message: ", craft_command_result['message'])
            self.write_to_csv(command, output)
            return craft_command_result, self.run_SENSE_INVENTORY_and_update()  # return if FAIL
        else:
            output = self.send_command(item_to_craft)
            craft_command_result = output['command_result']
            self.write_to_csv(item_to_craft, output)
            return craft_command_result, self.run_SENSE_INVENTORY_and_update()

    def run_a_command_and_update_map(self, a_command, sleep_time=0):
        """
        Run a command and update map
        Use it for commands that changes item's position in the env. like:
        'RESET', 'BREAK_BLOCK', 'PLACE_TREE_TAP', 'EXTRACT_RUBBER'
        """

        if a_command is not None:
            a_command_output = self.send_command(a_command)
            self.write_to_csv(a_command, a_command_output)
            if sleep_time:
                sleep_and_display(sleep_time)

        if self.render_bool:
            sense_all_command_result = self.visualize_env_2d()
        else:
            if a_command.split()[0] == 'RESET':
                sense_all_command_result = self.run_SENSE_ALL_and_update(parameter='NONAV', sense_recipes=True)  # Give all item's location
            else:
                sense_all_command_result = self.run_SENSE_ALL_and_update(parameter='NONAV')  # Give all item's location

        return a_command_output['command_result'], sense_all_command_result

    def execute_action(self, action):
        """
        A wrapper function that performs action and sense accourding to that action
        """

        if action.split(' ')[0].upper() in self.move_commands:
            move_output_command_result, sense__command_result = self.run_MOVE_and_update_location(action)
            return move_output_command_result, sense__command_result
        # elif action in self.recipes:
        elif action.split()[0].upper() == 'CRAFT':
            craft_command_result, sense_inventory_command_result = self.run_CRAFT_and_update_inventory(action.split()[1])
            return craft_command_result, sense_inventory_command_result
        else:
            a_command_result, sense_all_command_result = self.run_a_command_and_update_map(action)
            return a_command_result, sense_all_command_result

    def visualize_env_2d(self, sense_location=False):

        if sense_location:
            # sense__command_result = self.run_SENSE_LOCATIONS_and_update() # does not update blockInFront
            sense__command_result = self.run_SENSE_ALL_and_update()
        else:
            sense__command_result = self.run_SENSE_ALL_and_update(parameter='NONAV')  # Give all item's location

        x, z, y = self.player['pos']

        x2, y2 = 0, 0
        if self.player['facing'] == 'NORTH':
            x2, y2 = 0, -0.01
        elif self.player['facing'] == 'SOUTH':
            x2, y2 = 0, 0.01
        elif self.player['facing'] == 'WEST':
            x2, y2 = -0.01, 0
        elif self.player['facing'] == 'EAST':
            x2, y2 = 0.01, 0

        plt.figure("Polycraft World", figsize=[5, 4.5])
        plt.imshow(self.map_to_plot, cMAP="gist_ncar")
        plt.arrow(x, y, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH\n' + 'Agent is facing ' + self.player['facing'])
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.colorbar()
        plt.pause(0.01)
        plt.clf()

        return sense__command_result


# Adapted from env_v2.py


# from polycraft_tufts.envs.polycraft_interface import PolycraftInterface
from polycraft_tufts.rl_agent.dqn_lambda.envs.polycraft_interface import PolycraftInterface

# Env class to interface between polycraft socket connection and RL agents
class PolycraftMDP(PolycraftInterface):

    def __init__(self, use_trade, socket=None, host=None, port=None, task=None, agent_view_size=4, render=False, SIMULATE=False, using_ng=False):

        if (SIMULATE):
            return
        PolycraftInterface.__init__(self, use_trade, socket, host, port, task, render_bool=render, save_json=False, using_ng=using_ng)
        # PolycraftInterface.__init__(self, host, port, task, render_bool=render, save_json=False)

        # local view size
        self.agent_view_size = agent_view_size
        self.task = task

        # Desired actions for RL agent prenovelty
        # actions_id = {
        #     'MOVE w': 0,
        #     'TURN -90': 1,
        #     'TURN 90': 2,
        #     'BREAK_BLOCK': 3,
        #     'PLACE_TREE_TAP': 4,
        #     'EXTRACT_RUBBER': 5,
        #     # Create craft actions for every recipe from current state of PolycraftInterface
        #     'polycraft:wooden_pogo_stick': 6,
        #     'polycraft:tree_tap': 7,
        #     'minecraft:planks': 8,
        #     'minecraft:stick': 9,
        #     # Create select actions for every item from current state of PolycraftInterface
        #     'SELECT_ITEM polycraft:wooden_pogo_stick': 10,
        #     'SELECT_ITEM polycraft:tree_tap': 11,
        #     'SELECT_ITEM minecraft:planks': 12,
        #     'SELECT_ITEM minecraft:stick': 13,
        #     'SELECT_ITEM minecraft:crafting_table': 14,
        #     'SELECT_ITEM polycraft:sack_polyisoprene_pellets': 15,
        #     'SELECT_ITEM minecraft:log': 16,
        # }

        self.observation_space = None
        self.action_space = None
        self.accumulated_step_cost = 0
        self.last_step_cost = 0
        self.novel_items = []
        self.all_items = []
        self.first_space_init = False
        self.mdp_items_id = {}
        self.num_types = 0
        if not use_trade:
            self.generate_obs_action_spaces()
            self.first_space_init = False

    def set_items_id(self, items):
        # items_id = {}
        self.all_items = [None for i in range(len(items))]
        if 'minecraft:air' in items:
            #This should always be 0
            self.mdp_items_id['minecraft:air'] = 0
            # items_id['minecraft:air'] = 0
            # self.all_items.append('minecraft:air')
            self.all_items[0] = 'minecraft:air'
        # for item in sorted(items):
        for item in items:
            if item != 'minecraft:air':
                if 'minecraft:air' in items:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id))
                    # items_id[item] = len(items_id)
                else:
                    self.mdp_items_id.setdefault(item, len(self.mdp_items_id)+1)
                    # items_id[item] = len(items_id) + 1
                # self.all_items.append(item)
                self.all_items[self.mdp_items_id[item]] = item

        # return items_id

    def generate_obs_action_spaces(self, new_items=[]):
        # #MDP items_id must be same as env items_id, but we want to include items that the env has not seen before
        # #Since items_id is updated as we go on, copying items_id should actually always be fine as is
        # self.all_items = []
        # self.mdp_items_id = self.items_id.copy()
        # for item in new_items:
        #     if item not in self.mdp_items_id:
        #         self.mdp_items_id[item] = len(items_id)
        # #Notion used to prioritize reset states
        # self.novel_items = []
        # for item in self.mdp_items_id:
        #     self.all_items.append(item)
        #     if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
        #                     'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
        #                     'polycraft:sack_polyisoprene_pellets']:
        #         self.novel_items.append(item)

        # self.novel_items = []
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
            if item not in ['minecraft:air', 'minecraft:bedrock', 'minecraft:crafting_table', 'minecraft:log',
                            'minecraft:planks', 'minecraft:stick', 'polycraft:tree_tap', 'polycraft:wooden_pogo_stick',
                            'polycraft:sack_polyisoprene_pellets']:
                novel_items.append(item)
                if self.first_space_init:
                    if item not in self.novel_items:
                        print('WARNNING - Novel item {} has been discovered since last MDP action/obs space init, observations prior to and after this point will be mismatched'.format(item))
                # self.novel_items.append(item)

        self.novel_items = novel_items

        # self.mdp_items_id = self.set_items_id(item_list)
        self.set_items_id(item_list)

        #Need items_id to be aligned with MDP items id
        #TODO: ask Gyan if I can change this
        self.items_id = self.mdp_items_id.copy()
        # print('mdp items', self.mdp_items_id)
        # print('items', self.items_id)
        # print(self.all_items)

        self.mdp_inventory_items = list(self.mdp_items_id.keys())
        if 'minecraft:air' in self.mdp_inventory_items:
            self.mdp_inventory_items.remove('minecraft:air')
        if 'minecraft:bedrock' in self.mdp_inventory_items:
            self.mdp_inventory_items.remove('minecraft:bedrock')
        #remove pogostick?

        #Need some notion of entities in the obs space
        #I believe adding a new type for each id to differentiate between block and entity
        #  and extending relcoords to each of those is too much
        #Entities are also weird to put in the map because they can pop out on top of blocks
        #Solution - just add notion of how many entities of each id are in the world? So the agent
        #   can know when it caused an entity to pop up or pick one up. Downside is don't know how close
        #   to one we are, but picking them up should be handled in reset_to_interesting_state
        # self.mdp_entity_items = list(self.mdp_items_id.keys())
        # if 'minecraft:air' in self.mdp_inventory_items:
        #     self.mdp_inventory_items.remove('minecraft:air')
        # if 'minecraft:bedrock' in self.mdp_inventory_items:
        #     self.mdp_inventory_items.remove('minecraft:bedrock')
        #TODO: Best solution:
        # Only add notion of entity count if we've seen it post-novelty (+ blocks we know it's possible)
        # Extend action space to include pick_up actions for entities?
        #   move to and pick up entity if in map - *But what if can't motion plan? Just as confusing for the agent
        #   otherwise give greater neg reward


        # print(self.items_id)
        # print(self.recipes.keys())
        # print(self.inventory_items)

        # Generate all actions from current state of env and set action space
        # TODO: make sure at this point whatever novel object is present in env to be included
        self.manip_actions =  ['MOVE w',
                               'TURN -90',
                               'TURN 90',
                               'BREAK_BLOCK',
                               'PLACE_TREE_TAP',
                               'EXTRACT_RUBBER']

        # Add place_crafting_table to action list -> we can break it but not put it back currently
        self.manip_actions.append('PLACE_CRAFTING_TABLE')

        # Should crafting table be a recipe?
        self.crafting_actions = ['CRAFT ' + item for item in self.recipes.keys()]

        # REMOVE CRAFT crafting_table - don't think this is present in tournament, but is in API
        if 'CRAFT minecraft:crafting_table' in self.crafting_actions:
            self.crafting_actions.remove('CRAFT minecraft:crafting_table')

        # self.select_actions =  ['SELECT_ITEM polycraft:wooden_pogo_stick',
        #                         'SELECT_ITEM polycraft:tree_tap',
        #                         'SELECT_ITEM minecraft:planks',
        #                         'SELECT_ITEM minecraft:stick',
        #                         'SELECT_ITEM minecraft:crafting_table',
        #                         'SELECT_ITEM polycraft:sack_polyisoprene_pellets',
        #                         'SELECT_ITEM minecraft:log']
        self.select_actions = ['SELECT_ITEM ' + item for item in self.mdp_inventory_items]

        # TODO: planner has deselect action, but I don't see how you can deselect an item through API
        #   The initial selected_item is '', but there's no command I see to select nothing
        # And USE_HAND is different than break without object
        # print(self.execute_action('SELECT_ITEM '))
        # input('wait')

        # For testing purposes to assert that everything is working
        # self.manip_actions =  ['MOVE w',
        #                        'TURN -90',
        #                        'TURN 90']
        # self.crafting_actions = []
        # self.select_actions = []

        self.all_actions = self.manip_actions + self.crafting_actions + self.select_actions
        self.actions_id = {}
        for i in range(len(self.all_actions)):
            self.actions_id[self.all_actions[i]] = i
        # print(self.actions_id)
        self.action_space = spaces.Discrete(len(self.actions_id))

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
        for item in ['minecraft:air', 'minecraft:bedrock', 'minecraft:planks', 'minecraft:stick', 'polycraft:sack_polyisoprene_pellets','polycraft:wooden_pogo_stick']:
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

        # print(self.get_nearest_items())

        #No notion of selected item anywhere? Assuming we are holding nothing to start
        #       NOT ALWAYS GOING TO BE TRUE
        # self.selected_item = 'minecraft:air'
        # self.selected_item_id = 0
        # self.selected_item_id = self.items_id[self.selected_item]

        # Need to update map using updated items_id set
        sense_all_command_result = self.run_SENSE_ALL_and_update('NONAV')
        self.accumulated_step_cost += sense_all_command_result['stepCost']

        self.num_types = len(self.mdp_items_id)+1

        print('updated items, recipes, and actions in MDP')
        print('Items: ', self.mdp_items_id)
        print('Craft items: ', self.recipes.keys())
        print('Actions: ', self.actions_id)
        self.first_space_init = True



    def step(self, action_id):
        # action_id = int(input('action'))
        action = self.all_actions[action_id]

        # Need to map to action string?
        action_result, sense_result = self.execute_action(action)

        self.accumulated_step_cost += action_result['stepCost'] + sense_result['stepCost']
        self.last_step_cost = action_result['stepCost'] + sense_result['stepCost']
        obs = self.observation()
        info = self.get_info()
        # info['last_step_cost'] = action_result['stepCost'] + sense_all_result['stepCost']

        return obs, None, None, info

    def get_info(self):

        # info = {'bin_map': self.binary_map, 'map_limits': (self.x_max, self.y_max), \
        info = {'items_locs': self.items_location, \
                'entities_locs': self.entities_location, \
                'block_in_front': self.block_in_front, \
                'inv_quant_dict': self.inventory_quantity_dict, \
                # 'inv_quant_list': self.inventory_list, \
                # 'ingred_quant_dict': self.ingredients_quantity_dict, \
                'player': self.player,  # 'local_view_size': self.local_view_size, \
                'selected_item': self.selected_item,
                'total_step_cost': self.accumulated_step_cost,
                'last_step_cost': self.last_step_cost}

        return info

    def reset(self, task=None):
        if task is None:
            task = self.task
        self.accumulated_step_cost = 0
        a_command_result, sense_all_command_result = self.run_a_command_and_update_map(task, sleep_time=5)
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
        # TODO: No notion of selected item anywhere?
        state = np.concatenate((local_view.flatten(), [facing_id], nearest_items.flatten(), [self.selected_item_id], inventory))
        # print(state)

        # state.astype(np.float).ravel()

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

        # print(local_view)

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
        # print(self.inventory_list)
        inv = []
        for item in self.mdp_inventory_items:
            if item in self.inventory_quantity_dict:
                inv.append(self.inventory_quantity_dict[item])
            else:
                inv.append(0)
        return inv

    #
    # def action_valid(self, action):
    #     # print(action)
    #     # print ("INVENTORY_QUANTITY_DICT = {}".format(self.inventory_quantity_dict))
    #     # print ("BLOCK_IN_FRONT = {}".format(self.block_in_front['name']))
    #     # print ("RECIPES = {}".format(self.recipes))
    #
    #     # this boolean function returns if the current action is valid or not
    #     # print ("Inventory dict = {}".format(self.inventory_quantity_dict))
    #     if action in self.generic_actions:
    #         return self.check_for_further_valid_movement(action)
    #         # return True
    #
    #     elif action == 'BREAK_BLOCK':
    #         if self.block_in_front['name'] == 'minecraft:log':
    #             return True
    #         else:
    #             return False
    #
    #     elif action == 'minecraft:planks':
    #         # print ("INVENTORY_QUANTITY_DICT = {}".format(self.inventory_quantity_dict))
    #         if 'minecraft:log' in self.inventory_quantity_dict.keys() and \
    #                 self.inventory_quantity_dict['minecraft:log'] >= 1:
    #             return True
    #         else:
    #             return False
    #
    #     elif action == 'minecraft:stick':
    #         if 'minecraft:planks' in self.inventory_quantity_dict.keys() and self.inventory_quantity_dict[
    #             'minecraft:planks'] >= 2:
    #             return True
    #         else:
    #             return False
    #
    #     elif action == 'PLACE_TREE_TAP' or action == 'EXTRACT_RUBBER':
    #         # search the location of trees and check if
    #         # the agent is two blocks away and
    #         # facing correctly
    #         return self.check_for_further_validity()
    #
    #     elif action == 'polycraft:tree_tap':
    #         if 'minecraft:planks' in self.inventory_quantity_dict.keys() and 'minecraft:stick' in self.inventory_quantity_dict.keys() and \
    #                 self.inventory_quantity_dict['minecraft:planks'] >= 5 and self.inventory_quantity_dict[
    #             'minecraft:stick'] >= 1 and \
    #                 self.block_in_front['name'] == 'minecraft:crafting_table':
    #             return True
    #         else:
    #             return False
    #     # POGO_STICK can only be crafted if the agent has bag and
    #     # is in front of the CRAFTING_TABLE
    #     elif action == 'polycraft:wooden_pogo_stick':
    #         # print ("When action taken in wooden pogo stick")
    #         # print ("INV QUANT DICTIONARY_Keys = {}".format(self.inventory_quantity_dict.keys()))
    #         # print ("bags = {}".format(self.inventory_quantity_dict['polycraft:sack_polyisoprene_pellets']))
    #         # print ("sticks = {}".format(self.inventory_quantity_dict['minecraft:stick']))
    #         # print ("Planks = {}".format(self.inventory_quantity_dict['minecraft:planks']))
    #         # print ("Block in front = {}".format(self.block_in_front['name']))
    #         if 'polycraft:sack_polyisoprene_pellets' in self.inventory_quantity_dict.keys() and 'minecraft:stick' in self.inventory_quantity_dict.keys() and 'minecraft:planks' in self.inventory_quantity_dict.keys() and \
    #                 self.inventory_quantity_dict['polycraft:sack_polyisoprene_pellets'] >= 1 and \
    #                 self.inventory_quantity_dict['minecraft:stick'] >= 4 and self.inventory_quantity_dict[
    #             'minecraft:planks'] >= 2 and \
    #                 self.block_in_front['name'] == 'minecraft:crafting_table':
    #             return True
    #         else:
    #             return False
    #     # in all other cases there is an error
    #     else:
    #         # print (Fore.GREEN + "ACTION = {}".format(action))
    #         # print(Fore.RED + " ACTION : {} ERROR: INVALID ACTION ENTERED".format(action))
    #         print(" ACTION : {} ERROR: INVALID ACTION ENTERED".format(action))
    #
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
            if self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y-1][x] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x-1] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x+1] != self.mdp_items_id['minecraft:air'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:bedrock'] and self.map_to_plot[y+1][x] != self.mdp_items_id['minecraft:log']:
                return True
            else:
                return
        else:
            if self.map_to_plot[y+1][x] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y-1][x] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x-1] == self.mdp_items_id['minecraft:log']:
                return True
            elif self.map_to_plot[y][x+1] == self.mdp_items_id['minecraft:log']:
                return True
            else:
                return False

        # flag = False
        #
        #
        # for i in range(len(self.locations_logs)):
        #     loc = list(map(int, str.split(self.locations_logs[i], ',')))
        #     # print(loc)
        #     # if facing any direction there are three cases:
        #     # A: two blocks away B: 2 diagonal blocks away cases
        #     if self.player['facing'] == 'NORTH':
        #         if self.player['pos'][0] == loc[0] and \
        #                 self.player['pos'][2] == loc[2] + 2:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] - 1 \
        #                 and self.player['pos'][2] == loc[2] + 1:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] + 1 and \
        #                 self.player['pos'][2] == loc[2] + 1:
        #             flag = True
        #             break
        #     elif self.player['facing'] == 'SOUTH':
        #         if self.player['pos'][0] == loc[0] and \
        #                 self.player['pos'][2] == loc[2] - 2:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] - 1 and \
        #                 self.player['pos'][2] == loc[2] - 1:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] + 1 and \
        #                 self.player['pos'][2] == loc[2] - 1:
        #             flag = True
        #             break
        #     elif self.player['facing'] == 'EAST':
        #         if self.player['pos'][0] == loc[0] - 2 and \
        #                 self.player['pos'][2] == loc[2]:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] - 1 and \
        #                 self.player['pos'][2] == loc[2] - 1:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] - 1 and \
        #                 self.player['pos'][2] == loc[2] + 1:
        #             flag = True
        #             break
        #     elif self.player['facing'] == 'WEST':
        #         if self.player['pos'][0] == loc[0] + 2 and \
        #                 self.player['pos'][2] == loc[2]:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] + 1 and \
        #                 self.player['pos'][2] == loc[2] - 1:
        #             flag = True
        #             break
        #         elif self.player['pos'][0] == loc[0] + 1 and \
        #                 self.player['pos'][2] == loc[2] + 1:
        #             flag = True
        #             break
        #     else:
        #         flag = False
        #     # print (Fore.BLUE + "flag = {}".format(flag))
        # return flag
    #
    # def check_for_further_valid_movement(self, action):
    #     if action in self.looking_actions:
    #         return True
    #     else:
    #         position = [self.player['pos'][0], self.player['pos'][2]]
    #         # make the dictionary of all the air values:
    #         # check whether the movement is a valid movement or not
    #         # self.items_location['minecraft:air']
    #         # print ("MNECRAFT:AIR = {}".format(self.items_location['minecraft:air']))
    #         if action == 'MOVE_NORTH':
    #             coordinates = str(position[0]) + ',' + str(4) + ',' + str(position[1] - 1)
    #             # print ("coordinates = {}".format(coordinates))
    #             if coordinates in self.items_location['minecraft:air']:
    #                 return True
    #             else:
    #                 return False
    #
    #         elif action == 'MOVE_SOUTH':
    #             coordinates = str(position[0]) + ',' + str(4) + ',' + str(position[1] + 1)
    #             # print ("coordinates = {}".format(coordinates))
    #             if coordinates in self.items_location['minecraft:air']:
    #                 return True
    #             else:
    #                 return False
    #
    #         elif action == 'MOVE_EAST':
    #             coordinates = str(position[0] + 1) + ',' + str(4) + ',' + str(position[1])
    #             # print ("coordinates = {}".format(coordinates))
    #             if coordinates in self.items_location['minecraft:air']:
    #                 return True
    #             else:
    #                 return False
    #
    #         elif action == 'MOVE_WEST':
    #             coordinates = str(position[0] - 1) + ',' + str(4) + ',' + str(position[1])
    #             # print ("coordinates = {}".format(coordinates))
    #             if coordinates in self.items_location['minecraft:air']:
    #                 return True
    #             else:
    #                 return False
    #
    # # def calculate_reward(self, action, action_output):
    #
    # #     '''
    # #     returns the reward on the current state
    # #     '''
    # #     # +1000 for crafting POGO wooden_pogo_stick
    # #     if action == 'polycraft:wooden_pogo_stick' and action_output == 'SUCCESS':
    # #         reward = +1000
    # #     # +25 for CRAFTING CRAFT_STICKS, CRAFT_PLANKS and CRAFT_TREE_TAP
    # #     elif action in self.crafting_actions and action_output == 'SUCCESS':
    # #         reward = +25
    # #     # +10 for using BREAK_BLOCK PLACE_TREE_TAP and EXTRACT_RUBBER correctly
    # #     elif action in self.meta_actions and action_output == 'SUCCESS':
    # #         reward = +10
    # #     # -1 for every step
    # #     else:
    # #         # +5 for reaching near TREE and crafting_table
    # #         if self.block_in_front['name'] in self.important_locations:
    # #             reward = +5
    # #         # a reward of -1 everywhere else
    # #         else:
    # #             reward = -1
    #
    # #     return reward
    #
    # def calculate_reward(self, action, action_output):
    #
    #     '''
    #     returns the reward on the current state
    #     '''
    #     # +1000 for crafting POGO wooden_pogo_stick
    #     # if action == 'polycraft:wooden_pogo_stick' and action_output == 'SUCCESS':
    #     #     reward = +1000
    #     # # +25 for CRAFTING CRAFT_STICKS, CRAFT_PLANKS and CRAFT_TREE_TAP
    #     # elif action in self.crafting_actions and action_output == 'SUCCESS':
    #     #     reward = +25
    #     # # +10 for using BREAK_BLOCK PLACE_TREE_TAP and EXTRACT_RUBBER correctly
    #     # elif action in self.meta_actions and action_output == 'SUCCESS':
    #     #     reward = +10
    #     # # -1 for every step
    #     # else:
    #     # +5 for reaching near TREE and crafting_table
    #     if self.block_in_front['name'] in self.important_locations:
    #         reward = +100
    #     # a reward of -1 everywhere else
    #     else:
    #         reward = -1
    #
    #     return reward
    #
    #     # def calculate_reward(self, action, action_output):
    #
    # #     # this is an incremenental reward
    # #     # give a small positive reward if the agent successfully completes a navigation TASK
    # #     if self.block_in_front['name'] in self.important_locations:
    # #         reward = +1
    #
    # #     # give a +10 reward if the agent successfully vcomplete the breakblock Action
    # #     elif action in self.meta_actions and action_output == 'SUCCESS':
    # #         # if the breakblock itself also result in the log grabbing:
    # #         if self.inventory_list_dict['minecraft:log'] > self.inventory_list_dict_old['minecraft:log']:
    # #             reward = +200
    # #         else:
    # #             reward = +10
    # #       # cannot grab the info
    #
    # #     # if the inventory list grew and added one log
    # #     elif self.inventory_list_dict['minecraft:log'] > self.inventory_list_dict_old['minecraft:log']:
    # #         reward = +200
    #
    # #     # give a negative reward for everything else
    # #     else:
    # #         reward = -1
    #
    # #     return reward
    #
    # def is_terminal(self, action, action_output):
    #     '''
    #     boolean function returns if the state is a terminal state or not
    #     '''
    #
    #     # terminal state is reached if the wooden pogo stick is successfuly made.
    #     # if action == 'polycraft:wooden_pogo_stick' and action_output == 'SUCCESS':
    #     #     return True
    #
    #     # elif action == 'BREAK_BLOCK':
    #     #     return True
    #
    #     # print ("Block in Front = {}".format(self.block_in_front))
    #
    #     if self.block_in_front['name'] in self.important_locations:
    #         return True
    #
    #     else:
    #         return False
    #
    # # def is_terminal(self, action, action_output):
    # #     '''
    # #     boolean function returns if the state is a terminal state or not
    # #     '''
    #
    # #     # terminal state is reached if the wooden pogo stick is successfuly made.
    # #     # if action == 'polycraft:wooden_pogo_stick' and action_output == 'SUCCESS':
    # #     #     return True
    #
    # #     # elif action == 'BREAK_BLOCK':
    # #     #     return True
    #
    # #     # if self.block_in_front['name'] in self.important_locations:
    # #     #     return True
    #
    # #     # else:
    # #     #     return False
    #
    # #     # if the action is break
    # #     if action in self.meta_actions and action_output == 'SUCCESS' and self.inventory_list_dict['minecraft:log'] > self.inventory_list_dict_old['minecraft:log']:
    # #         return True
    # #     elif self.inventory_list_dict['minecraft:log'] > self.inventory_list_dict_old['minecraft:log']:
    # #         return True
    # #     else:
    # #         return False
    #

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
