# Adapted from env_v2.py

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







