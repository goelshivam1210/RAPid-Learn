# Gridworld env to match with updated RL polycraft interface and allow
#   for commands like SENSE_ALL etc without simulating socket connection
import copy
import math

import numpy as np
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
