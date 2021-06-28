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
        print('Sending command: {}'.format(command))
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
