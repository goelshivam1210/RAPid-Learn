import numpy as np
from collections import defaultdict
import time
import os 

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

ACTION_REMAPPING = {'Break': 'break\n',\
'Craft_tree_tap':'crafttree_tap\n',\
'Craft_plank': 'craftplank\n' , \
'Craft_stick': 'craftstick\n', \
'Extract_rubber': 'extractrubber\n',\
'Craft_pogo_stick':'craftpogo_stick\n'}

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


# Make a dictionary to store all the action pre conditions and effects Domain_Mmap
# Create two requirements (list), one for  inventory (1) and one for world (2)
# Take the plan -> Find the goal of the plan
# Check the proconditions of the plan
# Add any inventory requirements to 1 and facing requirements to 2
# For each item in 1, find the action (from the plan) that generates the item
# For each such action, find the preconditions
# Do this recursively until you reach the broken action
# Find the world and inventory requirements, return the success function for the failed operator

#We now have a list that gives us what all items need to be generated when one action fails
#Write a success func, that takes in the info and checks if it has generated all the items required
#If yes, return true
#Modify the existing success func to include even this success func in OR condition
#Write a novelty wrapper -> Tree unbreakable, new scrape action (yields 4 planks) and tree gone
class RewardFunctionGenerator:
    def __init__(self, plan=None, failed_action=None, domain_file="domain"):
        self.domain_file = domain_file
        self.precondition_map, self.effect_map = self.parse_domain(self.domain_file)
        self.precondition_objects_required = []
        self.objects_required = []
        self.generate_success_func(plan, failed_action)
        self.actions_that_generate_objects_required = self.actions_generating_objects_req()
        print("objects req are:", self.objects_required)
        time.sleep(5)
    # For each action in domain file, find preconditions and effects, save them

    def actions_generating_objects_req(self):
        actions_to_remove = []
        a = []
        reversed_action_mapping = {value : key for (key, value) in ACTION_REMAPPING.items()}
        for obj in self.objects_required:
            action_to_remove = list(self.effect_map.keys())[list(self.effect_map.values()).index([obj])]
            actions_to_remove.append(reversed_action_mapping[action_to_remove])
        return actions_to_remove            

    def parse_domain(self, filename):
        filename = "PDDL" + os.sep + filename + ".pddl"
        precondition_map = dict()
        effects_map = dict()
        with open(filename, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        for line_no, line_text in enumerate(all_lines):
            if ":action" in line_text:
                action_line = line_text.split(" ")
                precondition_map[action_line[1]] = []
                effects_map[action_line[1]] = []
                for action_line_no in range(line_no, len(all_lines)):
                    if ":precondition" in all_lines[action_line_no]:
                        i = 0
                        while True:
                            if "inventory" in all_lines[action_line_no + i]:
                                temp = all_lines[action_line_no + i].replace(")","(")
                                temp = temp.split("(") 
                                temp = temp[2].split(" ")
                                precondition_map[action_line[1]].append(temp[-1])
                                i+=1
                                continue 
                            elif "effect" in all_lines[action_line_no + i]:                                
                                break
                            else:
                                i+=1
                    elif ":effect" in all_lines[action_line_no]:
                        i = 0
                        while True:
                            if action_line_no + i == len(all_lines):
                                break
                            if "increase" in all_lines[action_line_no + i]:
                                temp = all_lines[action_line_no + i].replace(")","(")
                                temp = temp.split("(") 
                                temp = temp[2].split(" ")
                                if temp[-1] == 'air':
                                    i+=1
                                    continue
                                effects_map[action_line[1]].append(temp[-1])
                                i+=1
                                continue 
                            elif ":action" in all_lines[action_line_no + i]:                                
                                break
                            else:
                                i+=1
                        break
                
        return precondition_map, effects_map
        
    
    def return_precondition_list(self, action):
        preconditions = self.precondition_map[action]
        for precond in preconditions:
            if precond not in self.precondition_objects_required and precond not in self.failed_objects:
                self.precondition_objects_required.append(precond)
            if precond in self.failed_objects:
                self.objects_required.append(self.effect_map[action])


    def generate_success_func(self, plan, failed_action): #Break
    # def generate_success_func(self, plan, failed_action): #Break
        # failed_action = 'Break'
        # plan =  ['approach air tree_log', 'Break', 'approach air tree_log',\
        #  'Craft_plank', 'Craft_stick', 'Break', 'approach air tree_log', 'Craft_plank', \
        #      'approach tree_log crafting_table', 'Craft_tree_tap', \
        #      'approach crafting_table tree_log',\
        #           'Select_tree_tap', 'Extract_rubber', 'Break', 'approach air crafting_table',\
        #                'Craft_plank', 'Craft_stick', 'Craft_pogo_stick']
        failed_action = ACTION_REMAPPING[failed_action] # get the failed action
        self.failed_objects = self.effect_map[failed_action] # get the objects that could not be made
        goal_action = ACTION_REMAPPING[plan[-1]] # start from the goal to go recursively to the leaf node
        if "air" in self.failed_objects:
            self.failed_objects.remove('air') # failed objects is a list 

        self.return_precondition_list(goal_action)
        visited = [self.effect_map[goal_action][i] for i in range(len(self.effect_map[goal_action]))]

        for precond_obj in self.precondition_objects_required:
            if precond_obj in visited:
                continue
            print("effect map:", self.effect_map)
            action_that_yields_precond_obj = list(self.effect_map.keys())[list(self.effect_map.values()).index([precond_obj])]
            if action_that_yields_precond_obj is not failed_action:
                visited.append(self.effect_map[action_that_yields_precond_obj][i] for i in range(len(self.effect_map[action_that_yields_precond_obj])))
                self.return_precondition_list(action_that_yields_precond_obj)     

        self.objects_required = [self.objects_required[i][0] for i in range(len(self.objects_required))]
    
    def check_success(self, info):
        flag = []
        for object_req in self.objects_required:
            if object_req in info['inv_quant_dict'].keys():
                if info['inv_quant_dict'][object_req] > self.init_info['inv_quant_dict'][object_req]:
                    flag.append(True)
                else:
                    flag.append(False)
            else:
                return False
            return np.all(flag)

    def store_init_info(self, info):
        self.init_info = info

if __name__ == "__main__":
    reward = RewardFunctionGenerator()