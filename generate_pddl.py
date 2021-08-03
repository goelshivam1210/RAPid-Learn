import numpy as np
import os
import re
import sys
import csv
import math
import copy
import argparse
import subprocess

def generate_prob_pddl(pddl_dir,env):
    generate_problem_pddl(pddl_dir,env)

def generate_domain_pddl(pddl_dir, env, learned_operator):
    generate_domain(pddl_dir, env, learned_operator)

def generate_domain(pddl_dir, env, learned_operator):
    pass

def generate_problem_pddl(pddl_dir,env, filename: str = "problem"):

    filename = pddl_dir + os.sep + filename + ".pddl"

    hder = _generate_header_prob()
    objs = _generate_objects(env)
    goals = _generate_goals(env)
    ints = _generate_init(env)

    pddl = "\n".join([hder, objs, ints, goals, ")\n"])
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(pddl))
        print(f"Problem PDDL written to {filename}.")


def _generate_header_prob():
    return f"(define\n\t(problem adeGeneratedProblem)" + "\n\t" + f"(:domain adeGeneratedDomain)"

def _generate_objects(env):
    objects_list = []
    for item in env.items:
        objects_list.append("\t\t"+item+" - "+item)
    objs = "\n".join(objects_list)
    return "\n".join(["\t(:objects", objs, "\t)"])

def _generate_init(env):
    init_list = []

    wall_size = env.map_size*4-4

    air_size = env.map_size*env.map_size - wall_size - sum(env.items_quantity[item] for item in env.items_quantity)

    for item in env.items:
        init_list.append("\t\t(= (inventory "+str(item)+") "+str(env.inventory_items_quantity[item])+")")
        if item in env.items_quantity:
            init_list.append("\t\t(= (world "+str(item)+") "+str(env.items_quantity[item])+")")
        elif item == 'wall':
            init_list.append("\t\t(= (world wall) "+str(wall_size)+")")
        elif item == 'air':
            init_list.append("\t\t(= (world air) "+str(air_size)+")")
        else:
            init_list.append("\t\t(= (world "+str(item)+") "+"0"+")")

    if env.selected_item == '':
        init_list.append("\t\t(holding air)")
    else:
        init_list.append("\t\t(holding "+env.selected_item+")")

    init_list.append("\t\t(facing "+env.block_in_front_str+")")

    ints = "\n".join(init_list)
    return "\n".join(["\t(:init", ints, "\t)"])

# list(my_dict.keys())[list(my_dict.values()).index(112)]

def _generate_goals(env):
    goal_list = []
    for item in env.goal_item_to_craft:
        goal_list.append(item)
    goal = "".join(goal_list)
    print("goal:", goal)
    # goal = goal_list
    return "".join(["(:goal (>= (inventory ",goal,") 1))"])
 
    