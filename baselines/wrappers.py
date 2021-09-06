import gym
import copy
import numpy as np
from gym import spaces


class RewardShaping(gym.RewardWrapper):
    """Add intermediate rewards for reaching useful subgoals."""
    def __init__(self, env):
        super().__init__(env)
        self.last_inventory = copy.deepcopy(self.env.inventory_items_quantity)
        self.appropriate_next_action = self._determine_appropriate_next_action()

    def reward(self, reward):
        reward = -1

        # Break tree
        if self.appropriate_next_action == 'break_tree':
            if self.env.inventory_items_quantity['tree_log'] - self.last_inventory['tree_log'] > 0:
                reward += 50
        # Craft planks
        elif self.appropriate_next_action == 'craft_plank':
            if self.env.inventory_items_quantity['plank'] - self.last_inventory['plank'] > 0:
                reward += 50
        # craft sticks
        elif self.appropriate_next_action == 'craft_sticks':
            if self.env.inventory_items_quantity['stick'] - self.last_inventory['stick'] > 0:
                reward += 50
        # craft treetap
        elif self.appropriate_next_action == 'craft_treetap':
            if self.env.inventory_items_quantity['tree_tap'] - self.last_inventory['tree_tap'] > 0:
                reward += 200
        # extract rubber
        elif self.appropriate_next_action == 'extract_rubber':
            if self.env.inventory_items_quantity['rubber'] - self.last_inventory['rubber'] > 0:
                reward += 300
        # craft pogostick
        elif self.appropriate_next_action == 'craft_pogo_stick':
            if self.env.inventory_items_quantity['pogo_stick'] - self.last_inventory['pogo_stick'] > 0:
                reward += 1000

        # don't give out more reward if the env gets unsolvable (all trees are cut down)
        if self.env.items_quantity['tree_log'] == 0:
            if self.env.inventory_items_quantity['pogo_stick'] == 0:
                if self.env.inventory_items_quantity['rubber'] == 0 or (
                        self.env.inventory_items_quantity['plank'] < 2 or self.env.inventory_items_quantity[
                    'stick'] < 4):
                    reward = -1

        self.last_inventory = copy.deepcopy(self.env.inventory_items_quantity)
        self.appropriate_next_action = self._determine_appropriate_next_action()
        return reward

    def _determine_appropriate_next_action(self):
        if sum([self.env.inventory_items_quantity['tree_tap'],
                self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1:
            if self.env.inventory_items_quantity['stick'] == 0 or self.env.inventory_items_quantity['plank'] < 4:
                if self.env.inventory_items_quantity['stick'] == 0:
                    if self.env.inventory_items_quantity['plank'] < 2:
                        if self.env.inventory_items_quantity['tree_log'] == 0:
                            return 'break_tree'
                        else:
                            return 'craft_plank'
                    else:
                        return 'craft_stick'
                if self.env.inventory_items_quantity['plank'] < 4:
                    if self.env.inventory_items_quantity['tree_log'] == 0:
                        if self.env.block_in_front_id != 6:
                            return 'break_tree'
                    else:
                        return 'craft_plank'
            else:
                return 'craft_treetap'

        if sum([self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and \
                self.env.inventory_items_quantity['tree_tap'] >= 1:
            return 'extract_rubber'
        if self.env.inventory_items_quantity['pogo_stick'] < 1 and self.env.inventory_items_quantity['rubber'] >= 1:
            if self.env.inventory_items_quantity['stick'] < 4:
                if self.env.inventory_items_quantity['plank'] < 2:
                    if self.env.inventory_items_quantity['tree_log'] == 0:
                        return 'break_tree'
                    else:
                        return 'craft_plank'
                else:
                    return 'craft_stick'
            if self.env.inventory_items_quantity['plank'] < 2:
                if self.env.inventory_items_quantity['tree_log'] == 0:
                    return 'break_tree'
                else:
                    return 'craft_plank'
            return 'craft_pogo_stick'

class EpisodicWrapper(gym.Wrapper):
    """Terminate the episode and reset the environment after a number of timesteps has passed"""
    def __init__(self, env, timesteps_per_episode):
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.episode = 0
        self.timesteps_per_episode = timesteps_per_episode

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.timestep += 1

        if self.timestep == self.timesteps_per_episode:
            done = True
        return next_state, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode += 1
        self.timestep = 0
        return obs

class InfoExtenderWrapper(gym.Wrapper):
    """Appends several useful things to the info dict."""
    def __init__(self, env, mode='train'):
        super().__init__(env)
        self.env = env
        self.env.metadata['mode'] = mode

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        info['mode'] = self.env.metadata['mode']
        info['success'] = str(self.env.last_done)
        return next_state, reward, done, info

class StatePlaceholderWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_placeholders_inventory=0, n_placeholders_lidar=0):
        super(StatePlaceholderWrapper, self).__init__(env)
        self.n_placeholders_lidar = n_placeholders_lidar
        self.n_placeholders_inventory = n_placeholders_inventory

        # The last position is selected_item id
        n_lidar_obs = (len(self.env.items_lidar) + n_placeholders_lidar) * self.env.num_beams
        n_inventory_obs = len(self.env.inventory_items_quantity) + n_placeholders_inventory

        low = np.array([0] * n_lidar_obs + [0] * n_inventory_obs + [0])
        high = np.array(
            [self.env.max_beam_range] * n_lidar_obs +
            [len(self.env.items)] * n_inventory_obs +  # inventory items
            [len(self.env.items) + self.n_placeholders_inventory])  # selected item

        self.observation_space = spaces.Box(low, high, dtype=int)
        self.env.observation_space = self.observation_space

    def observation(self, observation):
        lidar_observation = observation[0:len(self.env.items_lidar) * self.env.num_beams]
        inventory_observation = observation[
                                len(self.env.items_lidar) * self.env.num_beams:-1]
        selected_observation = [observation[-1]]

        placeholders_lidar = np.zeros(self.n_placeholders_lidar * self.env.num_beams)
        placeholders_inventory = np.zeros(self.n_placeholders_inventory)
        return np.hstack([lidar_observation, placeholders_lidar, inventory_observation, placeholders_inventory,
                          selected_observation])

class ActionPlaceholderWrapper(gym.ActionWrapper):
    def __init__(self, env, n_placeholders_actions=0):
        super(ActionPlaceholderWrapper, self).__init__(env)
        self.n_placeholders_actions = n_placeholders_actions
        self.action_space = spaces.Discrete(len(self.env.actions_id) + self.n_placeholders_actions)
        self.env.action_space = self.action_space

    def action(self, action):
        # need to forbid trying to take the phantom actions here
        return min(len(self.env.actions_id) - 1, action)

    def reverse_action(self, action):
        return min(len(self.env.actions_id) - 1, action)