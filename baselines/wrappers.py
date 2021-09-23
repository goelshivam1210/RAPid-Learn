import gym
import copy
import numpy as np
from gym import spaces


class RewardShaping(gym.RewardWrapper):
    """Add intermediate rewards for reaching useful subgoals."""
    def __init__(self, env):
        super().__init__(env)
        print("Reward shaping: ON")
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
        if self.env.inventory_items_quantity['pogo_stick'] < 1:
            if self.env.inventory_items_quantity['rubber'] < 1:
                if self.env.inventory_items_quantity['tree_tap'] < 1:
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
                                return 'break_tree'
                            else:
                                return 'craft_plank'
                    else:
                        return 'craft_treetap'
                else:
                    return 'extract_rubber'
            else:
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
        else:
            return None

class EpisodicWrapper(gym.Wrapper):
    LOG_INTERVAL = 200

    """Terminate the episode and reset the environment after a number of timesteps has passed"""
    def __init__(self, env, timesteps_per_episode, verbose=False):
        super().__init__(env)
        self.env = env
        self.verbose = verbose
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
        if self.verbose and self.episode % EpisodicWrapper.LOG_INTERVAL == 0:
            print(f"Starting episode #{self.episode}")
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
    def __init__(self, env, n_placeholders_inventory=0, n_placeholders_lidar=0, env_has_fire_flag=False):
        super(StatePlaceholderWrapper, self).__init__(env)
        print(f"Wrapping env with StatePlaceholderWrapper (inv: {n_placeholders_inventory}, lidar: {n_placeholders_lidar})")
        self.n_placeholders_lidar = n_placeholders_lidar
        self.n_placeholders_inventory = n_placeholders_inventory
        self.env_has_fire_flag = env_has_fire_flag

        # The last position is selected_item id
        n_lidar_obs = (len(self.env.items_lidar) + n_placeholders_lidar) * self.env.num_beams
        n_inventory_obs = len(self.env.inventory_items_quantity) + n_placeholders_inventory

        # [lidar_observations, inventory_observations, selected_observations, fire_flag]
        low = [0] * n_lidar_obs + [0] * n_inventory_obs + [0] + [0]
        high = [self.env.max_beam_range] * n_lidar_obs + \
               [len(self.env.inventory_items_quantity) + self.n_placeholders_inventory] * n_inventory_obs + \
               [len(self.env.inventory_items_quantity) + self.n_placeholders_inventory] + \
               [2]

        self.observation_space = spaces.Box(np.array(low), np.array(high), dtype=int)
        self.env.observation_space = self.observation_space

    def observation(self, observation):
        inventory_observation_end_index = -1 if not self.env_has_fire_flag else -2

        lidar_observation = observation[0:len(self.env.items_lidar) * self.env.num_beams]
        inventory_observation = observation[len(self.env.items_lidar) * self.env.num_beams:inventory_observation_end_index]

        # the last element of the obs is the selected item, unless for the novelty envs, where the last element is
        # the fire flag and the selected item is the second-to-last item instead.
        if self.env_has_fire_flag:
            tail_observation = [observation[-2], observation[-1]]
        else:
            tail_observation = [observation[-1], 0]

        # insert the placeholders after each beam signal
        lidar_obs_with_placeholders = []
        beam_counter = 0
        for x in lidar_observation:
            beam_counter += 1
            lidar_obs_with_placeholders.append(x)
            if beam_counter == len(self.env.items_lidar):
                for i in range(self.n_placeholders_lidar):
                    lidar_obs_with_placeholders.append(0)
                beam_counter = 0

        placeholders_inventory = np.zeros(self.n_placeholders_inventory, dtype=int)
        return np.hstack([lidar_obs_with_placeholders, inventory_observation, placeholders_inventory,
                          tail_observation])

class ActionPlaceholderWrapper(gym.ActionWrapper):
    def __init__(self, env, n_placeholders_actions=0):
        super(ActionPlaceholderWrapper, self).__init__(env)
        print(f"Wrapping env with ActionPlaceholderWrapper (actions: {n_placeholders_actions})")
        self.n_placeholders_actions = n_placeholders_actions
        self.action_space = spaces.Discrete(len(self.env.actions_id) + self.n_placeholders_actions)
        self.env.action_space = self.action_space

    def action(self, action):
        # need to forbid trying to take the phantom actions here
        return min(len(self.env.actions_id) - 1, action)

    def reverse_action(self, action):
        return min(len(self.env.actions_id) - 1, action)