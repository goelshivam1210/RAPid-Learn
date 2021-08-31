import gym
import copy

class RewardShaping(gym.RewardWrapper):
    """Add intermediate rewards for reaching useful subgoals."""
    def __init__(self, env):
        super().__init__(env)
        self.last_inventory = copy.deepcopy(self.env.inventory_items_quantity)

    def reward(self, reward):
        reward = -1
        appropriate_next_action = self._determine_appropriate_next_action()

        # Approach tree
        if appropriate_next_action == 'approach_tree':
            if self.env.block_in_front_id == 6:
                reward += 50
        # Break tree
        elif appropriate_next_action == 'break_tree':
            if self.env.inventory_items_quantity['tree_log'] - self.self.last_inventory['tree_log'] > 0:
                reward += 50
        # Craft planks
        elif appropriate_next_action == 'craft_plank':
            if self.env.inventory_items_quantity['plank'] - self.self.last_inventory['plank'] > 0:
                reward += 50
        # craft sticks
        elif appropriate_next_action == 'craft_sticks':
            if self.env.inventory_items_quantity['stick'] - self.self.last_inventory['stick'] > 0:
                reward += 50
        # approach crafting table
        elif appropriate_next_action == 'approach_crafting_table':
            if self.env.block_in_front_id == 1:
                reward += 50
        # craft treetap
        elif appropriate_next_action == 'craft_treetap':
            if self.env.inventory_items_quantity['tree_tap'] - self.self.last_inventory['tree_tap'] > 0:
                reward += 50
        # extract rubber
        elif appropriate_next_action == 'extract_rubber':
            if self.env.inventory_items_quantity['rubber'] - self.self.last_inventory['rubber'] > 0:
                reward += 50
        # craft pogostick
        elif appropriate_next_action == 'craft_pogo_stick':
            if self.env.inventory_items_quantity['pogo_stick'] - self.self.last_inventory['pogo_stick'] > 0:
                reward += 50

        self.last_inventory = copy.deepcopy(self.env.inventory_items_quantity)
        return reward

    def _determine_appropriate_next_action(self):
        if sum([self.env.inventory_items_quantity['tree_tap'],
                self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and self.env.inventory_items_quantity[
            'tree_log'] == 0 and self.env.inventory_items_quantity['plank'] < 4 and self.env.inventory_items_quantity[
            'stick'] == 0:
            if self.env.block_in_front_id != 6:
                return 'approach_tree'
            else:
                return 'break_tree'

        if sum([self.env.inventory_items_quantity['tree_tap'],
                self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and \
                self.env.inventory_items_quantity['tree_log'] >= 1 and \
                self.env.inventory_items_quantity['plank'] < 4:
            return 'craft_plank'

        if sum([self.env.inventory_items_quantity['stick'], self.env.inventory_items_quantity['tree_tap'],
                self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and \
                self.env.inventory_items_quantity['plank'] >= 4 and self.env.inventory_items_quantity['stick'] == 0:
            return 'craft_sticks'

        if sum([self.env.inventory_items_quantity['tree_tap'],
                self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and \
                self.env.inventory_items_quantity['stick'] >= 1 and self.env.inventory_items_quantity['plank'] >= 4:
            if self.env.block_in_front_id != 1:
                return 'approach_crafting_table'
            else:
                return 'craft_treetap'

        if sum([self.env.inventory_items_quantity['rubber'],
                self.env.inventory_items_quantity['pogo_stick']]) < 1 and \
                self.env.inventory_items_quantity['tree_tap'] >= 1:
            if self.env.block_in_front_id != 6:
                return 'approach_tree'
            else:
                return 'extract_rubber'

        if self.env.inventory_items_quantity['pogo_stick'] < 1 and self.env.inventory_items_quantity['rubber'] >= 1:
            if self.env.inventory_items_quantity['stick'] < 4:
                return 'craft_stick'
            if self.env.inventory_items_quantity['plank'] < 2:
                return 'craft_plank'

            if self.env.block_in_front_id != 1:
                return 'approach_crafting_table'
            else:
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
