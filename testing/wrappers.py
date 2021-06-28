from gym_novel_gridworlds.wrappers import LimitActions
import gym
from gym import error, spaces, utils
import numpy as np
import math

######## Wrapper used for experimenting with agents and state/action reps with the fence hard novelty ########

# Single wrapper for use with experimentation with fence novelty and various representations
# Keeping in one place for initially testing purposes, ideally refactor to make more modular
# Testing with goal of moveTo(Tree)
class FenceExperimentWrapper(gym.core.Wrapper):
    def __init__(self, env, methodid, limit_actions, obs_rep, agent_view_size=2, num_beams=8):
        super().__init__(env)

        # method_id - outlined below
        # limit_actions - specific (F,L,R,B) or full action space
        # obs_rep - only map + orientation or including inventory and selected_item

        # Goal is to moveTo tree
        self.goal_block_id = 6
        self.obs_rep = obs_rep

        if limit_actions:
            self.env = LimitActions(self.env, {'Forward':0, 'Left':1, 'Right':2, 'Break':3})

        assert methodid < 5, "No methodid > 5 supported"
        # Methodid: (all method include orientation)
        #   0 - Local Lidar
        #   1 - Oriented agent map + rel_coords to nearest obj type
        #   2 - Oriented agent map + local lidar
        #   3 - Penetrating Lidar
        #   4 - Penetrating Lidar + oriented agent map
        #  In all cases, obs includes inventory, orientation, and selected_item
        self.methodid = methodid
        self.num_item_types = 10
        self.max_items = 20
        self.num_beams = num_beams
        self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotenuse of a square
        self.agent_view_size = agent_view_size
        items_to_exclude = ['air', 'pogo_stick', 'plank', 'stick', 'rubber', 'tree_tap']
        self.lidar_items = set(self.items_id.keys())
        set(map(self.lidar_items.remove, items_to_exclude))
        self.lidar_items_id = self.set_items_id(self.lidar_items)  # set IDs for all the lidar items
        self.num_lidar_items = len(self.lidar_items)

        # Abstract away item types that are irrelevant to the task
        # if self.map_rep == 0:
        #     # all diff types
        #     self.num_item_types = 10
        # elif self.map_rep == 1:
        #     # empty, obstacle, fence, goal
        #     self.num_item_types = 4

        # Lidar
        self.num_beams = num_beams
        self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotenuse of a square
        agent_map_size = (agent_view_size * 2 + 1) ** 2
        # set map size to 30 in original env
        map_size = self.env.map_size
        low_lidar = np.array([0] * (self.num_lidar_items * self.num_beams))
        high_lidar = np.array([self.max_beam_range] * (self.num_lidar_items * self.num_beams))
        low_agent_map = np.zeros(agent_map_size)
        high_agent_map = self.num_item_types * np.ones(agent_map_size)
        low_orientation = np.array([0])
        high_orientation = np.array([4])
        low_rel_coords = np.array([-map_size, -map_size])
        high_rel_coords = np.array([map_size, map_size])

        # Just Lidar
        if self.methodid == 0 or self.methodid == 3:
            low_map = np.concatenate((low_lidar,low_orientation))
            high_map = np.concatenate((high_lidar,high_orientation))
        # agent map + relCoords
        elif self.methodid == 1:
            low_map = np.concatenate((low_agent_map, low_orientation, low_rel_coords))
            high_map = np.concatenate((high_agent_map, high_orientation, high_rel_coords))
        # agent map + lidar
        elif self.methodid == 2 or self.methodid == 4:
            low_map = np.concatenate((low_lidar, low_agent_map, low_orientation))
            high_map = np.concatenate((high_lidar, high_agent_map, high_orientation))

        if self.obs_rep == 0:
            low = low_map
            high = high_map
        # add selected item and inventory
        elif self.obs_rep == 1:
            self.relevant_inv_items = ['crafting_table', 'tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick']
            low = np.concatenate((low_map,[0],np.zeros(len(self.relevant_inv_items))))
            high = np.concatenate((high_map, [self.num_item_types], self.max_items*np.ones(len(self.relevant_inv_items))))
        else:
            print("Error, obs_rep must be 1 or 0")
            quit()

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.env.action_space

    def get_lidarSignal(self, agent_loc = None, agent_facing_id = None, envmap=None):
        """
        Send several beams (num_beams) at equally spaced angles in 360 degrees in front of agent within a range
        For each bean store distance (beam_range) for each item in lidar_items_id if item is found otherwise 0
        and return lidar_signals
        """

        if agent_facing_id is None:
            agent_facing_str = self.agent_facing_str
        elif agent_facing_id == 0:
            agent_facing_str = 'NORTH'
        elif agent_facing_id == 1:
            agent_facing_str = 'SOUTH'
        elif agent_facing_id == 2:
            agent_facing_str = 'WEST'
        elif agent_facing_id == 3:
            agent_facing_str = 'EAST'
        if envmap is None:
            envmap = self.env.map
        if agent_loc is None:
            r, c = self.agent_location
        else:
            r , c = agent_loc

        direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}

        # Shoot beams in 360 degrees in front of agent
        angles_list = np.linspace(direction_radian[agent_facing_str] - np.pi,
                                  direction_radian[agent_facing_str] + np.pi,
                                  self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360

        # angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi,
        #                           direction_radian[self.agent_facing_str] + np.pi,
        #                           self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360

        lidar_signals = []
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)
            beam_signal = np.zeros(self.num_lidar_items, dtype=int)

            # Keep sending longer beams until hit an object or wall
            for beam_range in range(1, self.max_beam_range + 1):
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = envmap[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
                    if item in self.lidar_items_id:
                        obj_id_rc = self.lidar_items_id[item]
                        if beam_signal[obj_id_rc - 1] == 0:
                            beam_signal[obj_id_rc - 1] = beam_range
                    if self.methodid == 0 or self.methodid == 2:
                        break
                    # only break on wall, return readings for everything else up until that point
                    elif self.methodid == 3 or self.methodid == 4:
                        if item == 'wall':
                            break
            lidar_signals.extend(beam_signal)

        return lidar_signals

    def get_agentView(self, agent_loc = None, agent_facing_id = None, envmap=None):
        """
        Slice map with 0 padding based on agent_view_size

        :return: local view of the agent
        """

        if agent_facing_id is None:
            agent_facing_id = self.env.agent_facing_id
        if agent_loc is None:
            desired_center = self.env.agent_location
        else:
            desired_center = agent_loc
        if envmap is None:
            envmap = self.env.map

        extend = [self.agent_view_size, self.agent_view_size]  # row and column
        pad_value = 0

        extend = np.asarray(extend)
        map_ext_shp = self.env.map.shape + 2 * np.array(extend)
        map_ext = np.full(map_ext_shp, pad_value)
        insert_idx = [slice(i, -i) for i in extend]
        map_ext[tuple(insert_idx)] = envmap
        # map_ext[tuple(insert_idx)] = self.env.map
        # desired_center = self.env.agent_location

        region_idx = [slice(i, j) for i, j in zip(desired_center, extend * 2 + 1 + desired_center)]
        area = map_ext[tuple(region_idx)]

        if self.methodid == 2 or self.methodid == 1 or self.methodid == 4:
            # if self.env.agent_facing_id == 0:
            #     out = np.flip(area)
            # elif self.env.agent_facing_id == 1:
            #     out = area
            # elif self.env.agent_facing_id == 2:
            #     out = np.rot90(area, 1)
            # elif self.env.agent_facing_id == 3:
            #     out = np.rot90(area, 3)
            # else:
            #     print("unknown agent facing id: ", self.env.agent_facing_id)
            #     quit()
            if agent_facing_id == 0:
                out = np.flip(area)
            elif agent_facing_id == 1:
                out = area
            elif agent_facing_id == 2:
                out = np.rot90(area, 1)
            elif agent_facing_id == 3:
                out = np.rot90(area, 3)
            else:
                print("unknown agent facing id: ", agent_facing_id)
                quit()
        # elif self.methodid == 1:
        #     out = area
        else:
            print("methodid must be 1,2 or 4 to use agent map")
            exit()

        return out.flatten()

    def get_nearest_items(self, item_ids, agent_loc = None, agent_facing_id = None, envmap = None):
        # TODO: search outwards from agent and kill when all items have been found
        # Or integrate into env when making map
        nearest_dists = [np.inf] * len(self.env.items)
        nearest_coords = np.zeros((len(self.env.items), 2))

        if agent_loc is None:
            agent_location = self.env.agent_location
            # agent_x = self.env.agent_location[1]
            # agent_y = self.env.agent_location[0]
        else:
            agent_location = agent_loc
            # agent_x = agent_loc[1]
            # agent_y = agent_loc[0]
        if agent_facing_id is None:
            agent_facing_id = self.env.agent_facing_id
        if envmap is None:
            envmap = self.env.map

        # Do we want absolute coords or rel coords based on orientation?
        if agent_facing_id == 1:
            envmap = np.flipud(envmap)
            agent_y = self.env.map.shape[0]-1 - agent_location[0]
            agent_x = agent_location[1]
        elif agent_facing_id == 0:
            envmap = envmap
            agent_x = agent_location[1]
            agent_y = agent_location[0]
        elif agent_facing_id == 3:
            envmap = np.rot90(envmap, 1)
            agent_y = self.env.map.shape[1]-1 - agent_location[1]
            agent_x = agent_location[0]
        elif agent_facing_id == 2:
            envmap = np.rot90(envmap, 3)
            agent_y = agent_location[1]
            agent_x = self.env.map.shape[0]-1 - agent_location[0]

        # nearest dist should be manhattan distance
        for i in range(envmap.shape[0]):
            for j in range(envmap.shape[1]):
                item_id = envmap[i][j]
                if item_id in item_ids:
                    dist = np.abs(agent_x - j) + np.abs(agent_y - i)
                    # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
                    if dist < nearest_dists[item_id]:
                        nearest_dists[item_id] = dist
                        nearest_coords[item_id] = (i, j)
                        # if self.env.agent_facing_id == 1:
                        if agent_facing_id == 1:
                            nearest_coords[item_id] = (agent_y - i, agent_x - j)
                        else:
                            nearest_coords[item_id] = (agent_y - i, j - agent_x)

        return nearest_coords[item_ids]

    # #change map to only consider obstacles, permeable spaces, and goal space
    # def convert_map(self, map):
    #     if self.goal_type == 'itemid':
    #         for r in range(len(map.shape(0))):
    #             for c in range(len(map.shape(1))):
    #                 if map[r][c] == self.goal[0]:
    #                     map[r][c] = 2
    #                 else:
    #                     map[r][c] = 1
    #     # if self.goal_type == 'coord':
    #     else:
    #         for r in range(len(map.shape(0))):
    #             for c in range(len(map.shape(1))):
    #                 if r == self.goal[0] and c == self.goal[1]:
    #                     map[r][c] = 2
    #                 else:
    #                     map[r][c] = 1
    #
    # def set_movement_goal(self, goal_type, goal):
    #     assert goal_type in ['itemid', 'coord', 'coord_oriented'], "unsupported goal_type: {}".format(goal_type)
    #     self.goal_type = goal_type
    #     if goal_type == 'itemid':
    #         assert len(goal) == 1, "itemid goal must be len 1"
    #     elif goal_type == 'coord':
    #         assert len(goal) == 2, "coord goal must be len 2"
    #     elif goal_type == 'coord_oriented':
    #         assert len(goal) == 3, "coord_oriented goal must be len 3"
    #     self.movement_goal = goal

    def observation(self, obs=None):
        # if self.obstacle_rep:
        #     map = self.convert_map(self.env.map.copy())

        facing_id = self.env.agent_facing_id
        if self.methodid == 0 or self.methodid == 3:
            map_obs = np.concatenate((self.get_lidarSignal(), [facing_id]))
        elif self.methodid == 1:
            map_obs = np.concatenate((self.get_agentView().flatten(), [facing_id], self.get_nearest_items([6])[0]))
        elif self.methodid == 2 or self.methodid == 4:
            map_obs = np.concatenate((self.get_lidarSignal(), self.get_agentView(), [facing_id]))
        else:
            print("invalid methodid")
            quit()

        if self.obs_rep == 0:
            return map_obs
        else:
            try:
                selected_item = self.env.items_id[self.env.selected_item]
            # holding nothing will be 0 (can't hold air)
            except KeyError:
                selected_item = 0
            inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in
                               self.relevant_inv_items]
            return np.concatenate((map_obs,[selected_item],inventory_items))

    def reset(self):
        self.env.reset()
        return self.observation()

    def step(self, action_id):

        obs, reward, done, info = self.env.step(action_id)
        obs = self.observation()

        # Baseline reward structure - see if incentivizing novelty exploration through
        #   guided action selection or supplied reward performs better or worse
        # -5 reward if attempting to do an action that fails - found not to work, removed
        # -1 reward for any other timestep
        # +100 reward if reaching the goal
        reward = -1
        done = False
        # Was just learning to not do anything that could be impossible
        # if info['result'] == False:
        #     reward = -5
        if self.env.block_in_front_id == self.goal_block_id:
            done = True
            reward = 100

        # Update after each step
        self.env.last_reward = reward
        self.env.last_done = done

        return obs, reward, done, info


######## OLD WRAPPERS FROM TESTING, IGNORE ########

# # From sprint: Don't have functionality to edit layers - add notion of novel object before actually adding it
# class PreNoveltyWrapper(gym.core.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#         # EDIT ENV TO INCLUDE OBS AND ACTIONS RELEVANT FOR NOVELTIES
#         # Need to include upfront for now because do not have functionality to
#         #   add input and output nodes to pretrained net
#         # # axetobreak
#         # self.axe_name = 'wooden_axe'
#         # self.env.items.add(self.axe_name)
#         # self.env.items_id.setdefault(self.axe_name, len(self.env.items_id))
#         # self.env.entities.add(self.axe_name)
#         # self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
#         # self.env.actions_id.update(self.env.select_actions_id)
#         #
#         # # firewall
#         # self.env.items.add('fire_wall')
#         # self.env.items_id.setdefault('fire_wall', len(self.items_id))
#         # self.env.unbreakable_items.add('fire_wall')
#
#         # fence
#         self.fence_name = 'oak_fence'
#         self.env.items.add(self.fence_name)
#         self.env.items_id.setdefault(self.fence_name, len(self.items_id))
#         self.env.select_actions_id.update({'Select_' + self.fence_name: len(self.env.actions_id)})
#         self.env.actions_id.update(self.env.select_actions_id)
#
#         self.action_space = spaces.Discrete(len(self.actions_id))
#
#         assert not self.max_items < len(self.env.items), "Cannot have more than " + str(self.max_items) + " items"
#         # assert self.agent_view_size >= 1, "Increase the agent_view_size"
#
# # From sprint: general obs space wrapper
# class GridworldGeneralWrapper(gym.core.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#         # Observation Space
#         # self.max_items = 20
#         self.max_items = 10
#         self.agent_view_size = 2
#         low = np.concatenate((np.zeros(34),-10*np.ones(16)))
#         high = np.concatenate((10*np.ones(25), 20*np.ones(8), [10], 10*np.ones(16)))
#
#         self.observation_space = spaces.Box(low, high, dtype=int)
#         # self.observation_space = spaces.Box()
#
#         # self.observation_space = spaces.Box(low=0, high=self.max_items,
#         #                                     shape=(self.agent_view_size, self.agent_view_size, 1))
#         # self.observation_space = spaces.Dict({'agent_map': self.observation_space})
#
#     def get_agentView(self):
#         """
#         Slice map with 0 padding based on agent_view_size
#
#         :return: local view of the agent
#         """
#
#         extend = [self.agent_view_size, self.agent_view_size]  # row and column
#         pad_value = 0
#
#         extend = np.asarray(extend)
#         map_ext_shp = self.env.map.shape + 2 * np.array(extend)
#         map_ext = np.full(map_ext_shp, pad_value)
#         insert_idx = [slice(i, -i) for i in extend]
#         map_ext[tuple(insert_idx)] = self.env.map
#         desired_center = self.env.agent_location
#         num_rots = 0
#
#         # Make view box in front of agent, else box around agent
#         # if self.env.agent_facing_id == 0:
#         #     desired_center = (self.env.agent_location[0]-1-self.agent_view_size, self.env.agent_location[1])
#         #     # num_rots = 2
#         # elif self.env.agent_facing_id == 1:
#         #     desired_center = (self.env.agent_location[0]+1+self.agent_view_size, self.env.agent_location[1])
#         #     # num_rots = 0
#         # elif self.env.agent_facing_id == 2:
#         #     desired_center = (self.env.agent_location[0], self.env.agent_location[1]-1-self.agent_view_size)
#         #     # num_rots = 1
#         # elif self.env.agent_facing_id == 3:
#         #     desired_center = (self.env.agent_location[0], self.env.agent_location[1]+1+self.agent_view_size)
#         #     # num_rots = 3
#
#         # desired_center = self.env.agent_location
#
#         # region_idx = [slice(i, j) for i, j in zip(self.env.agent_location, extend * 2 + 1 + self.env.agent_location)]
#         region_idx = [slice(i, j) for i, j in zip(desired_center, extend * 2 + 1 + desired_center)]
#         area = map_ext[tuple(region_idx)]
#
#         if self.env.agent_facing_id == 0:
#             out = np.flip(area)
#         elif self.env.agent_facing_id == 1:
#             out = area
#         elif self.env.agent_facing_id == 2:
#             out = np.rot90(area, 1)
#         elif self.env.agent_facing_id == 3:
#             out = np.rot90(area, 3)
#
#         # only and issue with front view
#         # # fill out obs with 0's if looking past outer wall
#         # if len(out) == 0:
#         #     # TEMP FIX: if next to wall then returns nothing
#         #     return np.array([8,8,8,0,0,0,0,0,0])
#         # elif len(out) != (self.agent_view_size*2+1)**2:
#         #     temp = np.zeros((self.agent_view_size*2+1)**2)
#         #     for i in range(len(out)):
#         #         temp[i] = out[i]
#         #     return temp
#
#         # input('step')
#         # print(out)
#         return out.flatten()
#
#         # return map_ext[tuple(region_idx)]
#
#     def get_nearest_items(self, item_ids):
#         # agent_y, agent_x = self.env.agent_location
#         # TODO: search outwards from agent and kill when all items have been found
#         # Or integrate into env when making map
#         nearest_dists = [np.inf] * len(self.env.items)
#         # nearest_coords = -np.ones((len(self.env.items),2))
#         nearest_coords = np.zeros((len(self.env.items),2))
#         # print(agent_y, agent_x)
#         # print(self.env.map.shape)
#         # print(self.env.agent_location)
#         # print(self.env.map)
#
#         if self.env.agent_facing_id == 1:
#             map = np.flipud(self.env.map)
#             agent_y = self.env.map.shape[0]-1 - self.env.agent_location[0]
#             agent_x = self.env.agent_location[1]
#         elif self.env.agent_facing_id == 0:
#             map = self.env.map
#             agent_x = self.env.agent_location[1]
#             agent_y = self.env.agent_location[0]
#         elif self.env.agent_facing_id == 3:
#             map = np.rot90(self.env.map, 1)
#             agent_y = self.env.map.shape[1]-1 - self.env.agent_location[1]
#             agent_x = self.env.agent_location[0]
#         elif self.env.agent_facing_id == 2:
#             map = np.rot90(self.env.map, 3)
#             agent_y = self.env.agent_location[1]
#             agent_x = self.env.map.shape[0]-1 - self.env.agent_location[0]
#
#         # print(agent_y, agent_x)
#         # print(map)
#
#         # nearest dist should be manhattan distance
#         for i in range(self.env.map.shape[0]):
#             for j in range(map.shape[1]):
#                 item_id = map[i][j]
#                 if item_id in item_ids:
#                     dist = np.abs(agent_x-j) + np.abs(agent_y - i)
#                     # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
#                     if dist < nearest_dists[item_id]:
#                         nearest_dists[item_id] = dist
#                         # nearest_coords[item_id] = (i, j)
#                         if self.env.agent_facing_id == 1:
#                             nearest_coords[item_id] = (agent_y-i, agent_x-j)
#                         else:
#                             nearest_coords[item_id] = (agent_y-i, j-agent_x)
#         # print(nearest_dists)
#         # print(nearest_coords)
#         # print(nearest_coords[item_ids])
#         # make nearest coords relative to agent location and orientation
#
#         return nearest_coords[item_ids]
#
#     def observation(self, obs=None):
#     # def get_obs(self, _):
#     # def get_obs(self):
#         # do we only care about the number of tree_logs in the world?
#         # world_items_quantity =
#         # world_tree_quantity = sum([line.count(6) for line in self.env.map])
#         world_tree_quantity = np.sum([line == 6 for line in self.env.map])
#
#         # relevant_block_ids = [1, 6, 7]
#         relevant_block_ids = [1, 2, 3, 4, 5, 6, 7, 9]
#         # relevant_block_ids will be anything that can be in the world
#         # relevant_block_ids = [1,6,7,8,]
#         relevant_items = ['crafting_table', 'tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick', 'oak_fence']
#         # relevant_items = ['tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick']
#         nearest_item_blocks = self.get_nearest_items(relevant_block_ids)
#         inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in relevant_items]
#         # inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in self.env.items]
#
#         try:
#             selected_item = self.env.items_id[self.env.selected_item]
#         except KeyError:
#             selected_item = -1
#
#         # print(self.get_agentView().flatten())
#         # input("step")
#
#         observation = np.concatenate((self.get_agentView().flatten(),
#                                       # [self.env.agent_facing_id],
#                                       inventory_items,
#                                       [selected_item],
#                                       # [world_tree_quantity],
#                                       nearest_item_blocks.flatten()))
#         # print("agent loc: ", self.env.agent_location)
#         # print("env map: ", self.env.map)
#         # print("agent view: ", observation[:25])
#         # print("facing_id: ", observation[25])
#         # print("inventory_items: ", observation[26:35])
#         # print("selected_item: ", observation[35])
#         # print("world_tree_quantity: ", observation[36])
#         # print("nearest_item_blocks: ", observation[37:43])
#         # print(observation)
#
#         # observation = {'agent_map': self.get_agentView(),
#         #                'agent_facing_id': self.env.agent_facing_id,
#         #                'inventory_items_quantity': self.env.inventory_items_quantity,
#         #                'item_selected': self.env.selected_item,
#         #                # 'world_items_quantity': world_items_quantity,
#         #                'world_tree_quantity': world_tree_quantity,
#         #                'nearest_item_blocks': nearest_item_blocks
#         #                }
#
#         return observation
#
#     # def step(self, action):
#     #     obs, rew, done, info = self.env.step(action)
#     #     return self.get_obs(), int(info['result']), rew
#     #     # return self.get_obs(), int(info['result']), rew
#     #
#     #     # return [self.env.step(action)[0]], [1]
#     #
#     # def reset(self):
#     #     self.env.reset()
#     #     return self.get_obs()
#
# # From sprint: specific obs space wrapper
# class GridworldSpecificWrapper(gym.core.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#         # Observation Space
#         # self.max_items = 20
#         self.max_items = 10
#         self.agent_view_size = 2
#
#         # self.observation_space = spaces.Box(low=0, high=self.max_items,
#         #                                     shape=(self.agent_view_size, self.agent_view_size, 1))
#         # self.observation_space = spaces.Dict({'agent_map': self.observation_space})
#
#     def get_agentView(self):
#         """
#         Slice map with 0 padding based on agent_view_size
#
#         :return: local view of the agent
#         """
#
#         extend = [self.agent_view_size, self.agent_view_size]  # row and column
#         pad_value = 0
#
#         extend = np.asarray(extend)
#         map_ext_shp = self.env.map.shape + 2 * np.array(extend)
#         map_ext = np.full(map_ext_shp, pad_value)
#         insert_idx = [slice(i, -i) for i in extend]
#         map_ext[tuple(insert_idx)] = self.env.map
#         desired_center = self.env.agent_location
#         num_rots = 0
#
#         # Make view box in front of agent, else box around agent
#         # if self.env.agent_facing_id == 0:
#         #     desired_center = (self.env.agent_location[0]-1-self.agent_view_size, self.env.agent_location[1])
#         #     # num_rots = 2
#         # elif self.env.agent_facing_id == 1:
#         #     desired_center = (self.env.agent_location[0]+1+self.agent_view_size, self.env.agent_location[1])
#         #     # num_rots = 0
#         # elif self.env.agent_facing_id == 2:
#         #     desired_center = (self.env.agent_location[0], self.env.agent_location[1]-1-self.agent_view_size)
#         #     # num_rots = 1
#         # elif self.env.agent_facing_id == 3:
#         #     desired_center = (self.env.agent_location[0], self.env.agent_location[1]+1+self.agent_view_size)
#         #     # num_rots = 3
#
#         # desired_center = self.env.agent_location
#
#         # region_idx = [slice(i, j) for i, j in zip(self.env.agent_location, extend * 2 + 1 + self.env.agent_location)]
#         region_idx = [slice(i, j) for i, j in zip(desired_center, extend * 2 + 1 + desired_center)]
#         area = map_ext[tuple(region_idx)]
#
#         if self.env.agent_facing_id == 0:
#             out = np.flip(area)
#         elif self.env.agent_facing_id == 1:
#             out = area
#         elif self.env.agent_facing_id == 2:
#             out = np.rot90(area, 1)
#         elif self.env.agent_facing_id == 3:
#             out = np.rot90(area, 3)
#
#         # only and issue with front view
#         # # fill out obs with 0's if looking past outer wall
#         # if len(out) == 0:
#         #     # TEMP FIX: if next to wall then returns nothing
#         #     return np.array([8,8,8,0,0,0,0,0,0])
#         # elif len(out) != (self.agent_view_size*2+1)**2:
#         #     temp = np.zeros((self.agent_view_size*2+1)**2)
#         #     for i in range(len(out)):
#         #         temp[i] = out[i]
#         #     return temp
#
#         # input('step')
#         # print(out)
#         return out.flatten()
#
#         # return map_ext[tuple(region_idx)]
#
#     def get_nearest_items(self, item_ids):
#         # agent_y, agent_x = self.env.agent_location
#         # TODO: search outwards from agent and kill when all items have been found
#         # Or integrate into env when making map
#         nearest_dists = [np.inf] * len(self.env.items)
#         # nearest_coords = -np.ones((len(self.env.items),2))
#         nearest_coords = np.zeros((len(self.env.items),2))
#         # print(agent_y, agent_x)
#         # print(self.env.map.shape)
#         # print(self.env.agent_location)
#         # print(self.env.map)
#
#         if self.env.agent_facing_id == 1:
#             map = np.flipud(self.env.map)
#             agent_y = self.env.map.shape[0]-1 - self.env.agent_location[0]
#             agent_x = self.env.agent_location[1]
#         elif self.env.agent_facing_id == 0:
#             map = self.env.map
#             agent_x = self.env.agent_location[1]
#             agent_y = self.env.agent_location[0]
#         elif self.env.agent_facing_id == 3:
#             map = np.rot90(self.env.map, 1)
#             agent_y = self.env.map.shape[1]-1 - self.env.agent_location[1]
#             agent_x = self.env.agent_location[0]
#         elif self.env.agent_facing_id == 2:
#             map = np.rot90(self.env.map, 3)
#             agent_y = self.env.agent_location[1]
#             agent_x = self.env.map.shape[0]-1 - self.env.agent_location[0]
#
#         # print(agent_y, agent_x)
#         # print(map)
#
#         # nearest dist should be manhattan distance
#         for i in range(self.env.map.shape[0]):
#             for j in range(map.shape[1]):
#                 item_id = map[i][j]
#                 if item_id in item_ids:
#                     dist = np.abs(agent_x-j) + np.abs(agent_y - i)
#                     # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
#                     if dist < nearest_dists[item_id]:
#                         nearest_dists[item_id] = dist
#                         # nearest_coords[item_id] = (i, j)
#                         if self.env.agent_facing_id == 1:
#                             nearest_coords[item_id] = (agent_y-i, agent_x-j)
#                         else:
#                             nearest_coords[item_id] = (agent_y-i, j-agent_x)
#         # print(nearest_dists)
#         # print(nearest_coords)
#         # print(nearest_coords[item_ids])
#         # make nearest coords relative to agent location and orientation
#
#         return nearest_coords[item_ids]
#
#     def observation(self, obs=None):
#     # def get_obs(self, _):
#     # def get_obs(self):
#         # do we only care about the number of tree_logs in the world?
#         # world_items_quantity =
#         # world_tree_quantity = sum([line.count(6) for line in self.env.map])
#         world_tree_quantity = np.sum([line == 6 for line in self.env.map])
#
#         # relevant_block_ids = [1, 6, 7]
#         relevant_block_ids = [1, 6, 7]
#         # relevant_block_ids will be anything that can be in the world
#         # relevant_block_ids = [1,6,7,8,]
#         # relevant_items = ['crafting_table', 'tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick']
#         relevant_items = ['tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick']
#         nearest_item_blocks = self.get_nearest_items(relevant_block_ids)
#         inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in relevant_items]
#         # inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in self.env.items]
#
#         try:
#             selected_item = self.env.items_id[self.env.selected_item]
#         except KeyError:
#             selected_item = -1
#
#         # print(self.get_agentView().flatten())
#         # input("step")
#
#         observation = np.concatenate((self.get_agentView().flatten(),
#                                       # [self.env.agent_facing_id],
#                                       inventory_items,
#                                       [selected_item],
#                                       # [world_tree_quantity],
#                                       nearest_item_blocks.flatten()))
#         # print("agent loc: ", self.env.agent_location)
#         # print("env map: ", self.env.map)
#         # print("agent view: ", observation[:25])
#         # print("facing_id: ", observation[25])
#         # print("inventory_items: ", observation[26:35])
#         # print("selected_item: ", observation[35])
#         # print("world_tree_quantity: ", observation[36])
#         # print("nearest_item_blocks: ", observation[37:43])
#         # print(observation)
#
#         # observation = {'agent_map': self.get_agentView(),
#         #                'agent_facing_id': self.env.agent_facing_id,
#         #                'inventory_items_quantity': self.env.inventory_items_quantity,
#         #                'item_selected': self.env.selected_item,
#         #                # 'world_items_quantity': world_items_quantity,
#         #                'world_tree_quantity': world_tree_quantity,
#         #                'nearest_item_blocks': nearest_item_blocks
#         #                }
#
#         return observation
#
#     # def step(self, action):
#     #     obs, rew, done, info = self.env.step(action)
#     #     return self.get_obs(), int(info['result']), rew
#     #     # return self.get_obs(), int(info['result']), rew
#     #
#     #     # return [self.env.step(action)[0]], [1]
#     #
#     # def reset(self):
#     #     self.env.reset()
#     #     return self.get_obs()
#
# ## This used?
# # class SimpleCurriculumWrapper(gym.core.Wrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
# #
# #         self.items_quantity = {'crafting_table': 1, 'tree_log': 3}
# #         self.inventory_items_quantity['tree_tap'] = 1
# #
# #         # self.max_items = 10
# #         # self.agent_view_size = 2
#
#
# # Sprint testing wrapper
# import math
# class CurriculumPreNoveltyLidarInFront(gym.core.ObservationWrapper):
#     """
#     Send several beans (num_beams) at equally spaced angles in 360 degrees in front of agent + agent's current
#     inventory
#     """
#
#     def __init__(self, env, task_ind, novelty_ind=0, num_beams=8):
#         super().__init__(env)
#
#         # Observation Space
#         self.num_beams = num_beams
#         self.novelty_ind = novelty_ind
#         if novelty_ind == 0:
#             item = 'oak_fence'
#         else:
#             item = 'fire_wall'
#         # exclude all irrelevant items
#         # tree tap only in moveTo case
#         items_to_exclude = ['air', self.goal_item_to_craft, 'plank', 'stick', 'rubber']
#         if task_ind == 0 or task_ind == 1 or task_ind == 2 or task_ind == 3 or task_ind == 4:
#             self.lidar_items = ['crafting_table', 'tree_log', 'wall', item]
#             # self.lidar_items_id = [1,6,8,9]
#             self.lidar_items_id = {'crafting_table': 0, 'tree_log': 1, 'wall': 2, item: 3}
#         elif task_ind == 5:
#             self.lidar_items = ['crafting_table', 'tree_log', 'tree_tap', 'wall', item]
#             # self.lidar_items_id = [1,6,7,8,9]
#             self.lidar_items_id = {'crafting_table': 0, 'tree_log': 1, 'tree_tap': 2, 'wall': 3, item: 4}
#         elif task_ind == 6 or task_ind == 7 or task_ind == 8:
#             self.lidar_items = ['crafting_table', 'plank', 'rubber', 'stick', 'tree_log', 'tree_tap', 'wall', item]
#             # self.lidar_items_id = [1,6,7,8,9]
#             self.lidar_items_id = {'crafting_table': 0, 'plank': 1, 'rubber' : 2, 'stick' : 3, 'tree_log': 4, 'tree_tap': 5, 'wall': 6, item: 7}
#
#         # Add fence so beam for fence is present
#         if novelty_ind == 0:
#             self.fence_name = 'oak_fence'
#             self.env.items.add(self.fence_name)
#             self.env.items_id.setdefault(self.fence_name, len(self.items_id))
#             self.env.select_actions_id.update({'Select_' + self.fence_name: len(self.env.actions_id)})
#             self.env.actions_id.update(self.env.select_actions_id)
#             self.action_space = spaces.Discrete(len(self.actions_id))
#
#         # self.lidar_items = set(self.items_id.keys())
#         # set(map(self.lidar_items.remove, items_to_exclude))  # remove air and goal_item_to_craft from the lidar_items
#         # self.lidar_items_id = self.set_items_id(self.lidar_items)  # set IDs for all the lidar items
#         self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotenuse of a square
#         low = np.array([0] * (len(self.lidar_items) * self.num_beams) +
#                        [0] * (len(self.inventory_items_quantity)+1 - len(self.unbreakable_items)))
#         high = np.array([self.max_beam_range] * (len(self.lidar_items) * self.num_beams) +
#                         [20] * (len(self.inventory_items_quantity)+1 - len(
#             self.unbreakable_items)))  # 20 is max quantity of any item in inventory
#         self.observation_space = spaces.Box(low, high, dtype=int)
#
#     def get_lidarSignal(self):
#         """
#         Send several beams (num_beams) at equally spaced angles in 360 degrees in front of agent within a range
#         For each bean store distance (beam_range) for each item in lidar_items_id if item is found otherwise 0
#         and return lidar_signals
#         """
#
#         direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}
#
#         # Shoot beams in 360 degrees in front of agent
#         angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi,
#                                   direction_radian[self.agent_facing_str] + np.pi,
#                                   self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360
#
#         lidar_signals = []
#         r, c = self.agent_location
#         for angle in angles_list:
#             x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)
#             beam_signal = np.zeros(len(self.lidar_items_id), dtype=int)
#
#             # Keep sending longer beams until hit an object or wall
#             for beam_range in range(1, self.max_beam_range + 1):
#                 r_obj = r + np.round(beam_range * x_ratio)
#                 c_obj = c + np.round(beam_range * y_ratio)
#                 obj_id_rc = self.map[int(r_obj)][int(c_obj)]
#
#                 # If bean hit an object or wall
#                 if obj_id_rc != 0:
#                     item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
#                     if item in self.lidar_items_id:
#                         obj_id_rc = self.lidar_items_id[item]
#                         beam_signal[obj_id_rc - 1] = beam_range
#                     break
#
#             lidar_signals.extend(beam_signal)
#
#         return lidar_signals
#
#     def observation(self, obs=None):
#         """
#         observation is lidarSignal + inventory_items_quantity
#         :return: observation
#         """
#         selected_item_id = [0] if self.selected_item not in self.items_id else [self.items_id[self.selected_item]]
#
#         lidar_signals = self.get_lidarSignal()
#
#         # relevant_block_ids = [1, 6, 7]
#         # CHANGE BACK
#         if self.novelty_ind == 0:
#             relevant_items = ['tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick', 'oak_fence']
#         else:
#             relevant_items = ['tree_log', 'plank', 'stick', 'tree_tap', 'rubber', 'pogo_stick', 'pogo_stick']
#
#         # nearest_item_blocks = self.get_nearest_items(relevant_block_ids)
#         inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in relevant_items]
#
#         # obs = np.concatenate((lidar_signals, inventory_items, selected_item_id, nearest_item_blocks.flatten()))
#         obs = np.concatenate((lidar_signals, inventory_items, selected_item_id))
#         return obs
#         # obs = lidar_signals + [self.inventory_items_quantity[item] for item in sorted(self.inventory_items_quantity)
#         #                        if item not in self.unbreakable_items] + np.array(selected_item_id)
#
#         # return np.array(obs)
#
#
#     def get_nearest_items(self, item_ids):
#         # agent_y, agent_x = self.env.agent_location
#         # TODO: search outwards from agent and kill when all items have been found
#         # Or integrate into env when making map
#         nearest_dists = [np.inf] * len(self.env.items)
#         nearest_coords = np.zeros((len(self.env.items),2))
#
#
#         # Dont flip map - just have absolute coords
#         # if self.env.agent_facing_id == 1:
#         #     map = np.flipud(self.env.map)
#         #     agent_y = self.env.map.shape[0]-1 - self.env.agent_location[0]
#         #     agent_x = self.env.agent_location[1]
#         # elif self.env.agent_facing_id == 0:
#         map = self.env.map
#         agent_x = self.env.agent_location[1]
#         agent_y = self.env.agent_location[0]
#         # elif self.env.agent_facing_id == 3:
#         #     map = np.rot90(self.env.map, 1)
#         #     agent_y = self.env.map.shape[1]-1 - self.env.agent_location[1]
#         #     agent_x = self.env.agent_location[0]
#         # elif self.env.agent_facing_id == 2:
#         #     map = np.rot90(self.env.map, 3)
#         #     agent_y = self.env.agent_location[1]
#         #     agent_x = self.env.map.shape[0]-1 - self.env.agent_location[0]
#
#         # nearest dist should be manhattan distance
#         for i in range(self.env.map.shape[0]):
#             for j in range(map.shape[1]):
#                 item_id = map[i][j]
#                 if item_id in item_ids:
#                     dist = np.abs(agent_x-j) + np.abs(agent_y - i)
#                     # dist = np.sqrt((agent_x - j)**2 + (agent_y - i)**2)
#                     if dist < nearest_dists[item_id]:
#                         nearest_dists[item_id] = dist
#                         # nearest_coords[item_id] = (i, j)
#                         if self.env.agent_facing_id == 1:
#                             nearest_coords[item_id] = (agent_y-i, agent_x-j)
#                         else:
#                             nearest_coords[item_id] = (agent_y-i, j-agent_x)
#         # print(nearest_dists)
#         # print(nearest_coords)
#         # print(nearest_coords[item_ids])
#         # make nearest coords relative to agent location and orientation
#
#         return nearest_coords[item_ids]
#
#
# import math
# # General obs wrapper to test different movement policy approaches
# class MovementObsWrapper(gym.core.ObservationWrapper):
#     def __init__(self, env, methodid, obstacle_rep, agent_view_size=2, num_beams=8):
#         super().__init__(env)
#
#         # Methodid:
#         #   0 - Lidar + agent_facing_id
#         #   1 - agent map + agent_facing_id
#         #   2 - oriented agent map + agent_facing_id
#         # (for non-CHER case, would we have to add some notion of the goal location for when it's out of range?
#         self.methodid = methodid
#         self.obstacle_rep = obstacle_rep
#         self.goal_type = None
#         self.movement_goal = None
#
#         # Item types are either:
#         #   permeable, obstacle, goal
#         #   all item_types, goal
#         # For CHER we'd still want to do the same thing with converting goal locations right?
#         if not self.obstacle_rep:
#             self.num_item_types = 10
#             print("not doing all item rep yet")
#             quit()
#         else:
#             self.num_item_types = 3
#         # self.num_item_types = 3 if self.convert_map else 10
#
#         if self.methodid == 0:
#             # self.lidar_items = 3 #obstacle or goal
#             self.num_lidar_items = self.num_item_types - 1
#             self.num_beams = num_beams
#             self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotenuse of a square
#             low = np.array([0] * (self.num_lidar_items * self.num_beams) +
#                            [0])
#             high = np.array([self.max_beam_range] * (self.num_lidar_items * self.num_beams) +
#                             [4])
#         else:
#             # Add rel coords
#             self.agent_view_size = agent_view_size
#             agent_map_size = (agent_view_size*2+1)**2
#             map_size = 10
#             low = np.concatenate((np.zeros(agent_map_size), [0]))
#             high = np.concatenate((self.num_item_types * np.ones(agent_view_size), [4]))
#
#         self.observation_space = spaces.Box(low, high, dtype=int)
#
#     def get_lidarSignal(self, map):
#         """
#         Send several beams (num_beams) at equally spaced angles in 360 degrees in front of agent within a range
#         For each bean store distance (beam_range) for each item in lidar_items_id if item is found otherwise 0
#         and return lidar_signals
#         """
#
#         direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}
#
#         # Shoot beams in 360 degrees in front of agent
#         angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi,
#                                   direction_radian[self.agent_facing_str] + np.pi,
#                                   self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360
#
#         lidar_signals = []
#         r, c = self.agent_location
#         for angle in angles_list:
#             x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)
#             beam_signal = np.zeros(self.num_lidar_items, dtype=int)
#
#             # Keep sending longer beams until hit an object or wall
#             for beam_range in range(1, self.max_beam_range + 1):
#                 r_obj = r + np.round(beam_range * x_ratio)
#                 c_obj = c + np.round(beam_range * y_ratio)
#                 obj_id_rc = map[int(r_obj)][int(c_obj)]
#
#                 # If bean hit an object or wall
#                 if obj_id_rc != 0:
#                     if not self.obstacle_rep:
#                         item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
#                         if item in self.lidar_items_id:
#                             obj_id_rc = self.lidar_items_id[item]
#                             beam_signal[obj_id_rc - 1] = beam_range
#                         break
#                     if self.obstacle_rep:
#                         if obj_id_rc <= self.num_lidar_items:
#                             beam_signal[obj_id_rc - 1] = beam_range
#                         break
#             lidar_signals.extend(beam_signal)
#
#         return lidar_signals
#
#     def get_agentView(self, map):
#         """
#         Slice map with 0 padding based on agent_view_size
#
#         :return: local view of the agent
#         """
#
#         extend = [self.agent_view_size, self.agent_view_size]  # row and column
#         pad_value = 0
#
#         extend = np.asarray(extend)
#         map_ext_shp = map.shape + 2 * np.array(extend)
#         map_ext = np.full(map_ext_shp, pad_value)
#         insert_idx = [slice(i, -i) for i in extend]
#         map_ext[tuple(insert_idx)] = map
#         desired_center = self.env.agent_location
#
#         region_idx = [slice(i, j) for i, j in zip(desired_center, extend * 2 + 1 + desired_center)]
#         area = map_ext[tuple(region_idx)]
#
#         if self.methodid == 2:
#             if self.env.agent_facing_id == 0:
#                 out = np.flip(area)
#             elif self.env.agent_facing_id == 1:
#                 out = area
#             elif self.env.agent_facing_id == 2:
#                 out = np.rot90(area, 1)
#             elif self.env.agent_facing_id == 3:
#                 out = np.rot90(area, 3)
#             else:
#                 print("unknown agent facing id: ", self.env.agent_facing_id)
#                 quit()
#         elif self.methodid == 1:
#             out = area
#         else:
#             print("methodid must be 1 or 2 to use agent map")
#             exit()
#
#         return out.flatten()
#
#     #change map to only consider obstacles, permeable spaces, and goal space
#     def convert_map(self, map):
#         if self.goal_type == 'itemid':
#             for r in range(len(map.shape(0))):
#                 for c in range(len(map.shape(1))):
#                     if map[r][c] == self.goal[0]:
#                         map[r][c] = 2
#                     else:
#                         map[r][c] = 1
#         # if self.goal_type == 'coord':
#         else:
#             for r in range(len(map.shape(0))):
#                 for c in range(len(map.shape(1))):
#                     if r == self.goal[0] and c == self.goal[1]:
#                         map[r][c] = 2
#                     else:
#                         map[r][c] = 1
#
#     def set_movement_goal(self, goal_type, goal):
#         assert goal_type in ['itemid', 'coord', 'coord_oriented'], "unsupported goal_type: {}".format(goal_type)
#         self.goal_type = goal_type
#         if goal_type == 'itemid':
#             assert len(goal) == 1, "itemid goal must be len 1"
#         elif goal_type == 'coord':
#             assert len(goal) == 2, "coord goal must be len 2"
#         elif goal_type == 'coord_oriented':
#             assert len(goal) == 3, "coord_oriented goal must be len 3"
#         self.movement_goal = goal
#
#     def observation(self, obs=None):
#         if self.obstacle_rep:
#             map = self.convert_map(self.env.map.copy())
#
#         facing_id = self.env.agent_facing_id
#         if self.methodid == 0:
#             lidar_signals = self.get_lidarSignal(map)
#             return np.concatenate((lidar_signals,[facing_id]))
#         elif self.methodid == 1 or self.methodid == 2:
#             return np.concatenate((self.get_agentView(map).flatten(), [facing_id]))
#         else:
#             print("invalid methodid")
#             quit()
#
#
# class MovementGoalWrapper(gym.core.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def step(self, action_id):
#
#         obs, reward, done, info = self.env.step(action_id)
#
#         r, c = self.agent_location
#         close_to_fire_wall = False
#         # NORTH
#         if (0 <= (r - 1) <= self.map_size - 1) and self.map[r - 1][c] == self.env.items_id['fire_wall']:
#             close_to_fire_wall = True
#         # SOUTH
#         elif (0 <= (r + 1) <= self.map_size - 1) and self.map[r + 1][c] == self.env.items_id['fire_wall']:
#             close_to_fire_wall = True
#         # WEST
#         elif (0 <= (c - 1) <= self.map_size - 1) and self.map[r][c - 1] == self.env.items_id['fire_wall']:
#             close_to_fire_wall = True
#         # EAST
#         elif (0 <= (c + 1) <= self.map_size - 1) and self.map[r][c + 1] == self.env.items_id['fire_wall']:
#             close_to_fire_wall = True
#
#         if close_to_fire_wall:
#             reward = -self.reward_done // 2
#             done = True
#             info['message'] = 'You died due to fire_wall'
#
#         # Update after each step
#         self.env.last_reward = reward
#         self.env.last_done = done
#
#         return obs, reward, done, info
#
# # Wrapper for obs relating to recipe level space
# class RecipeLevelWrapper(gym.core.ObservationWrapper):
#     """
#     Send several beans (num_beams) at equally spaced angles in 360 degrees in front of agent + agent's current
#     inventory
#     """
#
#     def __init__(self, env):
#         super().__init__(env)
#
#         self.max_world_items = 5
#         self.max_inv_items = 20
#         self.max_item_types = 10
#         # moveTo or pickUp pogostick should never be an option in exploration because then planner would just say pickup
#         # also moveTo outer wall should be useless (maybe its not and we want to include?)
#         self.relevant_world_items = [1, 2, 4, 5, 6, 7]
#         # likewise don't care about air,pogostick,or wall in inv
#         self.relevant_inv_items = [1, 2, 4, 5, 6, 7]
#         # Should allow moveTo air?
#         # action space:                               Initially can be
#         #   moveTo(relevant_world_items) (6)          (path planner or learned policy)
#         #   craft(recipe) (4)                         (primitive)
#         #   breakBlock (1)                            (primitive)
#         #   select(item) (6) (7? to include deselect) (primtive)
#         #   place_crafting_table (1)                  (primitive)
#         #   tap_tree (1) (0?)                         (sequence or learned policy)
#         #   extract_rubber (place tree_tap separate or not?) (1) (primtiive)
#         # including deselect and separate tap_tree and extract_rubber
#         self.action_space = spaces.Discrete(22)
#         # Larger action space and smaller state space should be better for pretrained agent that knows
#         #   what actions are generally possible and what aren't given a state
#
#         ## Observation space as single Box
#         # low = np.array([0, 0] +
#         #                [0] * (len(self.relevant_inv_items)) +
#         #                [0] * (len(self.relevant_world_items)))
#         # # premake space for novel items or add as we go?
#         # high = np.array([len(self.items), len(self.items)] +
#         #                 [self.max_inv_items] * (len(self.relevant_inv_items)) +
#         #                 [self.max_world_items] * (len(self.relevant_world_items)))
#         # self.observation_space = spaces.Box(low, high, dtype=int)
#
#         # Observation space as Dict
#         # Align more exactly with actual types in env? e.g. selected_item can't be wall
#         block_in_front_space = spaces.Box(low=[0], high=[self.max_item_types])
#         selected_item_space = spaces.Box(low=[0], high=[self.max_item_types])
#         inventory_items_quantity_space = spaces.Box(low=[0] * (len(self.relevant_inv_items)), high=[self.max_item_types] * (len(self.relevant_inv_items)))
#         world_items_quantity_space = spaces.Box(low=[0] * (len(self.relevant_world_items)), high=[self.max_world_items] * (len(self.relevant_world_items)))
#         self.observation_space = spaces.Dict({'block_in_front': block_in_front_space,
#                                               'selected_item': selected_item_space,
#                                               'inventory_items_quantity': inventory_items_quantity_space,
#                                               'world_items_quantity': world_items_quantity_space})
#
#     def observation(self, obs=None):
#         """
#         observation is block_in_front + selected_item + inventory_items_quantity + items_in_world
#         :return: observation
#         """
#         block_in_front_id = self.env.block_in_front_id
#         selected_item_id = 0 if self.env.selected_item not in self.items_id else self.items_id[self.env.selected_item]
#         inventory_items = [self.env.inventory_items_quantity[item_name] for item_name in self.relevant_inv_items]
#
#         items_in_world = [np.count_nonzero(self.env.map == item_id) for item_id in self.relevant_world_items]
#
#         ## observation as single array
#         # obs = np.concatenate(([block_in_front_id, selected_item_id], inventory_items, items_in_world))
#         # return obs
#
#         ## observation as dict
#         obs = {'block_in_front': block_in_front_id,
#                'selected_item': selected_item_id,
#                'inventory_items_quantity': inventory_items,
#                'world_items_quantity': items_in_world}
#         return obs