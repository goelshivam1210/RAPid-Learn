import gym
import numpy as np
import tensorflow as tf
import time
import math

from polycraft_tufts.rl_agent.dqn_lambda.learning.q_functions import pogostick_mlp
from polycraft_tufts.rl_agent.dqn_lambda.learning.utils import *
from polycraft_tufts.rl_agent.dqn_lambda.learning.replay_memory import make_replay_memory
# from polycraft_tufts.rl_agent.dqn_lambda.envs.polycraft_mdp import MDP
from polycraft_tufts.rl_agent.dqn_lambda.detectors import get_inv_quant
from scipy.special import softmax

tf.get_logger().setLevel('ERROR')

# TODO: add default args
class DQNLambda_Agent(object):
    def __init__(self, seed, env, return_est_method, replay_capacity, history_len, discount, cache_size,
                 block_size, priority, learning_rate, prepopulate, max_epsilon, min_epsilon, eps_lambda,
                 batch_size, max_timesteps, update_freq, grad_clip=None, session=None, hidden_size=32, scope=None):
        if session is None:
            self.session = make_session(seed)
        else:
            self.session = session
        self.env = env
        # Make in utils
        self.q_function = pogostick_mlp(hidden_size)
        self.replay_memory = make_replay_memory(return_est=return_est_method, capacity=replay_capacity,
                                                history_len=history_len, discount=discount, cache_size=cache_size,
                                                block_size=block_size, priority=priority)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        assert prepopulate >= self.replay_memory.block_size, "ERROR: prepopulate must be at least block_size for initial sample"
        self.prepopulate = prepopulate
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        #don't want exploration to latch onto irrelevant novelties too hard
        if scope == 'exploration':
            self.min_epsilon = 0.4
        else:
            self.min_epsilon = min_epsilon
        if eps_lambda is None:
            eps_lambda = -math.log(0.01) / max_timesteps
        self.eps_lambda = eps_lambda

        if env.observation_space is None or env.action_space is None:
            # assert isinstance(self.env, MDP), "ERROR: observations and/or action space not defined and env is not a polycraft MDP"
            try:
                self.env.generate_obs_action_spaces()
            except:
                print("ERROR: observations and/or action space not defined and env is not a polycraft or gridworld MDP")
                quit()
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        if scope.startswith('moveTo'):
            input_shape = (self.env.observation_space.shape[0]+2,)
        else:
            input_shape = self.env.observation_space.shape

        # self.input_shape = (self.replay_memory.history_len, *self.env.observation_space.shape)
        self.input_shape = (self.replay_memory.history_len, *input_shape)
        self.n_actions = self.env.action_space.n

        self.legacy_mode = False
        if scope is None:
            self.scope = 'main_{}'.format(seed)
        else:
            self.scope = scope
        self.grad_clip = grad_clip
        self.build_training_funcs()
        self.batch_size = batch_size
        self.timesteps_trained = 0.0
        self.eps_trained = 0
        self.target_update_freq = update_freq
        # Believe max_timesteps doesn't matter if priority==0, which it is in all cases used atm
        #   for tournament setting don't know number to timesteps we will be training so don't want this to matter
        self.max_timesteps = max_timesteps
        # Want indicator of if we've ever found reward 1. for stat keeping and 2. to know if evaluate is feasible
        self.found_reward = False

    def build_training_funcs(self):
        # Build TensorFlow model
        state_ph  = tf.placeholder(self.env.observation_space.dtype, [None] + list(self.input_shape))
        action_ph = tf.placeholder(tf.int32, [None])
        return_ph = tf.placeholder(tf.float32, [None])

        qvalues = self.q_function(state_ph, self.n_actions, scope=self.scope)

        greedy_actions = tf.argmax(qvalues, axis=1)
        greedy_qvalues = tf.reduce_max(qvalues, axis=1)

        action_indices = tf.stack([tf.range(tf.size(action_ph)), action_ph], axis=-1)
        onpolicy_qvalues = tf.gather_nd(qvalues, action_indices)

        td_error = return_ph - onpolicy_qvalues
        loss = tf.reduce_mean(tf.square(td_error))

        def refresh(states, actions):
            assert len(states) == len(actions) + 1  # We should have an extra bootstrap state
            greedy_qvals, greedy_acts, onpolicy_qvals = self.session.run([greedy_qvalues, greedy_actions, onpolicy_qvalues], feed_dict={
                state_ph: states,
                action_ph: actions,
            })
            mask = (actions == greedy_acts[:-1])
            return greedy_qvals, mask, onpolicy_qvals

        self.refresh = refresh

        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        train_op = minimize_with_grad_clipping(self.optimizer, loss, main_vars, self.grad_clip)

        self.replay_memory.register_refresh_func(refresh)

        self.session.run(tf.global_variables_initializer())

        def epsilon_greedy(state, epsilon, info=None):
            if np.random.random() < epsilon:
                # # In random action selection, only allow actions that we know will succeed (at least prenovelty)
                # # Want to avoid doing this generally, because if an action suddenly becomes possible for whatever
                # #   reason post-novelty, we don't want to make it impossible for the agent to solve the task
                # # But in random exploration we can prevent the agent from doing things that probably won't
                # if info is not None:
                #     action = self.informed_random_action(info)
                # # # Biasing random action selecting to prefer movement when not in front of anything
                # # # Could/should do a lot more here on top of explicity exploration
                # # # print("WARN - biasing random exploration to movement")
                # # if self.env.block_in_front['name'] == 'minecraft:air':
                # #     probas = [0.2,0.2,0.2] + [0.4 / float(self.n_actions-3) for i in range(self.n_actions-3)]
                # #     action = np.random.choice(np.arange(0,self.n_actions), p=probas)
                # else:
                #     action = np.random.randint(self.n_actions)
                # # action = self.env.action_space.sample()
                action = np.random.randint(self.n_actions)
            else:
                # action = self.session.run(qvalues, feed_dict={state_ph: state[None]})[0]
                # print(action)
                action = self.session.run(greedy_actions, feed_dict={state_ph: state[None]})[0]
                # print(action)
                # quit()
            return action

        def epsilon_softmax(state, epsilon, info=None):
            if np.random.random() < epsilon:
                action = np.random.randint(self.n_actions)
            else:
                qvals = self.session.run(qvalues, feed_dict={state_ph: state[None]})[0]
                probas = softmax(qvals)
                # probas = np.exp(qvals)/np.sum(np.exp(qvals))
                action = np.random.choice(range(self.n_actions), p=probas)
            return action

        self.epsilon_greedy = epsilon_greedy
        self.epsilon_softmax = epsilon_softmax


        def train():
            state_batch, action_batch, return_batch = self.replay_memory.sample(self.batch_size)

            self.session.run(train_op, feed_dict={
                state_ph: state_batch,
                action_ph: action_batch,
                return_ph: return_batch,
            })
        self.train = train

    #
    def copy_variables_from_scope(self, scope):
        # print(scope)
        other_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        # print(other_vars)
        # for variable in other_vars:
        #     print(variable.name)
        #     print(self.session.run(variable))
        #
        # print(self.scope)
        this_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        # print(this_vars)
        # for variable in this_vars:
        #     print(variable.name)
        #     print(self.session.run(variable))

        assert len(other_vars) == len(this_vars), 'ERROR: TF variables have mismatched lengths between scopes we are trying to copy between'
        # print('copying vars')
        for i in range(len(other_vars)):
            self.session.run(this_vars[i].assign(self.session.run(other_vars[i])))
            # print('copied var mid loop')
            # print(this_vars[i])
            # print(self.session.run(this_vars[i]))

        # for this_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope):
        #     print('copied var post loop')
        #     print(this_var)
        #     print(self.session.run(this_var))
        #
        # quit()

    # #TODO: Move this to other agent? And do eps greedy outside
    # def informed_random_action(self, info, goal):
    #     # if self.scope == 'breakminecraftlog':
    #     #     for this_var, other_var in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exploration'),
    #     #                                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='breakminecraftlog')):
    #     #         print(this_var, other_var)
    #     #         print(self.session.run(this_var), self.session.run(other_var))
    #     #         break
    #     #Update both impermissible and probabilties based on novelty
    #
    #     # TODO: reason about goal of agent or past successful trajectories
    #     invalid_action = True
    #     # TODO: create valid action pool and sample from that rather than repeatedly sample and check?
    #     while invalid_action:
    #         action = np.random.randint(self.n_actions)
    #         action_str = self.env.all_actions[action]
    #         # print(action_str)
    #         # assert recipe ingredients in inv
    #         # TODO: Learning: check if goal set contains 'increase item'
    #         #   If so, only allow craft if it's the goal item or a component of the goal item which we do not have
    #         #       (In a way that we don't waste too many resources)
    #         # TODO: Exploration: Only equally consider 'wasteful' actions once to test, past that we want to discourage
    #         #    anything that may make it impossible to reach the goal (if we rely on plannable states to indicate
    #         #   success, then achieving subgoal but making task impossible will cause us to miss what we've learned)
    #         #TODO: Improve crafting operators such that as long as we don't have too little materials, we can craft a
    #         #   path to the goal item. Disallow 'impermissible' crafting actions (if it's more than necessary - exact or total?)
    #         if action_str.split()[0] == 'CRAFT':
    #             ## DISALLOW lower level craft actions in exploration, any other way we're just going to turn
    #             ##   everything into sticks every round
    #             item_str = action_str.split()[1]
    #             # if item_str == 'minecraft:planks':
    #             #     invalid_action = get_inv_quant(info, 'minecraft:log') < 1
    #             # elif item_str == 'minecraft:stick':
    #             #     invalid_action = get_inv_quant(info, 'minecraft:planks') < 2
    #             if item_str == 'minecraft:tree_tap':
    #                 invalid_action = get_inv_quant(info, 'minecraft:planks') < 5 or get_inv_quant(info,
    #                                                                                               'minecraft:stick') < 1 or \
    #                                  info['block_in_front']['name'] != 'minecraft:crafting_table'
    #             elif item_str == 'minecraft:wooden_pogo_stick':
    #                 invalid_action = get_inv_quant(info, 'minecraft:planks') < 2 or get_inv_quant(info,
    #                                                                                               'minecraft:stick') < 4 or get_inv_quant(
    #                     info, 'polycraft:sack_polyisoprene_pellets') < 1 or info['block_in_front'][
    #                                      'name'] != 'minecraft:crafting_table'
    #             else:
    #                 invalid_action = True
    #         # assert item in inv
    #         elif action_str.split()[0] == 'SELECT_ITEM':
    #             invalid_action = get_inv_quant(info, action_str.split()[1]) < 1 or info['selected_item'] == \
    #                              action_str.split()[1]
    #         # assert block in front is air
    #         elif action_str.split()[0] == 'MOVE':
    #             invalid_action = info['block_in_front']['name'] != 'minecraft:air'
    #         # assert block in front is not air
    #         elif action_str == 'BREAK_BLOCK':
    #             invalid_action = info['block_in_front']['name'] == 'minecraft:air' or info['block_in_front'][
    #                 'name'] == 'minecraft:crafting_table'
    #         elif action_str == 'PLACE_TREE_TAP':
    #             invalid_action = get_inv_quant(info, 'polycraft:tree_tap') < 1 or info['block_in_front'][
    #                 'name'] != 'minecraft:air'
    #         elif action_str == 'PLACE_CRAFTING_TABLE':
    #             invalid_action = get_inv_quant(info, 'minecraft:crafting_table') < 1 or info['block_in_front'][
    #                 'name'] != 'minecraft:air'
    #         # assert block in front is tree tap
    #         elif action_str == 'EXTRACT_RUBBER':
    #             invalid_action = info['block_in_front']['name'] != 'polycraft:tree_tap'
    #         else:
    #             invalid_action = False
    #         # print(invalid_action)
    #     return action

    def set_explore_epsilon(self, e):
        self.epsilon = e

    def run_episode(self, ep_t_limit, render=False, reset=True):
        t = 0
        reward_sum = 0
        if reset:
            obs = self.env.reset()
            if render:
                self.env.render()
                self.env.render()
        else:
            obs = self.env.observation()
            if render:
                self.env.render()
        success = True

        while True:
            self.replay_memory.store_obs(obs)
            # with history_len == 1, just wraps obs in array
            obs = self.replay_memory.encode_recent_observation()
            print("WARNING: need to update epsilon externally, change to be internal")
            action = self.epsilon_greedy(obs, self.epsilon)
            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
            t += 1
            if t >= ep_t_limit:
                done = True
                success = False
            self.replay_memory.store_effect(action, reward, done)
            if render:
                self.env.render()

            if done:
                self.timesteps_trained += t
                self.eps_trained += 1
                # Must prepopulate buffer to at least block_size
                if self.timesteps_trained > self.prepopulate and self.eps_trained % self.target_update_freq == 0:
                    train_frac = max(0, 0, (self.timesteps_trained / self.max_timesteps))
                    self.replay_memory.refresh(train_frac)
                    num_train_iterations = self.replay_memory.cache_size // self.batch_size
                    for _ in range(num_train_iterations):
                        self.train()
                return reward_sum, success, t

    # Can experiment with approaches on how to not get stuck here
    #   (e.g. in evaluate trials sometimes the agent just turns left/right repeatedly,
    #    or tries forward or break repeatedly without any effect)
    def run_evaluate_episode(self, ep_t_limit, render=False, reset=True):
        t = 0
        reward_sum = 0
        if reset:
            obs = self.env.reset()
            if render:
                self.env.render()
                self.env.render()
        else:
            obs = self.env.observation()
            if render:
                self.env.render()

        # # Idea: keep track of history of length N of states and actions, if performing the same action in the
        # #       same state within the history length, pick a random action instead
        # # Could also restrict impossible actions (although notion of what's 'impossible' may change post novelty)
        # # Or just keep some (but lower) amount of random_eps even in evaluate
        # past_actions = -1*np.ones(2)
        # past_states = np.zeros((2,len(obs)))
        # past_states[0] = obs

        success = True
        while True:
            action = self.epsilon_greedy(np.array([obs]), 0.0)

            # # check history for same action/obs pair
            # for i in range(2):
            #     if action == past_actions[i] and np.all(obs == past_states[i]):
            #         print("picking random action")
            #         action = np.random.randint(4)

            obs, reward, done, info = self.env.step(action)
            #
            # past_actions[t%2] = action
            # past_states[(t+1)%2] = obs

            if render:
                self.env.render()
            reward_sum += reward
            t += 1
            if t >= ep_t_limit:
                done = True
                success = False

            if done:
                return reward_sum, success, t

    ###################################################################
    ############ Start Polycraft Tournament Specific Funcs ############
    ###################################################################

    # Reimplement when we have buffer logic in collection agent resolved
    # # Allows for functionality to incorporate previously generated experience
    # # Potentially useful in tournament settings if we find a new goal path
    # # Related idea - could wait to feed data, or refeed data, once a reward is found?
    # # Format of experience? Constant stream, batch of episodes, single episode, single transition?
    # def store_episode(self, episode):
    #     # for episode in episodes:
    #     t = 0
    #     # transition is (s,a)? (s,a,s')?
    #     for transition in episode:
    #         obs = transition[0]
    #         action = transition[1]
    #         self.replay_memory.store_obs(obs)
    #         # obs = self.replay_memory.encode_recent_observation()
    #         success = self.compute_success(obs)
    #         reward = 100 if success else -1
    #         self.replay_memory.store_effect(action, reward, success)
    #         t += 1
    #         #only update on success?
    #         if success:
    #             self.timesteps_trained += t
    #             self.eps_trained += 1
    #             if self.timesteps_trained > self.prepopulate and self.eps_trained % self.target_update_freq == 0:
    #                 train_frac = max(0, 0, (self.timesteps_trained / self.max_timesteps))
    #                 self.replay_memory.refresh(train_frac)
    #                 num_train_iterations = self.replay_memory.cache_size // self.batch_size
    #                 for _ in range(num_train_iterations):
    #                     self.train()

    def check_update(self):
        # Must prepopulate buffer to at least block_size
        # if self.timesteps_trained > self.prepopulate and self.eps_trained % self.target_update_freq == 0:
        if self.timesteps_trained > self.prepopulate and self.timesteps_trained % self.target_update_freq == 0:
            if self.timesteps_trained - self.prepopulate <= self.target_update_freq:
                print("Prepopulate of buffer achieved, beginning training updates every {} timesteps\n".format(self.target_update_freq))
            # train_frac only matters when priority!=0
            train_frac = max(0, 0, (self.timesteps_trained / self.max_timesteps))
            self.replay_memory.refresh(train_frac)
            num_train_iterations = self.replay_memory.cache_size // self.batch_size
            for _ in range(num_train_iterations):
                self.train()

    def update_epsilon(self, epsilon=None):
        if epsilon is None:
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                      math.exp(-self.eps_lambda * self.timesteps_trained)
        print('updating epsilon for scope {}, new val is {} after {} timesteps\n'.format(self.scope, epsilon, self.timesteps_trained))
        self.set_explore_epsilon(epsilon)

    ## Mirror replay memory methods to make outer loop cleaner ##
    def store_obs(self, obs):
        # print('    {} storing_obs {}'.format(self.scope, obs))
        self.replay_memory.store_obs(obs)

    def get_action(self, obs, info=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        # obs_ = self.replay_memory.encode_recent_observation()
        # action = self.epsilon_greedy(np.array([obs]), epsilon, info)
        action = self.epsilon_softmax(np.array([obs]), epsilon, info)
        return action

    def store_effect(self, action, rew ,done):
        # print('    {} storing_effect {}, {}, {}'.format(self.scope, action, rew ,done))
        #TODO technically can find detrimental outcome and want to set found_rew to True, but we're subtracting stepcost
        #   so not a great way to check here
        if rew > 0:
            self.found_reward = True
        self.replay_memory.store_effect(action, rew, done)


    # Deprecated - moving logic to AgentCollection object to handle experience sharing logic
    # # TODO: develop smarter interface for use with rest of agent - for now simply assume we are
    # #       passed sole control of agent and env to train for N seconds
    # def train_for_time(self, time_to_train, ep_t_limit, render=False, reset=False):
    #     start = time.time()
    #     longest_ep_time = 0
    #
    #     # Don't have logic to handle tournament trial suddenly ending in the middle of training
    #     # Leaving a 10 seconds buffer over the longest previous training episode for each training ep
    #     while time_to_train - (time.time() - start) > longest_ep_time + 10:
    #         print("new inner loop ep, time left: {}".format(time_to_train - (time.time() - start)))
    #         ep_start_time = time.time()
    #         # TODO: need some notion of an episode, even if just for updates. Ideally we would want
    #         #       to explore for n timesteps, update network, reset to the initial state, explore for n more, etc.
    #         rew_sum, success, t = self.run_episode(ep_t_limit, render=render,reset=reset)
    #         # TODO: If we have extra time, we would ideally like to keep learning to achieve the subgoal
    #         #           even if we have already done so. (e.g. reset to different start state and run another episode)
    #         # For now if we manage to achieve subgoal, return control to rest of agent
    #         if success:
    #             # Achieved subgoal
    #             return True
    #         longest_ep_time = max(longest_ep_time, time.time()-ep_start_time)
    #         print(longest_ep_time)
    #
    #         #TODO: need func to reset agent to start state or more generally to some 'interesting' state
    #         #  for now we can see how updating and resuming from same state works
    #
    #     # Did not manage to complete subgoal
    #     return False

