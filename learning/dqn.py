import gym
import numpy as np
import tensorflow as tf
import time
import math
import os

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
        self.saver = tf.train.Saver()

        def epsilon_greedy(state, epsilon, info=None):
            if np.random.random() < epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = self.session.run(greedy_actions, feed_dict={state_ph: state[None]})[0]
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

        this_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

        assert len(other_vars) == len(this_vars), 'ERROR: TF variables have mismatched lengths between scopes we are trying to copy between'
        for i in range(len(other_vars)):
            self.session.run(this_vars[i].assign(self.session.run(other_vars[i])))

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
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            reward_sum += reward
            t += 1
            if t >= ep_t_limit:
                done = True
                success = False

            if done:
                return reward_sum, success, t

    def check_update(self):
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
        self.replay_memory.store_obs(obs)

    def get_action(self, obs, info=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        action = self.epsilon_softmax(np.array([obs]), epsilon, info)
        return action

    def store_effect(self, action, rew ,done):
        if rew > 0:
            self.found_reward = True
        self.replay_memory.store_effect(action, rew, done)
    
    def save_model(self, novelty_name, operator_name):
        model_dir = "Policy_models"+os.sep+novelty_name+"_"+operator_name
        os.makedirs(model_dir,exist_ok=True)
        self.saver.save(self.session, model_dir+os.sep+novelty_name+"_"+operator_name)

    def load_model(self,novelty_name, operator_name):
        model_dir = "Policy_models"+os.sep+novelty_name+"_"+operator_name+os.sep
        # print ("model_dire v= {}".format(model_dir))
        self.saver.restore(self.session, tf.train.latest_checkpoint(model_dir))

