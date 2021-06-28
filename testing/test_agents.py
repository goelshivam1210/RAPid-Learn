import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
import math
# from learning.sarsa_lambda_agent import SARSALAMBDA_Agent
# from learning.simple_dqn import SimpleDQN
from polycraft_tufts.rl_agent.dqn_lambda.learning.dqn import DQNLambda_Agent
import os
import pickle
import time
from polycraft_tufts.rl_agent.dqn_lambda.learning.utils import make_session
from wrappers import FenceExperimentWrapper
from utils import save_results, load_dqn_model
from plotting import plot_results
from config import *
import sys

# def run_experiment(env, agent_type, demo_dict=None, num_timesteps=1000000, ep_t_limit=50, seed=10, session=None):
def run_experiment(env, agent_type, seed=10, session=None):

    start = time.time()

    # Most params moved to config.py
    actionCnt = env.action_space.n
    D = env.observation_space.shape[0]

    # Removing testing for other agents
    # Create agent
    # if agent_type == 0:
    #     agent = SARSALAMBDA_Agent(None, actionCnt, GAMMA, LEARNING_RATE, LAMBDA_TRACE, D, env)
    # elif agent_type == 1:
    #     agent = SimpleDQN(actionCnt, D, NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON, seed)
    # elif agent_type == 2:
    if agent_type == 2:
        # agent = DQNLambda_Agent(seed=seed, env=env, return_est_method='watkins-0.99', replay_capacity=NUM_TIMESTEPS, history_len=1, discount=GAMMA, cache_size=80000, block_size=100, priority=0.0, learning_rate=LEARNING_RATE, prepopulate=50000, max_epsilon=MAX_EPSILON, batch_size=32, max_timesteps=NUM_TIMESTEPS, session=session, hidden_size=NUM_HIDDEN)
        agent = DQNLambda_Agent(seed=seed, env=env, return_est_method=RETURN_EST, replay_capacity=MEM_SIZE, history_len=HISTORY_LEN, discount=GAMMA, cache_size=CACHE_SIZE, block_size=BLOCK_SIZE, priority=PRIORITY, learning_rate=LEARNING_RATE, prepopulate=PREPOPULATE, max_epsilon=MAX_EPSILON, batch_size=BATCH_SIZE, max_timesteps=NUM_TIMESTEPS, session=session, hidden_size=NUM_HIDDEN, update_freq=UPDATE_FREQ)
    else:
        print("ERROR: Agent type must be 2, other agents disabled")
        quit()

    # Run training loop
    ep_num = 0
    total_t_step = 0
    rewards_list = []
    success_list = []
    t_per_ep_list = []
    log_every = 100

    # TODO: add evaluate at intervals within training loop
    # Train until num_timesteps is exceeded
    while total_t_step < NUM_TIMESTEPS:

        # set eps for episode
        # Edited to be off of timestep rather than episode number since episodes may be variable length
        #   and we want to train based off of number of timesteps, not episodes
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
                  math.exp(-EPS_LAMBDA * total_t_step)
                  # math.exp(-EPS_LAMBDA * ep_num)
        agent.set_explore_epsilon(epsilon)

        #Run agent for an episode and log stats
        rew, success, ep_t = agent.run_episode(env, EP_T_LIMIT)
        ep_num += 1
        total_t_step += ep_t
        rewards_list.append(rew)
        success_list.append(success)
        t_per_ep_list.append(ep_t)
        if ep_num % log_every == 0:
            print("experiment time after {} eps, {} timesteps: ".format(ep_num, total_t_step), time.time() - start)

            # Run 10 evaluate trials every log interval
            # for i in range(10):
            #     input("waiting to start eval trial {}".format(i))
            #     rew_eval, success_eval, ep_t_eval = agent.run_evaluate_episode(env, 50, True)
            #     print("evaluate trial {} after {} episodes training: rew = {}, success = {}, num_t = {}".format(i,ep_num,rew_eval,success_eval,ep_t_eval))

    print("experiment time: ", time.time() - start)

    env.close()
    # Must be a better way to do this
    # if agent_type == 2:
    #     agent.reset_graph()

    # return stats for saving
    return agent, ep_num, rewards_list, success_list, t_per_ep_list

# TODO: FIX - looks like issue isn't just with dqn-lambda, subsequent runs of anything take longer
#    changing so that each thing is run once and writing bash script to run everything
# Issue fixed with differentiating tf scopes, but haven't transitioned back from bach script
def run_experiments(agent_types, methodids, limit_actions, obs_reps, init_demos, trial_num, num_trials=1, num_timesteps=500000, config_num=0):
    # config_num = 0

    # Having issues with tensorflow computational complexity increasing between trials, limit to 1 for now
    # assert num_trials == 1, "ERROR: having issues with computational complexity of multiple trials in single execution, please set num_trials to 1"
    assert True not in init_demos, "ERROR: Demos currently disabled, please set init_demos to False"

    for agent_type in agent_types:
        # TODO: can eliminate if using bash script
        if agent_type == 2:
            # Make seed == trial_num <- needs to be passed in upfront because can't run multiple trials within single ex
            session = make_session(int(trial_num))
        else:
            session = None
        for methodid in methodids:
            for limit_action in limit_actions:
                for obs_rep in obs_reps:
                    for init_demo in init_demos:
                        print("Starting config ", config_num, ": ", agent_type, " ", methodid, " ", limit_action, " ",
                              obs_rep, " ", init_demo)
                        # TODO: reimplement, need to get rid of for now since running everything individually
                        # Keep results for all trials of specific config in one place and plot together
                        # Current Day and Time
                        # stamp = time.strftime("%m_%d_%I%M", time.localtime())
                        # savedir = '/home/dev/tufts/polycraft_exploration/results/{}'.format(stamp)
                        # savedir = '/home/dev/tufts/polycraft_exploration/results/{}_{}_{}'.format(agent_type,methodid,limit_action)
                        # TEMP: extending savedir to include split based off of specific hyperparams
                        # How many do we include to be distinct? Differentiate based on type?
                        # Rather than edit dir based on params, put all params in file within config num
                        # savedir = '/home/dev/tufts/polycraft_exploration/results/{}_{}_{}/config_{}'.format(agent_type,methodid,limit_action, config_num)
                        savedir = 'results/{}_{}_{}/config_{}'.format(agent_type,methodid,limit_action, config_num)
                        for trial in range(num_trials):
                            print("config {}, trial {}".format(config_num, trial))
                            # Make env
                            env = gym.make('NovelGridworld-v6')
                            env = inject_novelty(env, 'fence', 'hard', 'oak')
                            env = FenceExperimentWrapper(env, methodid, limit_action, obs_rep, agent_view_size=AGENT_VIEW_SIZE, num_beams=NUM_BEAMS)
                            # Run experiment
                            agent, num_eps, rewards_list, success_list, t_per_ep_list = run_experiment(env, agent_type, seed=int(trial), session=session)
                            # agent, num_eps, rewards_list, success_list, t_per_ep_list = run_experiment(env, agent_type, seed=int(trial_num), session=session)
                            # save results
                            save_results(savedir, trial, agent, agent_type, methodid, limit_action, obs_rep, init_demo, num_eps, rewards_list, success_list, t_per_ep_list, plot=True)
                        save_param_dict(savedir, methodid, limit_action, obs_rep, num_eps)
                        # plot results
                        plot_results([savedir+os.sep+'data/'], title_str="{}_{}_{}_{}_{}".format(agent_type,methodid,str(limit_actions),obs_rep,init_demo), num_timesteps=NUM_TIMESTEPS)

                        config_num += 1

import json
# TODO: num_eps_trained is variable across trials
def save_param_dict(savedir, methodid, limit_action, obs_rep, num_eps_trained):
    param_dict = {
        'max_timesteps': NUM_TIMESTEPS,
        'ep_t_limit': EP_T_LIMIT,
        'hidden_size': NUM_HIDDEN,
        'discount': GAMMA,
        'learning_rate': LEARNING_RATE,
        # 'decay_rate': DECAY_RATE,
        # 'use_demos': USE_DEMOS,
        'max_epsilon': MAX_EPSILON,
        'min_epsilon': MIN_EPSILON,
        'exploration_stop': EXPLORATION_STOP,
        'eps_lambda': EPS_LAMBDA,
        # 'agent_view_size': AGENT_VIEW_SIZE,
        # 'num_beams': NUM_BEAMS,
        'batch_size': BATCH_SIZE,
        'block_size': BLOCK_SIZE,
        'cache_size': CACHE_SIZE,
        'grad_clip': GRAD_CLIP,
        'history_len': HISTORY_LEN,
        'replay_capacity': MEM_SIZE,
        'prepopulate': PREPOPULATE,
        'priority': PRIORITY,
        'return_est_method': RETURN_EST,
        'update_freq': UPDATE_FREQ,
        'env_methodid': methodid,
        'env_limit_action': limit_action,
        'env_obs_rep': obs_rep,
        'env_agent_view_size': AGENT_VIEW_SIZE,
        'env_num_lidar_beams': NUM_BEAMS,
        'num_eps_trained': num_eps_trained,
    }
    with open(savedir+os.sep+'params.pkl', 'wb') as f:
        pickle.dump(param_dict, f)
    with open(savedir+os.sep+'params.txt', 'w') as f:
        f.write(json.dumps(param_dict))

# Only currently set up for DQNLambda agent
# Should we differentiate between a trainable and non-trainable agent? So we don't have to provide params for
#   or rebuild the replay_mem or train_funcs?
def test_trained_agent(load_path, env=None, num_eps=10, ep_t_limit=50, render=True):
    with open(load_path+os.sep+'params.pkl', 'rb') as f:
        params = pickle.load(f)

    if env is None:
        env = gym.make('NovelGridworld-v6')
        env = inject_novelty(env, 'fence', 'hard', 'oak')
        env = FenceExperimentWrapper(env, int(params['env_methodid']), params['env_limit_action'], params['env_obs_rep'], agent_view_size=params['env_agent_view_size'],
                                     num_beams=params['env_num_lidar_beams'])

    # Get rid of params not necessary for agent
    # Or explicitly pass in correct params
    keys_to_del = []
    for key in params.keys():
        if key.startswith('env'):
            keys_to_del.append(key)
            # del params[key]
    for key in keys_to_del:
        del params[key]
    del params['ep_t_limit']
    del params['min_epsilon']
    del params['exploration_stop']
    del params['eps_lambda']

    # assumes load path contains params.pkl file saved from function above and
    #   models file containing a trained_model_trial_0.meta
    agent = load_dqn_model(load_path+os.sep+'models'+os.sep, params, env)

    for i in range(num_eps):
        input("waiting to start eval trial {}".format(i))
        rew_eval, success_eval, ep_t_eval = agent.run_evaluate_episode(env, ep_t_limit, render)
        print("evaluate trial {}: rew = {}, success = {}, num_t = {}".format(i, rew_eval, success_eval, ep_t_eval))


# TODO: extend both plotting and experiments to use config rather than values within functions
#   Extend both to allow for hyperparameter search + merging of multiple trials
if __name__ == '__main__':
    # Experiment params
    # SARSALambda, SimpleDQN, DQNLambda
    # agent_types = [0,1,2]
    # # lidar, agent_map, both, fulllidar, fullidaragentmap
    # methodids = [0,1,2,3,4]
    # # All actions or only L,R,F,Break
    # limit_actions = [False, True]
    # # Only map or include selected_item and inventory
    # obs_reps = [0,1]
    # # Start by giving experience from single demo trajectory or not
    # init_demos = [0,1]
    # num_trials == 1

    # agent_types = [int(sys.argv[1])]
    # methodids = [int(sys.argv[2])]
    # limit_action = sys.argv[3] == 'True'
    # limit_actions = [limit_action]
    # obs_reps = [int(sys.argv[4])]
    # init_demos = [int(sys.argv[5])]
    # trial_num = sys.argv[6]
    # # Temp: add additional layer of discretion to plotting and saving results based on hyperparams.
    # #   Specifically make subfolder based on hyperparams within training params folders.
    # config_num = sys.argv[7]

    # Hyperparams set up in config through argparser

    # run_experiments(agent_types, methodids, limit_actions, obs_reps, init_demos, trial_num, num_trials=1, num_timesteps=NUM_TIMESTEPS)
    # run_plotting()

    run_experiments(AGENT_TYPES, METHODIS, LIMIT_ACTIONS, OBS_REP, INIT_DEMOS, TRIAL_NUM, config_num=CONFIG_NUM, num_trials=1, num_timesteps=NUM_TIMESTEPS)
    # test_trained_agent('/home/dev/tufts/polycraft_exploration/results/2_1_True/config_23')