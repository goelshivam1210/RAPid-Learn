import pickle
import os
import csv
from polycraft_tufts.rl_agent.dqn_lambda.learning.dqn import DQNLambda_Agent
import tensorflow as tf

# def save_results(agent, agent_type, methodid, limit_action, obs_rep, init_demo, num_eps, rewards_list, success_list, t_per_ep_list, plot=False):
#     if agent_type == 0:
#         agent_type_str = 'SarsaLambda'
#         data = [agent.weights, num_eps]
#     elif agent_type == 1:
#         agent_type_str = 'SimpleDQN'
#         data = [agent._model, num_eps]
#     elif agent_type == 2:
#         agent_type_str = 'DQNLambda'
#         data = None
#
#     # Save training results
#     os.makedirs("results" + os.sep + "data" + os.sep + "fenceNoveltyHardSparseShivam", exist_ok=True)
#     data_file_name = "results/data/fenceNoveltyHardSparseShivam/" + '_'.join([agent_type_str,str(methodid),str(limit_action),str(obs_rep),str(init_demo)]) + "_train_results.csv"
#     with open(data_file_name, 'a') as f:  # append to the file created
#         writer = csv.writer(f)
#         for i in range(num_eps):
#             # ep_num, ep_rew, ep_success, ep_t_num
#             # Can take sum while plotting if desired, keeping separate for now
#             data = [i, rewards_list[i], success_list[i], t_per_ep_list[i]]
#             writer.writerow(data)
#
#     if agent_type < 2:
#         # Save final model
#         # TODO: save model every N eps or with best SR rather than just at end
#         os.makedirs("results" + os.sep + "models" + os.sep + "fenceNoveltyHardSparseShivam", exist_ok=True)
#         model_file_name = "results/models/fenceNoveltyHardSparseShivam/" + '_'.join([agent_type_str,str(methodid),str(limit_action),str(obs_rep), str(init_demo)]) + "_trained_model.bin"
#         output_file = open(model_file_name, "wb")
#         pickle.dump(data, output_file)
#         output_file.close()

# Merging save_results with run_experiments
def save_results(savedir, trial_num, agent, agent_type, methodid, limit_action, obs_rep, init_demo, num_eps, rewards_list, success_list,
                 t_per_ep_list, plot=False):
    if agent_type == 0:
        agent_type_str = 'SarsaLambda'
        model_data = [agent.weights, num_eps]
    elif agent_type == 1:
        agent_type_str = 'SimpleDQN'
        model_data = [agent._model, num_eps]
    elif agent_type == 2:
        agent_type_str = 'DQNLambda'
        model_data = [agent, num_eps]
        # data = None

    os.makedirs(savedir + os.sep + 'data', exist_ok=True)
    data_file_name = savedir + '/data/trial_{}.csv'.format(trial_num)
    with open(data_file_name, 'w') as f:  # overwrite existing file if there (shouldn't be)
        writer = csv.writer(f)
        for i in range(num_eps):
            # ep_num, ep_rew, ep_success, ep_t_num
            # Can take sum while plotting if desired, keeping separate for now
            data = [i, rewards_list[i], success_list[i], t_per_ep_list[i]]
            writer.writerow(data)

    if agent_type < 2:
        # Save final model
        # TODO: save model every N eps or with best SR rather than just at end
        os.makedirs(savedir + os.sep + 'models', exist_ok=True)
        model_file_name = savedir + '/models/trained_model_trial_{}.bin'.format(trial_num)
        output_file = open(model_file_name, "wb")
        pickle.dump(model_data, output_file)
        output_file.close()
    else:
        os.makedirs(savedir + os.sep + 'models', exist_ok=True)
        model_file_dir = savedir + '/models/trained_model_trial_{}.ckpt'.format(trial_num)
        save_dqn_model(model_data[0], model_data[1], model_file_dir)

# data[0] is weights
# data[1] is num_timesteps trained so far
def load_model(model_path):
    bin_file = open(model_path, "rb")
    data = pickle.load(bin_file)
    return data[0], data[1]

def save_dqn_model(agent, num_timesteps, savefile):
    assert isinstance(agent, DQNLambda_Agent), "save_dqn_model can only be called with DQNLambda agent"
    # Get all variables including adam optimizer variables (to pick up training where we left off)
    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agent.scope)
    # Get only model weights needed to evaluation
    vars = tf.trainable_variables(scope=agent.scope)
    # for v in vars:
    #     print(agent.session.run(v))
    # In the case where we want to retrain in the face of a novelty if we decide to do so, do we want adam vars?
    saver = tf.train.Saver(vars)
    saver.save(agent.session, savefile, global_step=num_timesteps)

def load_dqn_model(savefile, params, env, seed=0):
    num_eps_trained = params['num_eps_trained']
    del params['num_eps_trained']
    # assert env observation and action space are the same
    agent = DQNLambda_Agent(seed=seed, env=env, **params)
    vars = tf.trainable_variables(scope=agent.scope)
    saver = tf.train.Saver(vars)
    saver.restore(agent.session, savefile + 'trained_model_trial_0.ckpt-{}'.format(num_eps_trained))
    # saver = tf.train.import_meta_graph(savedir + 'trained_model_trial_0.ckpt-{}'.format(num_eps_trained))
    # saver.restore(agent.session, tf.train.latest_checkpoint(savedir))
    vars = tf.trainable_variables(scope=agent.scope)
    # for v in vars:
    #     print(agent.session.run(v))
    return agent

    # agent = DQNLambda_Agent(seed=seed, env=env, return_est_method=RETURN_EST, replay_capacity=MEM_SIZE,
    #                         history_len=HISTORY_LEN, discount=GAMMA, cache_size=CACHE_SIZE, block_size=BLOCK_SIZE,
    #                         priority=PRIORITY, learning_rate=LEARNING_RATE, prepopulate=PREPOPULATE,
    #                         max_epsilon=MAX_EPSILON, batch_size=BATCH_SIZE, max_timesteps=NUM_TIMESTEPS,
    #                         session=session, hidden_size=NUM_HIDDEN, update_freq=UPDATE_FREQ)


# TODO incorporate demos more explicitly - had interleaved into run_experiment code requiring commenting things in and out
#   keep generation as separate notion of training execution but include loading and initialization
def save_demo(demo_dict, file_str):
    # ep_dict = run_experiment(env, agent_type, init_demo)
    # with open('demos/demo_methodid'+str(methodid)+'_obsrep'+str(obs_rep)+'.pkl','wb') as f:
    #     pickle.dump(ep_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(file_str,'wb') as f:
        pickle.dump(demo_dict, f, pickle.HIGHEST_PROTOCOL)

def load_demo(file_str):
    # with open('demos/demo_methodid' + str(methodid) + '_obsrep' + str(obs_rep) + '.pkl', 'rb') as f:
    #     demo_dict = pickle.load(f)
    with open(file_str, 'rb') as f:
        demo_dict = pickle.load(f)
    return demo_dict

def get_demo_dict(env, ep_t_limit=50):
    obss = []
    actions = []
    rews = []
    dones= []
    done = False
    env.reset()
    env.render()
    env.render()
    t_step = 0
    while not done:
        action = int(input("action"))
        obs, rew, done, info = env.step(action)
        env.render()
        # print(obs, rew, done, info)
        obss.append(obs)
        actions.append(action)
        rews.append(rew)
        dones.append(done)
        if t_step >= ep_t_limit:
            break
    ep_dict = {'obss': obss,
               'actions': actions,
               'rews': rews,
               'dones': dones}
    return ep_dict

# def run_demo_ep(demo_dict, agent):
#     if isinstance(agent, SimpleDQN):
#         for i in range(len(demo_dict['obss'])):
#             agent.process_step(demo_dict['obss'][i], True, True, demo_dict['actions'][i])
#             agent.give_reward(demo_dict['rews'][i])
#         agent.finish_episode()
#         agent.update_parameters()
#     else:
#         # # All we need in sarsa agent case is psis and td_errors
#         # for i in range(len(demo_dict['psis'])):
#         #     agent.update_eligibility((demo_dict['psis'][i]))
#         #     agent.update_weights(demo_dict['td_errors'][i])
#         # # obs = demo_dict['obss'][0]
#         # action = demo_dict['actions'][0]
#         # psi = demo_dict['psis'][0]
#         # # psi = agent.compute_psi(obs, None, action)
#         # for i in range(len(demo_dict['actions'])):
#         #     action_ = demo_dict['actions'][i+1]
#         #     psi_ = agent.compute_psi(demo_dict['obss'][i+1], None, action_, )
#         #     psi = agent.compute_psi(demo_dict['obss'][i], None, demo_dict['actions'][i])
#         #
#         #     agent.process_step(demo_dict['obss'][i], True, True, demo_dict['actions'][i])
#         #     agent.give_reward(demo_dict['rews'][i])
#         # agent.finish_episode()
#         # agent.update_parameters()
#         # print("SARSA agent demo_dict currently disabled")
#         print("demo_dict for any agent other than SimpleDQN currently disabled")
#         quit()