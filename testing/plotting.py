import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Changing to fit with run_experiments
# Only set up now to plot all results within single directory - Need to change if we want to plot comparisons again
def plot_results(results_dirs, title_str = None, label_strs = None, num_timesteps=500000, interval=50, save_name='plot'):
    srs_per = 10
    plt.figure(figsize=(8, 4))
    if title_str is not None:
        plt.title(title_str)
    plt.ylim = (0.0, 1.0)
    plt.ylabel('Success Rate Past {} Episodes'.format(srs_per))
    plt.xlabel('Timesteps')

    i = 0
    for results_dir in results_dirs:
        # Need way to average over trials when episodes between trials will have different timestep lengths
        # Can just plot closest to every 50 timesteps even if not exact?
        total_success_rates_per_100 = []
        for results_path in os.listdir(results_dir):
            successes = []
            num_successes = 0
            success_rates = []
            success_rates_per_100 = []
            ep_ts = []
            sum_ep_ts = [0]
            with open(results_dir + os.sep + results_path) as f:
                reader = csv.reader(f)
                count = 0.0
                for row in reader:
                    count += 1
                    # track overall success rate and success rate past 100 episodes
                    # success = bool(row[2])
                    # successes.append(int(success))
                    success = 1 if row[2] == 'True' else 0
                    successes.append(success)
                    if success:
                        num_successes += 1
                    success_rates.append(num_successes/count)
                    if count > srs_per-1:
                        success_rates_per_100.append(sum(successes[-srs_per:])/float(srs_per))
                    ep_ts.append(int(row[3]))
                    sum_ep_ts.append(sum_ep_ts[-1] + int(row[3]))
            filtered_success_rates_per_100 = []
            init_step = 0
            x_count = 1

            # Filter success_rates_per_100 such that we have consistent x intervals between trials
            i = 0
            while True:
                target = init_step + interval * x_count
                if target > num_timesteps-interval:
                    break
                if np.abs(sum_ep_ts[i] - target) < np.abs(sum_ep_ts[i + 1] - target):
                    if x_count < srs_per:
                        filtered_success_rates_per_100.append(sum(successes[:x_count])/float(x_count))
                    else:
                        filtered_success_rates_per_100.append(success_rates_per_100[i-srs_per])
                    x_count += 1
                else:
                    i += 1
            # TODO: fix alg, just manually doing for now
            filtered_success_rates_per_100.append(success_rates_per_100[i-srs_per])
            new_ep_ts = np.arange(init_step,num_timesteps+interval, interval)
            total_success_rates_per_100.append(filtered_success_rates_per_100)

        # avg and plot
        avg_success_rates_per_100 = np.mean(np.asarray(total_success_rates_per_100), axis=0)
        if label_strs is not None:
            plt.plot(new_ep_ts[1:], avg_success_rates_per_100, label=label_strs[i])
            i+=1
        else:
            plt.plot(new_ep_ts[1:], avg_success_rates_per_100)

        # plot success rate in past 100 episodes at timestep intervals
        plt.legend()
        # plt.savefig('results/plots/' + '_'.join(title_str.split(', ')) + '.png')
        # plt.savefig(results_dir + os.sep +'..' + os.sep + 'plot.png')
        plt.savefig(results_dir + os.sep +'..' + os.sep + save_name + '_per_{}.png'.format(srs_per))
        plt.close()
        # plt.show()

# # Change so results_paths is NxM - where n is number of different lines and M is trials to avg over for each line
# # Change results_paths to results_dirs?
# def plot_results(results_paths, title_str = None, label_strs = None):
#     plt.figure(figsize=(8, 4))
#     if title_str is not None:
#         plt.title(title_str)
#     plt.ylim = (0.0, 1.0)
#     plt.ylabel('Success Rate Past 100 Episodes')
#     plt.xlabel('Timesteps')
#
#     i = 0
#     for results_path in results_paths:
#         # REMOVE
#         # results_path = 'results/data/fenceNoveltyHardSparse3/' + results_path
#         rews = []
#         running_rew = None
#         reward_sum = 0
#         running_rews = []
#         successes = []
#         num_successes = 0
#         success_rates = []
#         success_rates_per_100 = []
#         ep_ts = []
#         sum_ep_ts = [0]
#         with open(results_path) as f:
#             reader = csv.reader(f)
#             count = 0.0
#             for row in reader:
#                 count += 1
#                 # Track individual rewards and running_reward
#                 rew = int(row[1])
#                 reward_sum += rew
#                 running_rew = reward_sum if running_rew is None else running_rew * 0.95 + reward_sum * 0.05
#                 rews.append(rew)
#                 running_rews.append(running_rew)
#
#                 # track overall success rate and success rate past 100 episodes
#                 # success = bool(row[2])
#                 # successes.append(int(success))
#                 success = 1 if row[2] == 'True' else 0
#                 successes.append(success)
#                 if success:
#                     num_successes += 1
#                 success_rates.append(num_successes/count)
#                 if count > 99:
#                     success_rates_per_100.append(sum(successes[-100:])/100.)
#                 ep_ts.append(int(row[3]))
#                 sum_ep_ts.append(sum_ep_ts[-1] + int(row[3]))
#         sum_ep_ts = sum_ep_ts[1:]
#         ep_iterations = range(0, len(ep_ts), 1)
#         if label_strs is not None:
#             plt.plot(sum_ep_ts[99:], success_rates_per_100, label=label_strs[i])
#             i+=1
#         else:
#             plt.plot(sum_ep_ts[99:], success_rates_per_100)
#
#     # # plot running rew as a function of episodes
#     # plt.plot(ep_iterations, running_rews)
#     # plt.ylabel('Average Return')
#     # plt.xlabel('Iterations')
#     # # plt.savefig('temp_plot_name.png')
#     # plt.show()
#
#     # # plot overall success rate against timesteps spent training
#     # plt.plot(sum_ep_ts, success_rates)
#     # plt.ylabel('Overall Success Rate')
#     # plt.xlabel('Timesteps')
#     # plt.show()
#
#     # plot success rate in past 100 episodes at timestep intervals
#     # Not quite right
#     plt.legend()
#     plt.savefig('results/plots/' + '_'.join(title_str.split(', ')) + '.png')
#     plt.close()
#     # plt.show()


def get_label_str_from_filename(results_file, relevant_param_ids = None):
    if relevant_param_ids is None:
        relevant_param_ids = [0,1,2,3,4]
    params = results_file.split('_')
    alg_str = params[0]
    if params[1] == '0':
        map_str = 'lidar'
    elif params[1] == '1':
        map_str = 'agentmap'
    elif params[1] == '2':
        map_str = 'lidar_agentmap'
    elif params[1] == '3':
        map_str = 'lidarfull'
    elif params[1] == '4':
        map_str = 'lidar_relcoords'
    elif params[1] == '5':
        map_str = 'agentmap_lidarfull_relcoords'
    else:
        print("invalid methodid in get_label_str")
        quit()
    if params[2] == 'True':
        action_str = 'limited actions'
    else:
        action_str = 'all actions'
    if params[3] == '0':
        obs_str = 'only map obs'
    else:
        obs_str = 'full obs'
    if params[4] == '0':
        demo_str = 'no demo'
    else:
        demo_str = 'single demo'
    param_strs = np.array([alg_str, map_str, action_str, obs_str, demo_str])
    label_str = ', '.join(param_strs[relevant_param_ids])
    return label_str

# if __name__ == '__main__':
#     config_nums = ['23'] #['1','2','3','4']
#     num_timesteps = [10000]
#     for config_num in config_nums:
#         for num_timestep in num_timesteps:
#             plot_results(['/home/dev/tufts/polycraft_exploration/results/2_1_True/config_{}/data/'.format(config_num)],
#                          title_str="{}_{}_{}_{}_{}".format(2, 1, 'True', 0, 0),
#                          num_timesteps=num_timestep, save_name='plot_{}'.format(str(num_timestep)))
