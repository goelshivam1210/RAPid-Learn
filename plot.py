import matplotlib.pyplot as plt
import numpy
import matplotlib as mpl
from numpy.core.fromnumeric import std
import seaborn as sns
import glob
import os
import csv
import numpy as np
from params import EPS_TO_EVAL
sns.set()
# we need plots for all the xperiments
# each plot will plot the episodes versus rewards 
# for episodes in learning phase, we need to average all the n runs, and hence get one data point for each stop
# for episodes in prenovelty and post novelty. we will simply plot the reward for every episode..
# We also need to average out the 10 trials.


# One folder for each novelty
# Within the folder, 10 folders (with hash id) (2 files - one for train and one for test) for each trial of that novelty
# Get the test file and load it ['Mode', 'Episode_no', 'Trial_no', 'Timesteps', 'Reward', 'Done']
# Pre novelty eps: Eps vs Reward (Mean for 10 trials), Eps vs Done (Mean for 10 trials)
# During learning: For each episode - mean the trials and then plot eps vs Reward (Mean for 10 trials) and Eps vs Done (Mean for 10 trials)
# Post novelty: Eps vs Reward (Mean for 10 trials) and Eps vs Done (Mean for 10 trials)

novelty_name = 'axetobreakeasy' # give from args
learning_type = 'epsilon-greedy'
data_dir = 'data/*'
type_file = 'test'

if __name__ == '__main__':  
    files = []
    for name in glob.glob(data_dir):
        print ("name = ", name)
        name_split = (name.split("_"))
        if name_split[1] == novelty_name and name_split[2] == learning_type: # axetobreakhard_smart-exploration_5923c3d4f30347baa07ceab346579a58
            files.append(name)
    # for test
    print ("files", files)

    Final_data = []
    max_eps = []    
    max_index = []
    eval_interval = 0
    for file_no in range(len(files)):
        Data = []
        print("File")
        with open((files[file_no]+os.sep+type_file+"results.csv"), 'r') as f: # open one file
            reader = csv.reader(f)
            lines = f.readlines()
            index = 0
            for line_number in range(len(lines)): # 400,0,334,-334,0 # Mode, Episode_no, Trial_no, Timesteps, Reward, Done
                data = []
                line = (lines[line_number].strip()).split(",")
                if line[0] == 'Mode':
                    continue
                if line:
                    print("Line:", line)
                    index += 1
                    mode = int(line[0])
                    if mode == 0:
                        data.append(float(line[0])) # Append the mode
                        data.append(float(line[2])) # Append the pre-novelty trials no 
                        data.append(float(line[3])) # Append the timesteps taken
                        data.append(float(line[4])) # Append the reward received
                        data.append(float(line[5])) # Append the done status
                    elif mode == 1: # learning mode
                        data.append(float(line[0])) # Append the mode
                        # here we try to find out the maximum number of episodes the trial took to converge
                        # try:
                        # next_line = next(reader)
                        next_line = lines[line_number+1].split(",")
                        print("Line: ", line)
                        print("next line: ", next_line)
                        if int(next_line[0]) == 1:
                            eval_interval = float(next_line[1]) - float(line[1])
                        if int(next_line[0]) == 2:
                            max_eps.append(float(line[1]))
                            max_index.append(index)
                        data.append(float(line[1])) # Append the learning episode no 
                        data.append(float(line[3])) # Append the learning mean timesteps
                        data.append(float(line[4])) # Append the learning mean reward 
                        data.append(float(line[5])) # Append the learning mean done  
                    elif mode == 2: # post novelty trials
                        data.append(float(line[0])) # Append the mode
                        data.append(float(line[1]) + float(line[2])) # Append the post-learning episode no 
                        data.append(float(line[3])) # Append the post-learning mean timesteps
                        data.append(float(line[4])) # Append the post-learning mean reward 
                        data.append(float(line[5])) # Append the post-learning mean done  
                Data.append(np.asarray(data)) 
            # print("DAta is:", data)
        Final_data.append(np.asarray(Data))

    Final_data = np.asarray(Final_data)
    print("max_eps: ", max_eps)
    max_episodes_for_convergence = max(max_eps)
    highest_index = max_eps.index(max_episodes_for_convergence)
    print ("max_index = ", max_index)
    print("max_eps for convegence: ", max_episodes_for_convergence)
    print ("eval_interval", eval_interval)
    
    dummy_array = np.zeros((Final_data.shape[0], Final_data[highest_index].shape[0], Final_data[0].shape[-1]))
    dummy_array[highest_index] = Final_data[highest_index]
    for count, trial_data in enumerate(Final_data): # count -> which file, trial data -> File's data
        to_insert_list = []
        if count is not highest_index and max_eps[count] < max_episodes_for_convergence:
            print("count: ", count)
            for episode in range(int(max_eps[count]+eval_interval), int(max_episodes_for_convergence+eval_interval), int(eval_interval)):
                to_insert_list.append([1, episode, np.mean(trial_data[int(max_index[count])-5:int(max_index[count]), 2]), \
                np.mean(trial_data[int(max_index[count])-5:int(max_index[count]), 3]), np.mean(trial_data[int(max_index[count])-5:int(max_index[count]), 4])])
            to_insert_list = np.asarray(to_insert_list)
            print("to insert list:", to_insert_list.shape)
            temp = np.insert(Final_data[count], max_index[count], to_insert_list, 0)
            for ind in range(int(max_index[count]), int(Final_data[highest_index].shape[0])):
                temp[ind][1] = Final_data[highest_index][ind][1]
            dummy_array[count] = temp

    # with open('data/foo.txt', 'w') as outfile:
    #     for slice_2d in dummy_array:
    #         np.savetxt(outfile, slice_2d)

    # np.savetxt("data/foo.csv", dummy_array, delimiter=",")



    # get all the trials
    # using the max 
    # # insert in all the arrays the mean value of the last 50 episodes.

    # for count, trial_data in enumerate(Final_data): # count -> which file, trial data -> File's data
    #     if count is not highest_index:
    #         while True:
    #             np.


    print ("final data shape = ",Final_data[1].shape)
    print ("dummy array = ", np.array_str(dummy_array, precision=2, suppress_small=True))
            # plot_eps.append(np.asarray(episodes_pre))
            # plot_timesteps.append(np.asarray(time))

# find the means and STDs
mean_rewards = np.mean(dummy_array[:,:,3], axis = 0)
std_reward = np.std(dummy_array[:,:,3], axis = 0)
mpl.style.use('seaborn')
sns.set(font_scale=2.3) 
sns.set_style("white")
fig, ax = plt.subplots(figsize=(12,7.5))
# print("Episodes: ", Final_data[0,:,0])
# print("Reward: ", Final_data[0,:,2])


ax.plot(dummy_array[0,:,1] ,mean_rewards, label="Eps vs Reward", linewidth = 2.5, antialiased=True, color='b')
ax.fill_between(dummy_array[0,:,1], mean_rewards-std_reward, mean_rewards+std_reward, alpha=0.2, antialiased=True, color='b')
plt.xlabel('Episodes')
plt.ylabel('Average rewards')
plt.legend()
plt.tight_layout()
plt.show()

# plt.show()


    # for eps in range(max(len(episodes_pre[i]) for i in range(len(episodes_pre)))): # episodes_pre: [[1,2,3],[1,2,3,4]] # Rewards_pre [[900,910,920],[870,880,890,900]]
    #     plot_eps.append(eps) # plt_eps: [1,2,3,4]
    #     reward_pre.append() # reward_pre: [885,895,900,910]
 

# '''

# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import glob
# import seaborn as sns
# # %matplotlib inline
# sns.set()

# plt.style.use('seaborn-whitegrid')
# # sns.despine()
# # print (sys.argv[1])
# window_size = int(sys.argv[1])

# # level-1 spotter
# def parser(directory, col):

#     all_data = []
#     for name in glob.glob(directory): 
#         data = []
#         with open(name, mode = 'r') as infile:
#             reader = csv.reader(infile)
#             for line in reader:
#                 reward = float(line[col])
#                 data.append(reward)
#         # data = np.array(data)
#         all_data.append(np.asarray(data))
#         #print (all_data)
#     all_data = np.asarray(all_data)
#     # print (all_data.shape)
#     return all_data

# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# def get_top_vals(a, n):
#     M = a.shape[0]
#     perc = (np.arange(M-n,M)+1.0)/M*100
#     return np.percentile(a,perc)


# if __name__ == '__main__':
#     # spotter
#     dir_lvl1 = 'data_from_cluster/run7/spotter/MiniGrid-SpotterLevel1-v0/*.csv'
#     dir_lvl2 = 'data_from_cluster/run7/spotter/MiniGrid-SpotterLevel2-v0/*.csv'
#     dir_lvl3 = 'data_from_cluster/run7/spotter/MiniGrid-SpotterLevel3-v0/*.csv'
#     # vanilla_Q
#     dir_vanilla_q_lvl_1 = 'data_from_cluster/baseline_run2/data_spotter_baselines[i]/baseline/MiniGrid-SpotterLevel1-v0/vanilla_baselines/*.csv'
#     dir_vanilla_q_lvl_2 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel2-v0/vanilla_baselines/*.csv'
#     dir_vanilla_q_lvl_3 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel3-v0/vanilla_baselines/*.csv'
#     #augmneted_Q
#     dir_augmented_q_lvl_1 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel1-v0/augmented_simple/*.csv'
#     dir_augmented_q_lvl_2 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel2-v0/augmented_simple/*.csv'
#     dir_augmented_q_lvl_3 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel3-v0/augmented_simple/*.csv'
#     # augmented_lower_level_Q
#     dir_augmented_lower_q_1 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel1-v0/augmented_lower_level/*.csv'
#     dir_augmented_lower_q_2 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel2-v0/augmented_lower_level/*.csv'
#     dir_augmented_lower_q_3 = 'data_from_cluster/baseline_run2/data_spotter_baselines/baseline/MiniGrid-SpotterLevel3-v0/augmented_lower_level/*.csv'


#     #########################SPOTTER###########################################################
    
#     # get the raw
#     spotter_1_raw = parser(dir_lvl1, 1)
#     spotter_2_raw = parser(dir_lvl2, 1)
#     spotter_3_raw = parser(dir_lvl3, 1)

#     # value for normalizing 
#     sp_norm_1 = np.mean(get_top_vals(np.mean(spotter_1_raw, axis = 0)[:-1], 5000))
#     sp_norm_2 = np.mean(get_top_vals(np.mean(spotter_2_raw, axis = 0)[:-1], 7000))
#     sp_norm_3 = np.mean(get_top_vals(np.mean(spotter_3_raw, axis = 0)[:-1], 5000))

#     # normalize all spotters
#     spotter_1 = spotter_1_raw/sp_norm_1
#     spotter_2 = spotter_2_raw/sp_norm_2
#     spotter_3 = spotter_3_raw/sp_norm_3

#     # compute means
#     spotter_1_mean =  np.mean(spotter_1, axis = 0)[:-1]
#     spotter_2_mean =  np.mean(spotter_2, axis = 0)[:-1]
#     spotter_3_mean =  np.mean(spotter_3, axis = 0)[:-1]

#     # compute stds
#     spotter_1_std =  np.std(spotter_1, axis = 0)[:-1]
#     spotter_2_std =  np.std(spotter_2, axis = 0)[:-1]
#     spotter_3_std =  np.std(spotter_3, axis = 0)[:-1]

#    #########################################################################################
#    ####################################### Vanillla Q learning ##############################

#     vanilla_q_1 = parser(dir_vanilla_q_lvl_1, 2)/sp_norm_1
#     vanilla_q_2 = parser(dir_vanilla_q_lvl_2, 2)/sp_norm_2
#     vanilla_q_3 = parser(dir_vanilla_q_lvl_3, 2)/sp_norm_3

#     vanilla_q_1_std = np.std(vanilla_q_1, axis = 0)
#     vanilla_q_2_std = np.std(vanilla_q_2, axis = 0)
#     vanilla_q_3_std = np.std(vanilla_q_3, axis = 0)

#     vanilla_q_1_mean = np.mean(vanilla_q_1, axis = 0)
#     vanilla_q_2_mean = np.mean(vanilla_q_2, axis = 0)
#     vanilla_q_3_mean = np.mean(vanilla_q_3, axis = 0)

#     ####################################### HLA QL learning ######################################

#     aug_q_1 = parser(dir_augmented_q_lvl_1, 2)/sp_norm_1
#     aug_q_1_std = np.std(aug_q_1, axis = 0)
#     aug_q_1_mean = np.mean(aug_q_1, axis = 0)

#     aug_q_2 = parser(dir_augmented_q_lvl_2, 2)/sp_norm_2
#     aug_q_2_std = np.std(aug_q_2, axis = 0)
#     aug_q_2_mean = np.mean(aug_q_2, axis = 0)
    
#     aug_q_3 = parser(dir_augmented_q_lvl_3, 2)/sp_norm_3
#     aug_q_3_std = np.std(aug_q_3, axis = 0)
#     aug_q_3_mean = np.mean(aug_q_3, axis = 0)

#     #################### Lower level Q#####################################################

#     aug_lower_q_1 = parser(dir_augmented_lower_q_1, 2)/sp_norm_1
#     aug_lower_q_1_std = np.std(aug_lower_q_1, axis = 0)
#     aug_lower_q_1_mean = np.mean(aug_lower_q_1, axis = 0)

#     aug_lower_q_2 = parser(dir_augmented_lower_q_2, 2)/sp_norm_2
#     aug_lower_q_2_std = np.std(aug_lower_q_2, axis = 0)
#     aug_lower_q_2_mean = np.mean(aug_lower_q_2, axis = 0)

#     aug_lower_q_3 = parser(dir_augmented_lower_q_3, 2)/sp_norm_3
#     aug_lower_q_3_std = np.std(aug_lower_q_3, axis = 0)
#     aug_lower_q_3_mean = np.mean(aug_lower_q_3, axis = 0)

#     ##########################################################################################

#     # grab all the datas into one arrays
#     spotter = np.concatenate((spotter_1_mean, spotter_2_mean, spotter_3_mean))
#     vanilla_q = np.concatenate((vanilla_q_1_mean, vanilla_q_2_mean, vanilla_q_3_mean))
#     aug_q = np.concatenate((aug_q_1_mean, aug_q_2_mean, aug_q_3_mean))
#     aug_lower_q = np.concatenate((aug_lower_q_1_mean, aug_lower_q_2_mean, aug_lower_q_3_mean))

#     # grab all the stds
#     spotter_std = np.concatenate((spotter_1_std, spotter_2_std, spotter_3_std))
#     vanilla_q_std = np.concatenate((vanilla_q_1_std, vanilla_q_2_std, vanilla_q_3_std))
#     aug_q_std = np.concatenate((aug_q_1_std, aug_q_2_std, aug_q_3_std))
#     aug_lower_q_std = np.concatenate((aug_lower_q_1_std, aug_lower_q_2_std, aug_lower_q_3_std)) 

#     # compute pos and negs
#     spotter_std_pos = spotter+spotter_std
#     spotter_std_neg = spotter-spotter_std

#     vanilla_q_std_pos = vanilla_q + vanilla_q_std
#     aug_q_std_pos = aug_q + aug_q_std
#     aug_lower_q_std_pos = aug_lower_q + aug_lower_q_std

#     vanilla_q_std_neg = vanilla_q - vanilla_q_std
#     aug_q_std_neg = aug_q - aug_q_std
#     aug_lower_q_std_neg = aug_lower_q - aug_lower_q_std

#     ################################################################################################

#     # calculate moving averages

#     window_size_qlearning = 50
#     # compute the sliding window average
#     spotter_ma = moving_average(spotter, window_size)
#     vanilla_q_ma = moving_average(vanilla_q, window_size_qlearning)
#     aug_q_ma = moving_average(aug_q, window_size_qlearning)
#     aug_lower_q_ma = moving_average(aug_lower_q, window_size_qlearning)

#     ################################################################################################


#     plt.plot(spotter_ma, label = 'SPOTTER')
#     plt.fill(spotter_std_pos, color = 'aliceblue')
#     plt.fill(spotter_std_neg, color = 'aliceblue')

#     plt.plot(vanilla_q_ma, label = 'V.Q.L.', color = 'forestgreen')
#     plt.fill(vanilla_q_std_pos, color = 'honeydew')
#     plt.fill(vanilla_q_std_neg, color = 'honeydew')

#     plt.plot(aug_q_ma, label = 'H.L.A.Q.L.', color = 'indianred')
#     plt.fill(aug_q_std_pos, color = 'lavenderblush')
#     plt.fill(aug_q_std_neg, color = 'lavenderblush')

#     plt.plot(aug_lower_q_ma, label = 'H.L.A.L.Q.L.', color = 'mediumpurple')
#     plt.fill(aug_lower_q_std_pos, color = 'lavender')
#     plt.fill(aug_lower_q_std_neg, color = 'lavender')

#     plt.xlabel("Number of episodes")
#     plt.ylabel("Average cumulative reward per episode")
#     plt.grid(True)
#     plt.legend(loc = 7)
#     plt.yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
#     plt.xlim(xmin=0.0, xmax=40000)
#     # plt.ylim(ymin=0.0)
#     sns.set_style("ticks")
#     fs = 3.0

#     ax = plt.gca()
#     plt.axvline(x = 10000, linewidth=3.0, color='k')
#     plt.axvline(x = 30000, linewidth=3.0, color='k')

#     # plt.suptitle('Level 1',fontsize=12, y = 1.5, x = 10000)
#     plt.text(4000, 0.85, 'Level-1', horizontalalignment='center', verticalalignment='center', fontsize = 10)
#     plt.text(21000, 0.85, 'Level-2', horizontalalignment='center', verticalalignment='center', fontsize = 10)
#     plt.text(35000, 0.85, 'Level-3', horizontalalignment='center', verticalalignment='center', fontsize = 10)

#     plt.tight_layout()
#     plt.savefig('final_plot_baselines_spotter_stds_600_WG.png', dpi = 600)
#     # plt.savefig('final_plot_baselines_spotter_stds_600.png', dpi = 600)
#     # plt.savefig('final_plot_baselines_spotter_stds_300.png', dpi = 300)
#     # plt.savefig('final_plot_baselines_spotter_stds_900.png', dpi = 900)

# '''