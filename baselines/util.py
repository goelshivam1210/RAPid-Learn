def get_difference_in_obs_action_space(novelty_name):
    d_obs_inventory, d_obs_lidar, d_actions = 0, 0, 0

    if novelty_name == 'axetobreakeasy' or novelty_name == 'axetobreakhard':
        d_obs_lidar = 1
        d_obs_inventory = 1
        d_actions = 1
    elif novelty_name == 'firecraftingtableeasy' or novelty_name == 'firecraftingtablehard':
        d_obs_lidar = 1
        d_obs_inventory = 1
        d_actions = 2
    elif novelty_name == 'rubbertree':
        d_obs_lidar = 1
    elif novelty_name == 'axefirecteasy':
        d_obs_lidar = 2
        d_obs_inventory = 2
        d_actions = 3
    elif novelty_name == 'scrapeplank':
        d_actions = 1

    return d_obs_inventory, d_obs_lidar, d_actions