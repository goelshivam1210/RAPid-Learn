def get_difference_in_obs_action_space(novelty_name):
    d_obs_inventory, d_obs_lidar, d_actions = 0, 0, 0

    if novelty_name == 'axetobreakeasy':
        # Axe is given as a lidar obs (even though it isn't needed)
        d_obs_lidar = 1

        # Axe is in inventory
        d_obs_inventory = 1

        # can now select
        d_actions = 1

    elif novelty_name == 'axetobreakhard':
        # axe item
        d_obs_lidar = 1
        d_obs_inventory = 1

        # select axe and approach axe
        d_actions = 2

    elif novelty_name == 'firecraftingtableeasy':
        # water item
        d_obs_lidar = 1
        d_obs_inventory = 1

        # select water and spray
        d_actions = 2

    elif novelty_name == 'firecraftingtablehard':
        # water item
        d_obs_lidar = 1
        d_obs_inventory = 1

        # select water, spray and approach water
        d_actions = 3

    elif novelty_name == 'rubbertree':
        # rubber tree
        d_obs_lidar = 1

        # rubber tree log
        d_obs_inventory = 1

        # approach rubber tree
        d_actions = 1

    elif novelty_name == 'axefirecteasy':
        # axe & water
        d_obs_lidar = 2
        d_obs_inventory = 2

        # select axe, approach axe, spray water, approach water, select water
        d_actions = 5

    elif novelty_name == 'scrapeplank':
        # new scrapeplank action
        d_actions = 1

    return d_obs_inventory, d_obs_lidar, d_actions