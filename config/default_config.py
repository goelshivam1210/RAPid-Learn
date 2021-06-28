default_param_dict = {
    'return_est_method': 'pengs-median',
    'replay_capacity': 10000,
    'history_len': 1,
    'discount': 0.99,
    'cache_size': 5000,
    'block_size': 100,
    'priority': 0,
    'learning_rate': 0.001,
    'prepopulate': 250,
    'max_epsilon': 0.8,
    'min_epsilon': 0.2,
    'eps_lambda': None,
    'batch_size': 32,
    'max_timesteps': 50000,
    # 'session': None,
    'hidden_size': 32,
    'update_freq': 50, #1,
}

# import math

# NUM_TIMESTEPS = 10000
# EP_T_LIMIT = 50
# NUM_HIDDEN = 32
# GAMMA = 0.99
# LEARNING_RATE = .001
# MAX_EPSILON = 0.8
# MIN_EPSILON = 0.1
# EXPLORATION_STOP = NUM_TIMESTEPS * 1.0
# EPS_LAMBDA = -math.log(0.01) / EXPLORATION_STOP  # speed of decay #Explore almost entire time
# AGENT_VIEW_SIZE = 2
# # DQNLambda args
# BATCH_SIZE = 32
# BLOCK_SIZE = 100
# CACHE_SIZE = 5000
# GRAD_CLIP = 40.0
# HISTORY_LEN = 1
# MEM_SIZE = 50000
# PREPOPULATE = 500
# PRIORITY = 0
# RETURN_EST = 'pengs-median'
# UPDATE_FREQ = 1
