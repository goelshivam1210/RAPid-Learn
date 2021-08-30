import math

# params that need changing
# MAX_EPSILON, MIN_RHO, MAX_RHO, MAX_TIMESTEPS, 
# SCORE_TO_CHECK, EXPLORATION_STOP, NO_OF_SUCCESSFUL_DONE

# MAX_TIMESTEPS = 300
# MAX_EPSILON = 0.70
# MAX_RHO = 0.80 
# MIN_RHO = 0.20
# EXPLORATION_STOP = 30000
# SCORE_TO_CHECK = 925
# NO_OF_SUCCESSFUL_DONE = 45

# # firecraftingtablehard # EPSILON_GREEDY
# MAX_TIMESTEPS = 500
# MAX_EPSILON = 0.3 # epsilon greedy
# MAX_RHO = 0.80 # doesnt matter 
# MIN_RHO = 0.20 # doesnt matter
# EXPLORATION_STOP = 50000
# SCORE_TO_CHECK = 900
# NO_OF_SUCCESSFUL_DONE = 45

# # firecraftingtablehard # SMART_EXPLORATION 16K 
MAX_TIMESTEPS = 150
MAX_EPSILON = 0.7
MAX_RHO = 0.7
MIN_RHO = 0.10
EXPLORATION_STOP = 20000
SCORE_TO_CHECK = 900
NO_OF_SUCCESSFUL_DONE = 45

# # axetobreakhard # EPSILON_GREEDY # ~ 11K episodes (eps = 0.3) 
# MAX_TIMESTEPS = 300
# MAX_EPSILON = 0.3
# MAX_RHO = 0.80 
# MIN_RHO = 0.20
# EXPLORATION_STOP = 30000
# SCORE_TO_CHECK = 950
# NO_OF_SUCCESSFUL_DONE = 45

# # axetobreakhard # SMART_EXPLORATION # ~8K episodes (0.5, 0.6)eps, rho ~ 6.5K (0.40, 0.40 eps rho)
# MAX_TIMESTEPS = 300
# MAX_EPSILON = 0.40
# MAX_RHO = 0.40 
# MIN_RHO = 0.10
# EXPLORATION_STOP = 20000
# SCORE_TO_CHECK = 950
# NO_OF_SUCCESSFUL_DONE = 45

# # axetobreakeasy # firecraftingtableeasy # FCTEASY eps = 0.2 ~700 episodes  In this why is the inventory tree_log > 1 inserted??
# MAX_TIMESTEPS = 300
# # MAX_EPSILON = 0.20
# MAX_EPSILON = 0.2 #0.2
# MAX_RHO = 0.80 
# MIN_RHO = 0.20
# EXPLORATION_STOP = 1000 #10000
# SCORE_TO_CHECK = 965
# NO_OF_SUCCESSFUL_DONE = 46


# # rubber_tree # EPS_GREEDY 
# MAX_TIMESTEPS = 300
# MAX_EPSILON = 0.20
# MAX_RHO = 0.80 
# MIN_RHO = 0.20
# EXPLORATION_STOP = 20000
# SCORE_TO_CHECK = 950
# NO_OF_SUCCESSFUL_DONE = 47

# # rubbertree # SMART_EXPLORATION
# MAX_TIMESTEPS = 300
# MAX_EPSILON = 0.70
# MAX_RHO = 0.80 
# MIN_RHO = 0.20
# EXPLORATION_STOP = 30000
# SCORE_TO_CHECK = 925
# NO_OF_SUCCESSFUL_DONE = 45


# remains same always
UPDATE_RATE = 10 # network weights update rate
MAX_EPISODES = 100000
EPS_TO_EVAL = 2
EVAL_INTERVAL = 400
NUM_HIDDEN = 16
GAMMA = 0.95
LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MIN_EPSILON = 0.05
# MAX_EPSILON = 0.70
random_seed = 2
# EXPLORATION_STOP = 30000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay
# MIN_RHO = 0.20 # constant for using guided policies.
# MAX_RHO = 0.80 
PRINT_EVERY = 201 # logging
# convergence criteria
NO_OF_EPS_TO_CHECK = 70
# SCORE_TO_CHECK = 925
NO_OF_DONES_TO_CHECK = 50
# NO_OF_SUCCESSFUL_DONE = 45