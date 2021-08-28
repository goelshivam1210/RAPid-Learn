import math

UPDATE_RATE = 10 # network weights update rate
MAX_TIMESTEPS = 150
MAX_EPISODES = 100000
EPS_TO_EVAL = 3
EVAL_INTERVAL = 100
NUM_HIDDEN = 16
GAMMA = 0.95
LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MIN_EPSILON = 0.05
MAX_EPSILON = 0.7
random_seed = 2
EXPLORATION_STOP = 30000
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay
MIN_RHO = 0.2 # constant for using guided policies.
MAX_RHO = 0.80 
PRINT_EVERY = 201