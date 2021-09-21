import math
import random

#Smart-exploration
MAX_EPSILON = 0.3

# #Epsilon-greedy
# MAX_EPSILON = 0.1

MAX_TIMESTEPS = 300
MAX_RHO = 0.4 
MIN_RHO = 0.10
EXPLORATION_STOP = 2000 #10000
SCORE_TO_CHECK = 950
NO_OF_SUCCESSFUL_DONE = 45

# remains same always
UPDATE_RATE = 10 # network weights update rate
MAX_EPISODES = 100000
EPS_TO_EVAL = 1
EVAL_INTERVAL = 50
NUM_HIDDEN = 24
GAMMA = 0.98
LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MIN_EPSILON = 0.05
random_seed = random.randint(0, 9)
print ("Random seed is ", random_seed)
# random_seed = 3
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay
PRINT_EVERY = 201 # logging
# convergence criteria
NO_OF_EPS_TO_CHECK = 70
NO_OF_DONES_TO_CHECK = 50