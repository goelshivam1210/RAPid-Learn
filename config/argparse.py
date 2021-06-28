import math
import argparse

# Used by DQNLambda
def intfloat(value):
    '''Allows an int argument to be formatted as a float.'''
    if float(value) != int(float(value)):
        raise argparse.ArgumentError
    return int(float(value))

ap = argparse.ArgumentParser()
ap.add_argument("-N", "--num_timesteps", default=500000, type=int, required=False, help="Number of Timesteps to train agent for")
ap.add_argument("-T", "--ep_t_limit", default = 50, type=int, required = False, help="Timstep limit for each episode")
ap.add_argument("-H", "--num_hidden", default=32,type=int, required=False, help="Hidden layer size")
ap.add_argument("-G", "--gamma", default=0.99, type=float, required=False, help="Discount Factor")
ap.add_argument("-A", "--learning_rate", default=0.001, type=float, required=False, help="Initial learning Rate")
ap.add_argument("-E", "--max_epsilon", default=0.8, type=float, required=False, help="Max Epsilon Value")
ap.add_argument("-M", "--min_epsilon", default=0.1, type=float, required=False, help="Min Epsilon Value")
ap.add_argument("-F", "--exploration_fraction", default=1.0, type=float, required=False, help="Fraction of training to decay epsilon over")
ap.add_argument("-R", "--exploration_decay_rate", default=0.01, type=float, required=False, help="Speed of epsilon decay")
ap.add_argument("-V", "--agent_view_size", default=2, type=int, required=False, help="Agent View size for agentmap")
# DQNLambda specific
ap.add_argument('--batch-size', type=int, default=32,
                    help='(int) Minibatch size for training. Default: 32')
ap.add_argument('--block-size', type=intfloat, default=100,
                    help='(int) Refresh the cache using sequences of this length. Cannot use with --legacy. Default: 100')
ap.add_argument('--cache-size', type=intfloat, default='80e3',
                    help='(int) Capacity of the cache. Cannot use with --legacy. Default: 80e3')
ap.add_argument('--grad-clip', type=float, default=40.0,
                    help='(float) Max magnitude for each gradient component. Default: 40.0')
ap.add_argument('--history-len', type=int, default=1,
                    help='(int) Number of recent observations fed to Q-network. Default: 1') #4
ap.add_argument('--mem-size', type=intfloat, default='1e6',
                    help='(int) Capacity of the replay memory. Default: 1e6')
ap.add_argument('--prepopulate', type=intfloat, default='50e3',
                    help='(int) Initialize replay memory with random policy for this many timesteps. Default: 50e3')
ap.add_argument('--priority', type=float, default=0.0,
                    help='(float) Extent to which cache samples are prioritized by TD error. Must be in [0, 1]. '
                         'High value may degrade performance. Cannot use with --legacy. Default: 0.0')
ap.add_argument('--return-est', type=str, default='watkins-0.99',
                    help="(str) Estimator used to compute returns. See README. Default: 'watkins-0.99'")
ap.add_argument('--update-freq', type=intfloat, default='10e3',
                    help='(int) Frequency of cache update (or target network update, with --legacy). Default: 10e3')

args = vars(ap.parse_args())
# print("args: ", args)

NUM_TIMESTEPS = args['num_timesteps']
EP_T_LIMIT = args['ep_t_limit']
NUM_HIDDEN = args['num_hidden']
GAMMA = args['gamma']
LEARNING_RATE = args['learning_rate']
MAX_EPSILON = args['max_epsilon']
MIN_EPSILON = args['min_epsilon']
EXPLORATION_STOP = NUM_TIMESTEPS * args['exploration_fraction']
EPS_LAMBDA = -math.log(args['exploration_decay_rate']) / EXPLORATION_STOP  # speed of decay #Explore almost entire time
AGENT_VIEW_SIZE = args['agent_view_size']
# DQNLambda args
BATCH_SIZE = args['batch_size']
BLOCK_SIZE = args['block_size']
CACHE_SIZE = args['cache_size']
GRAD_CLIP = args['grad_clip']
HISTORY_LEN = args['history_len']
MEM_SIZE = args['mem_size']
PREPOPULATE = args['prepopulate']
PRIORITY = args['priority']
RETURN_EST = args['return_est']
# Want to make based of off episodes or timesteps?
UPDATE_FREQ = args['update_freq']# / float(EP_T_LIMIT)
