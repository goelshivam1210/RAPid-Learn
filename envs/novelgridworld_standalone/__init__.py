from polycraft_tufts.rl_agent.dqn_lambda.envs.novelgridworld_standalone.pogostick_v1_env import PogostickV2Env
from gym.envs.registration import register

register(
    id='NovelGridworld-Pogostick-v2',
    entry_point='polycraft_tufts.rl_agent.dqn_lambda.envs.novelgridworld_standalone.pogostick_v1_env:PogostickV2Env',
)