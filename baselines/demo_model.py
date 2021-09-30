import argparse

import gym
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from SimpleDQN import SimpleDQN
from baselines.util import get_difference_in_obs_action_space
from baselines.wrappers import StatePlaceholderWrapper, ActionPlaceholderWrapper, EpisodicWrapper
from params import *

ENV_ID = 'NovelGridworld-Pogostick-v1'
N_PLACEHOLDERS_INVENTORY = 2
N_PLACEHOLDERS_LIDAR = 2
N_PLACEHOLDERS_ACTIONS = 5


class Demo:
    def __init__(self, args):
        self.novelty_name = args["novelty_name"] if args["novelty_name"] else "prenovelty"
        self.algorithm = args["algorithm"]
        self.model_path = args["model_path"]

        self.n_episodes = args["n_episodes"]


        self.env = gym.make(ENV_ID)

        if self.novelty_name != "prenovelty":
            self.env = inject_novelty(self.env, self.novelty_name)

        d_obs_inventory, d_obs_lidar, d_actions = get_difference_in_obs_action_space(self.novelty_name)
        self.env = StatePlaceholderWrapper(self.env,
                                           n_placeholders_inventory=N_PLACEHOLDERS_INVENTORY - d_obs_inventory,
                                           n_placeholders_lidar=N_PLACEHOLDERS_LIDAR - d_obs_lidar)
        self.env = ActionPlaceholderWrapper(self.env, n_placeholders_actions=N_PLACEHOLDERS_ACTIONS - d_actions)

        # Environment wrappers
        self.env = EpisodicWrapper(self.env, 500, verbose=True)
        check_env(self.env, warn=True)

    def run(self):
        if self.algorithm == "PPO":
            self._run_ppo_demo()
        elif self.algorithm == "policy_gradient":
            self._run_policy_gradient_demo()


    def _run_ppo_demo(self):
        model = PPO.load(self.model_path, self.env)
        evaluate_policy(model, self.env, self.n_episodes, deterministic=False, render=True)

    def _run_policy_gradient_demo(self):
        model = SimpleDQN(int(self.env.action_space.n), int(self.env.observation_space.shape[0]),
                               NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON,
                               False, {}, self.env.actions_id, random_seed, "guided_policy", "ucb")

        model.load_model("", "", path_to_load=self.model_path)
        model._explore_eps = MIN_EPSILON
        model.reset()

        obs = self.env.reset()
        info = self.env.get_info()
        done = False
        timestep = 0

        while not done:
            action = model.process_step(obs, True, timestep)
            timestep += 1
            obs, rew, done, info = self.env.step(action)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/luk/repos/RAPid-Learn/data/76d952deb2e84aa38356fbca4ec3cda5-2021-09-30_21:10:41-PPO-10episodes-rewardshapingon/prenovelty/trial-0/model_prenovelty.zip")
    ap.add_argument("--algorithm", default="PPO", type=str)
    ap.add_argument("-N", "--novelty_name", default=None,
                    help="Novelty to inject: #axetobreakeasy #axetobreakhard #firecraftingtableeasy #firecraftingtablehard #rubbertree #axefirecteasy",
                    type=str)
    ap.add_argument("--n_episodes", default=5, type=int)

    args = vars(ap.parse_args())

    demo = Demo(args)
    demo.run()
