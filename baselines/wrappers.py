import gym

class RewardShaping(gym.RewardWrapper):
    """Add intermediate rewards for reaching useful subgoals."""
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return reward


class EpisodicWrapper(gym.Wrapper):
    """Terminate the episode and reset the environment after a number of timesteps has passed"""
    def __init__(self, env, timesteps_per_episode):
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.episode = 0
        self.timesteps_per_episode = timesteps_per_episode

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.timestep += 1

        if self.timestep == self.timesteps_per_episode - 1:
            done = True
        return next_state, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode += 1
        self.timestep = 0
        return obs
