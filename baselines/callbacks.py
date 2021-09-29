from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEvalCallback(BaseCallback):
    """
    Run evaluation every n episodes
    """

    def __init__(self, evaluate_every_n: int, n_eval_episodes: int, render):
        super(CustomEvalCallback, self).__init__()
        self.evaluate_every_n = evaluate_every_n
        self.n_episodes = 0
        self.n_eval_episodes = n_eval_episodes
        self.render = render

    def _on_step(self) -> bool:
        # Check if enough episodes have passed to trigger an evaluation. Switch flags for monitor back appropriately
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        self.n_episodes += np.sum(done_array).item()
        if self.n_episodes > self.evaluate_every_n:
            self.training_env.metadata['mode'] = 'learn-postnovelty-test'
            evaluate_policy(self.model, self.training_env, n_eval_episodes=self.n_eval_episodes, render=self.render,
                            deterministic=False)
            self.training_env.metadata['mode'] = 'learn-postnovelty-train'
            self.n_episodes = 0
        return True
