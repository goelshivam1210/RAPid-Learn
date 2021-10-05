from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from params import NO_OF_DONES_TO_CHECK, NO_OF_SUCCESSFUL_DONE


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
        self.success_buffer = []

    def _on_step(self) -> bool:
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        if self.evaluate_every_n > 0 and self.n_eval_episodes > 0:
            # Check if enough episodes have passed to trigger an evaluation. Switch flags for monitor back appropriately
            self.n_episodes += np.sum(done_array).item()
            if self.n_episodes > self.evaluate_every_n:
                self.training_env.metadata['mode'] = 'eval'
                evaluate_policy(self.model, self.training_env, n_eval_episodes=self.n_eval_episodes, render=self.render,
                                deterministic=False)
                self.training_env.metadata['mode'] = 'learn'
                self.n_episodes = 0
        if done_array[-1]:
            self.success_buffer.append(self.locals["infos"][-1]["success"])
            running_success_mean = np.sum(self.success_buffer[-NO_OF_DONES_TO_CHECK:])
            if running_success_mean >= NO_OF_SUCCESSFUL_DONE:
                print(f"Agent converged with {running_success_mean}% success.")
                return False

        return True

