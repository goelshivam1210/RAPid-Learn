'''
Author: Shivam Goel
Email: goelshivam1210@gmail.com
'''





import time
import numpy as np
import math
import tensorflow as tf
import os


from simulate_tournament import NoveltyRecoveryAgent


class Learner:
    def __init__(self, failed_action, env, novelty_flag=False) -> None:
        self.encounter_novelty_flag = novelty_flag
        self.env = env
        self.failed_action = failed_action


        self.novelty_recovery = NoveltyRecoveryAgent(self.env, encountered_novelty= self.encounter_novelty_flag)

    def learn_state(self):
        self.novelty_recovery.create_learning_agent()

    def learn_policy(self):
        pass


if __name__ == '__main__':
    failed_action = None
    # env =  

    learn = Learner()


    pass




        
