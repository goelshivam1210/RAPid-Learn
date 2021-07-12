# author: Shivam Goel
# contact: shivam.goel@tufts.edu

# This class contains the implementation of the operator generalization component of the 
# RapidL architecture

class OperatorGeneralizaion:
    def __init__(self, action, policy) -> None:

        self.failed_action = action 
        self.learned_policy = policy

    def get_old_action_preconditions(self):
        '''
        pass old action object or dict

        '''
        preconds = self.failed_action
        return preconds
        # pass

    def get_old_action_effects(self):

        return self.failed_action
        # pass

    def policy_to_action(self):
        '''

        main function converts the given learned policy to the 
        operator with preconditions and effects

        ### I/P: policy  
        ### O/P: action to be added to the PDDL domain.
        '''
        # start with the existing action's preconditions and effects
        failed_action_preconds = self.get_old_action_preconditions()
        failed_action_effects = self.get_old_action_effects()
        new_action_name = self.generate_action_name(failed_action_preconds, failed_action_effects)
        new_action_preconds = self.generate_preconds(failed_action_preconds, failed_action_effects)
        new_action_effects = self.generate_effects(failed_action_preconds, failed_action_effects)

        new_action_pddl = self.pddl_parser(new_action_name, new_action_preconds, new_action_effects)
        
        return new_action_pddl

    def generate_action_name (self):
        pass

    def generate_preconds (self):
        pass

    def generate_effects (self):

        pass

    # yash can write this.
    def pddl_parser(self):
        pass

# current approach:
'''
# Take in the pre conditions of the failed operator, sample the state representation vectors of the same state from the game, and then
# compare it with the symbolic representation of the similar states, probably label them. Then given a state representation ask the function to
# ]
'''