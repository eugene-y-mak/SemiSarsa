import numpy as np


class Policy:  # Given state, return action
    def __init__(self, env, approximator, eps):
        self.env = env
        self.approximator = approximator
        self.epsilon = eps

    """
    Chooses best action that maximizes q function
    """
    def __getitem__(self, state):  # uses Epsilon greedy
        best_action = None
        roll_dice = np.random.rand()
        if roll_dice < self.epsilon:
            return self.env.action_space.sample()
        best_action_value = -np.inf
        for action in range(self.env.action_space.n):
            action_value = self.approximator.value(state, action)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        return best_action

    # """
    # Updates policy with a new approximator. Thus, the behavior of getting the best action will change.
    # """
    # def update(self, approximator):
    #     self.approximator = approximator
