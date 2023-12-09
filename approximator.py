import numpy as np


class Approximator:
    def __init__(self, env):
        state, info = env.reset()
        self.action_to_weights = {}
        self.state_space_size = len(state)
        self.weight_dimension = self.state_space_size + 1  # + 1 for bias term
        for action in range(env.action_space.n):
            self.action_to_weights[action] = np.random.rand(self.weight_dimension)

    """
    Approximator function. Given state and action, return action-value 
    """
    def value(self, state, action) -> float:
        weights = self.action_to_weights[action]
        return weights[0] + state.dot(weights[1:])

    def __getitem__(self, action):
        return self.action_to_weights[action]

    def __setitem__(self, action, new_weights):
        self.action_to_weights[action] = new_weights
