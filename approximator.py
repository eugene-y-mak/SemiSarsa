import numpy as np


class Approximator:
    def __init__(self, env, bias=False, random_init=False):
        state, info = env.reset()
        self.action_to_weights = {}
        self.state_space_size = len(state)
        self.has_bias = bias
        if bias:
            self.weight_dimension = self.state_space_size + 1  # + 1 for bias term
        else:
            self.weight_dimension = self.state_space_size
        for action in range(env.action_space.n):
            # if random_init:
            #     self.action_to_weights[action] = np.random.rand(self.weight_dimension)
            # else:
            self.action_to_weights[action] = np.zeros(self.weight_dimension)

    """
    Approximator function. Given state and action, return action-value 
    """
    def value(self, state, action) -> float:
        weights = self.action_to_weights[action]
        if self.has_bias:
            return weights[0] + state.dot(weights[1:])
        else:
            return state.dot(weights)

    def __getitem__(self, action):
        return self.action_to_weights[action]

    def __setitem__(self, action, new_weights):
        self.action_to_weights[action] = new_weights
