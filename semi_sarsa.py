import numpy as np
from approximator import Approximator
from policy import Policy
import gymnasium as gym


class SemiSarsa:
    def __init__(self, env, n_step=1, alpha=0.4, epsilon=0.2, gamma=0.9):
        self.env = env
        self.n_step = n_step
        self.store_size = self.n_step + 1
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.approximator = Approximator(self.env)  # action-value function
        self.policy = Policy(self.env, self.approximator, self.epsilon)

    def run(self):
        num_episodes = 0
        while num_episodes < 1000:  # per episode
            episode_reward = 0
            initial_state, _ = self.env.reset()
            state_dimension = len(initial_state)
            current_action = self.policy[initial_state]
            T = np.inf
            t = 0
            tau = None
            actions = np.empty(self.n_step + 1)
            rewards = np.empty(self.n_step + 1)
            states = np.empty([self.n_step + 1, state_dimension])
            states[0 % self.store_size] = initial_state
            actions[0 % self.store_size] = current_action
            while tau != T - 1:  # exit if tau = T - 1
                if t < T:
                    next_state, reward, terminated, truncated, info = self.env.step(current_action)
                    episode_reward += reward
                    rewards[t + 1 % self.store_size] = reward
                    states[t + 1 % self.store_size] = next_state
                    if terminated:
                        T = t + 1
                    else:
                        actions[t + 1 % self.store_size] = self.policy[initial_state]
                tau = t - self.n_step + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n_step, T)):
                        G += self.gamma ** (i - tau - 1) * rewards[i % self.store_size]
                    if tau + self.n_step < T:
                        state = states[tau + self.n_step % self.store_size]
                        action = actions[tau + self.n_step % self.store_size]
                        G += self.gamma ** self.n_step * self.approximator.value(state=state, action=action)
                    tau_action = actions[tau % self.store_size]
                    tau_state = states[tau % self.store_size]
                    current_action_value = self.approximator.value(state=tau_state, action=tau_action)
                    self.approximator[tau_action] += self.alpha * (G - current_action_value) * \
                                                                         np.append(tau_state, 1)
                    # updates policy with new approximator
                    self.policy.approximator = self.approximator
            num_episodes += 1
            print(f'episode: {num_episodes}, reward: {episode_reward}')
            print(self.approximator.action_to_weights)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")  # , render_mode='human'
    sarsa_model = SemiSarsa(env)
    print("Start!")
    sarsa_model.run()
