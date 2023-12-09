import numpy as np
from approximator import Approximator
from policy import Policy
import gymnasium as gym


class SemiSarsa:
    def __init__(self, environment, n_step=8, alpha=.3/8, epsilon=0.01, gamma=1):
        self.env = environment
        self.n_step = n_step
        self.store_size = self.n_step + 1
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.approximator = Approximator(self.env)  # action-value function
        self.policy = Policy(self.env, self.approximator, self.epsilon)

    def run(self):
        num_episodes = 0
        while num_episodes < 500:  # per episode
            episode_reward = 0
            current_state, _ = self.env.reset()
            state_dimension = len(current_state)
            current_action = self.policy[current_state]
            T = np.inf
            t = 0
            actions = np.zeros(self.store_size)
            rewards = np.zeros(self.store_size)
            states = np.zeros([self.store_size, state_dimension])
            states[0 % self.store_size] = current_state
            actions[0 % self.store_size] = current_action
            while True:  # exit if tau = T - 1
                if t < T:
                    current_state, reward, terminated, truncated, info = self.env.step(current_action)
                    episode_reward += reward
                    rewards[(t + 1) % self.store_size] = reward
                    states[(t + 1) % self.store_size] = current_state
                    if terminated:
                        T = t + 1
                    else:
                        current_action = self.policy[current_state]
                        actions[(t + 1) % self.store_size] = current_action
                tau = t - self.n_step + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n_step, T) + 1):
                        G += self.gamma ** (i - tau - 1) * rewards[i % self.store_size]
                    if (tau + self.n_step) < T:
                        state = states[(tau + self.n_step) % self.store_size]
                        action = actions[(tau + self.n_step) % self.store_size]
                        G += self.gamma ** self.n_step * self.approximator.value(state=state, action=action)
                    tau_action = actions[tau % self.store_size]
                    tau_state = states[tau % self.store_size]
                    current_action_value = self.approximator.value(state=tau_state, action=tau_action)
                    #print(f"Return: {G}")
                    #print(f"current_action_value: {current_action_value}")
                    self.approximator[tau_action] += self.alpha * (G - current_action_value) * tau_state #np.insert(tau_state, 0, 1) #tau_state  #

                    # updates policy with new approximator
                    self.policy.approximator = self.approximator
                    #print(self.policy.approximator.action_to_weights)
                if tau == T - 1:
                    break
                t += 1
            num_episodes += 1
            print(f'episode: {num_episodes}, reward: {episode_reward}')
            #print(self.approximator.action_to_weights)


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    #env = gym.make("CartPole-v1", render_mode='human')
    #env = gym.make("Acrobot-v1", render_mode='human')
    #env = gym.make("ALE/Breakout-v5")
    sarsa_model = SemiSarsa(env)
    print("Start")
    sarsa_model.run()
