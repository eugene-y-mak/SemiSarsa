import gymnasium as gym
from semi_sarsa import SemiSarsa
env = gym.make("CartPole-v1") # , render_mode='human'

# Get number of actions from gym action space
# noinspection PyUnresolvedReferences
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

sarsa_model = SemiSarsa(env)
print("Start!")
sarsa_model.run()
# for episode in range(n_episodes):
#     episode_reward = 0
#     state, _ = env.reset()
#     terminated = False
#     time_step_max = 500
#     time_step = 0
#     print("Starting episode")
#     while not terminated and time_step < time_step_max:
#         action = policy(state)
#         state, reward, terminated, truncated, info = env.step(action)
#         # print(next_state, reward, done, truncated, info)
#         episode_reward += reward
#         time_step += 1
#         print(time_step)
#     if time_step == time_step_max:
#         print("Successfully reached max time!")
#     print(f'episode: {episode + 1}, reward: {episode_reward}')

