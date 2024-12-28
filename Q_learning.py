import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode='human')

n_bins = 24
bins = [np.linspace(-x, x, n_bins) for x in [4.8, 5.0, 0.418, 5.0]]

Q_table = np.zeros([n_bins] * 4 + [env.action_space.n])

def discretize(state):
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discrete_state)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 300
total_rewards = []

for episode in range(num_episodes):
    done = False
    total_reward = 0
    state, info = env.reset()
    state = discretize(state)

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize(next_state)

        max_future_q = np.max(Q_table[next_state])
        Q_table[state + (action,)] += alpha * (reward + gamma * max_future_q - Q_table[state + (action,)])

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    total_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes} finished with reward: {total_reward}")

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards During Training')
plt.show()

state, info = env.reset()
state = discretize(state)
done = False
total_reward = 0

print("Testing the trained agent...")

while not done:
    action = np.argmax(Q_table[state])
    next_state, reward, done, truncated, info = env.step(action)
    next_state = discretize(next_state)
    total_reward += reward
    state = next_state
    env.render()

print(f"Total reward during testing: {total_reward}")

env.close()
