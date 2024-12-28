import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
ALPHA = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 10
N_EPISODES = 300

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=ALPHA)

replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)

def select_action(state):
    if random.random() < EPSILON:
        return random.choice(range(action_size))
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state)
        return torch.argmax(q_values).item()

def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0]
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

episode_rewards = []
for episode in range(N_EPISODES):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        train()
        total_reward += reward
    episode_rewards.append(total_reward)
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.load_state_dict(model.state_dict())
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{N_EPISODES} finished with reward: {total_reward}")

print("Testing the trained agent...")
state = env.reset()[0]
done = False
total_reward = 0
while not done:
    action = select_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    total_reward += reward

print(f"Total reward during testing: {total_reward}")

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve (Training Progress)')
plt.show()
