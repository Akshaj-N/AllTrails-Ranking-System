# Must do pip install 'gymnasium[atari,accept-rom-license]'
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torchvision.transforms as T
from tqdm import trange

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEM_SIZE = 100_000
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000

# Preprocessing transform
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84)),
    T.ToTensor()
])

# Define CNN Q-network
class CNNQNet(nn.Module):
    def __init__(self, action_dim):
        super(CNNQNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(states),
                torch.tensor(actions, dtype=torch.long).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# Process frame
def process_obs(obs):
    obs = np.array(obs)
    obs = transform(obs)
    return obs

# DQN Agent
class DQNAgent:
    def __init__(self, action_dim):
        self.q_net = CNNQNet(action_dim).to(device)
        self.target_net = CNNQNet(action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(MEM_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0).to(device))
            return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones) * GAMMA * max_next_q_values
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# Training loop
def train_dqn(num_episodes=500):
    env = gym.make("ALE/Backgammon-v5", render_mode="rgb_array")
    agent = DQNAgent(action_dim=env.action_space.n)

    for episode in trange(num_episodes):
        obs, _ = env.reset()
        state = process_obs(obs).to(device)
        total_reward = 0

        for _ in range(10000):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = process_obs(next_obs).to(device)

            agent.replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode} — Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train_dqn()

