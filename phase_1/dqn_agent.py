# phase_1/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon=1.0, eps_min=0.05, eps_decay=0.999):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = None  # to be attached externally

    # --------------------------------------------------------
    def select_action(self, state_vec):
        """Epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    # --------------------------------------------------------
    def update(self, batch_size):
        """Runs one gradient update from replay buffer."""
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # --------------------------------------------------------
    def decay_epsilon(self):
        """Decay epsilon once per episode, not per gradient step."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # --------------------------------------------------------
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
