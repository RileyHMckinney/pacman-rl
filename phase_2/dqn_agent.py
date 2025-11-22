import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import config


class DQN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Epsilon parameters are all taken from config
        self.epsilon = config.EPSILON_START
        self.eps_min = config.EPSILON_MIN
        self.eps_decay = config.EPSILON_DECAY

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = None

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q = self.model(s)
        return int(q.argmax().item())

    def update(self, batch_size):
        """
        Perform one gradient update step from replay buffer.
        NOTE: epsilon is NOT decayed here anymore; it is decayed once per episode in the training loop.
        """
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch

        states      = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions     = torch.tensor(actions).long().to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s, a)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values using target network
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # No epsilon decay here anymore
        return loss.item()

    def update_target(self):
        """Sync target network with online network."""
        self.target.load_state_dict(self.model.state_dict())
