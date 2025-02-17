import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TextSequenceEnvironment:
    def __init__(self, sequences, embedder):
        self.sequences = sequences
        self.embedder = embedder
        self.n = len(sequences)
        self.current_state = list(range(self.n))

    def reset(self):
        random.shuffle(self.current_state)
        return self.current_state.copy()

    def step(self, action):
        i, j = action
        self.current_state[i], self.current_state[j] = self.current_state[j], self.current_state[i]
        reward = self._evaluate_sequence()
        done = self._is_terminal()
        return self.current_state, reward, done

    def _evaluate_sequence(self):
        embeddings = [self.embedder.embed(self.sequences[i]['caption']) 
                     for i in self.current_state]
        return sum(np.dot(embeddings[i], embeddings[i+1].T) 
                for i in range(len(embeddings)-1))

    def _is_terminal(self):
        return len(self.current_state) == self.n

    def valid_actions(self):
        return [(i, j) for i in range(len(self.current_state)) 
                        for j in range(i+1, len(self.current_state))]

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.memory = []
        self.batch_size = 32
        self.q_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay