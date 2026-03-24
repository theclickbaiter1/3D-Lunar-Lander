import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic

# Force CPU. MPS is known to cause training anomalies/divergence in some RL algorithms.
device = torch.device("cpu")
print(f"Using device: {device}")

class PrioritizedReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, alpha=0.6):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dead = np.zeros((max_size, 1), dtype=np.float32)
        
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.alpha = alpha
        
    def add(self, state, action, reward, next_state, dead):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead
        
        self.priorities[self.ptr] = max_prio
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size, beta=0.4):
        if self.size == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return (
            torch.FloatTensor(self.state[indices]),
            torch.FloatTensor(self.action[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.dead[indices]),
            indices,
            torch.FloatTensor(weights).unsqueeze(1)
        )
        
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-5

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if noise != 0: 
            action = action + np.random.normal(0, noise, size=action.shape[0])
            
        return np.clip(action, -0.99, 0.99) # Clip slightly inside bounds to prevent gradients from zeroing out when tanh saturates completely

    def train(self, replay_buffer, batch_size=64, beta=0.4):
        state, action, reward, next_state, dead, indices, weights = replay_buffer.sample(batch_size, beta)
        
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        dead = dead.to(device)
        weights = weights.to(device)

        # Compute target Q
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - dead) * 0.99 * target_Q).detach()

        # Compute critic loss
        current_Q = self.critic(state, action)
        td_errors = target_Q - current_Q
        critic_loss = (weights * (td_errors ** 2)).mean()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            
        # Update PER priorities
        prios = np.abs(td_errors.cpu().data.numpy().flatten())
        replay_buffer.update_priorities(indices, prios)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
