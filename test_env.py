import torch
from mujoco_env import LunarLander3DEnv
from agent import DDPG, PrioritizedReplayBuffer

print("Testing initialization...")
env = LunarLander3DEnv()
obs, _ = env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPG(state_dim, action_dim, 1.0)
buffer = PrioritizedReplayBuffer(1000, state_dim, action_dim)

print("Testing environment step...")
for i in range(10):
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    buffer.add(obs, action, reward, next_obs, terminated)
    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()

print("Testing backpropagation...")
# buffer size is 10, let's train with batch_size 4
agent.train(buffer, batch_size=4)
print("Test complete. Models initialized, forward passed, environment stepped, buffer sampled, and backpropagation executed successfully.")
