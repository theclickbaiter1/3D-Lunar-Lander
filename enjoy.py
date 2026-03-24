import time
import numpy as np
import torch
import os
from mujoco_env import LunarLander3DEnv
from agent import DDPG

def enjoy():
    # Set to human render mode to visualize the simulation using mujoco-python-viewer
    env = LunarLander3DEnv(render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([1.0] * action_dim) 

    agent = DDPG(state_dim, action_dim, max_action)
    
    # Check if a model checkpoint exists
    if os.path.exists("checkpoint_actor.pth"):
        print("Loading trained agent weights...")
        agent.load("checkpoint")
    else:
        print("No checkpoint found! Running an untrained, random agent.")

    print("Starting simulation viewer...")
    
    episodes = 5
    for ep in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            # Select action deterministically (no noise during inference)
            action = agent.select_action(state, noise=0.0)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            
            # Slow down slightly for easier viewing
            time.sleep(0.01)
            
        print(f"Watch Episode {ep+1} complete. Final Score: {score:.2f}")

    env.close()

if __name__ == "__main__":
    enjoy()
