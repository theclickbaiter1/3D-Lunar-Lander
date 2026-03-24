import os
import numpy as np
import torch
from mujoco_env import LunarLander3DEnv
from agent import DDPG, PrioritizedReplayBuffer
from collections import deque
import pickle

env = LunarLander3DEnv(render_mode=None)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = np.array([1.0] * action_dim) # All engines range -1 to 1 directly through step function

agent = DDPG(state_dim, action_dim, max_action)
replay_buffer = PrioritizedReplayBuffer(max_size=100000, state_dim=state_dim, action_dim=action_dim)

def train():
    num_episodes = 10000
    max_steps = 1000
    batch_size = 128
    
    scores = []
    scores_window = deque(maxlen=100)
    max_avg_score = -np.inf
    start_ep = 1
    
    # Check if a saved training state exists
    if os.path.exists("training_state.pkl") and os.path.exists("latest_actor.pth"):
        print("Found existing training state. Resuming...")
        agent.load("latest")
        with open("training_state.pkl", "rb") as f:
            state_dict = pickle.load(f)
            start_ep = state_dict['ep'] + 1
            scores = state_dict['scores']
            scores_window = state_dict['scores_window']
            max_avg_score = state_dict['max_avg_score']
            
            # Reconstruct the replay buffer safely
            if 'replay_buffer_size' in state_dict:
                replay_buffer.size = state_dict['replay_buffer_size']
                replay_buffer.ptr = state_dict['replay_buffer_ptr']
                replay_buffer.state = state_dict['rb_state']
                replay_buffer.action = state_dict['rb_action']
                replay_buffer.reward = state_dict['rb_reward']
                replay_buffer.next_state = state_dict['rb_next_state']
                replay_buffer.dead = state_dict['rb_dead']
                replay_buffer.priorities = state_dict['rb_priorities']
                
        print(f"Resuming from Episode {start_ep} with Best Avg Score: {max_avg_score:.2f}")
    else:
        print("Starting Training Loop for LunarLander3D ...")
    
    try:
        for ep in range(start_ep, num_episodes + 1):
            state, _ = env.reset()
            score = 0
            
            for step in range(max_steps):
                # Select action with decaying noise for exploration
                explore_noise = max(0.01, 0.2 - (0.2 * ep / 1000.0))
                action = agent.select_action(state, noise=explore_noise)
                
                # Step the environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                dead = 1 if terminated else 0
                
                # Add to replay buffer
                replay_buffer.add(state, action, reward, next_state, dead)
                
                state = next_state
                score += reward
                
                # Train the agent
                if replay_buffer.size > batch_size * 2:
                    agent.train(replay_buffer, batch_size=batch_size)
                    
                if terminated or truncated:
                    break
                    
            scores.append(score)
            scores_window.append(score)
            
            avg_score = np.mean(scores_window)
            
            print(f"Episode {ep}\tScore: {score:.2f}\tAvg Score: {avg_score:.2f}")
            
            # Periodically auto-save latest state just in case of crash without KeyboardInterrupt
            if ep % 50 == 0:
                agent.save("latest")
                with open("training_state.pkl", "wb") as f:
                    pickle.dump({
                        'ep': ep,
                        'scores': scores,
                        'scores_window': scores_window,
                        'max_avg_score': max_avg_score,
                        'replay_buffer_size': replay_buffer.size,
                        'replay_buffer_ptr': replay_buffer.ptr,
                        'rb_state': replay_buffer.state,
                        'rb_action': replay_buffer.action,
                        'rb_reward': replay_buffer.reward,
                        'rb_next_state': replay_buffer.next_state,
                        'rb_dead': replay_buffer.dead,
                        'rb_priorities': replay_buffer.priorities
                    }, f)
            
            if avg_score > max_avg_score and ep >= 100:
                max_avg_score = avg_score
                print(f"*** New max avg score: {max_avg_score:.2f}! Saving checkpoint. ***")
                agent.save("checkpoint")
                
            if avg_score >= 500.0:
                print(f"Environment solved in {ep} episodes! Checkpoint saved.")
                agent.save("checkpoint_solved")
                break
                
    except KeyboardInterrupt:
        print("\nTraining paused by user. Saving current state...")
        agent.save("latest")
        with open("training_state.pkl", "wb") as f:
            pickle.dump({
                'ep': ep - 1, # Don't count the interrupted episode
                'scores': scores[:-1] if len(scores) > 0 and ep == len(scores) else scores,
                'scores_window': scores_window,
                'max_avg_score': max_avg_score,
                'replay_buffer_size': replay_buffer.size,
                'replay_buffer_ptr': replay_buffer.ptr,
                'rb_state': replay_buffer.state,
                'rb_action': replay_buffer.action,
                'rb_reward': replay_buffer.reward,
                'rb_next_state': replay_buffer.next_state,
                'rb_dead': replay_buffer.dead,
                'rb_priorities': replay_buffer.priorities
            }, f)
        print("State saved successfully to 'latest_*.pth' and 'training_state.pkl'. Run train.py to resume.")

if __name__ == "__main__":
    train()
