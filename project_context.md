# Context Handoff: 3D Lunar Lander Upgrade

**To the AI Assistant:**
The USER is upgrading from a 2D Deep Q-Network (DQN) Lunar Lander to a fully 3D custom MuJoCo environment using a Continuous Control algorithm.

### 1. The Goal
We need to build a custom Gymnasium environment (`LunarLander3D-v0`) using the MuJoCo physics engine from scratch, and pair it with a Deep Deterministic Policy Gradient (DDPG) or Proximal Policy Optimization (PPO) agent that can learn to fly it.

### 2. Core Requirements
*   **Physics Engine (`lunar_lander_3d.xml`)**: Write a MuJoCo XML defining a landing pad and a 3D lander body. It must have 5 independent linear actuators (Main Thruster pointing up, and North, South, East, West thrusters for pitch/roll control).
*   **Environment (`mujoco_env.py`)**: Wrap the XML in a Gymnasium interface.
    *   **Action Space:** Continuous `Box` of shape (5,) for the 5 engine throttles.
    *   **State Space:** 3D coordinates, velocities, and orientation (quaternions or euler angles).
    *   **Reward Function:** Must include our previously perfected reward shaping (Penalize fuel use, penalize horizontal drifting, heavy penalty for hovering/slow-descent, and a massive +500 bonus for surviving/touching the pad).
*   **Brain (`agent.py` & `model.py`)**: Implement a DDPG or PPO algorithm since the standard DQN we used previously mathematically cannot handle continuous tracking of 5 simultaneous engine throttles.
*   **Training (`train.py`)**: Create the training loop that utilizes prioritized experience replay and tracks the `max_avg_score` to automatically save the best weights to `checkpoint.pth`.

### 3. Immediate Next Steps for the AI
1. The USER has created an empty folder at `/Users/sstasbih/Desktop/Projects/3d_lunar_lander`.
2. Generate all the necessary files mentioned above in this new workspace to get the 3D physics simulation and the continuous-control learning agent up and running!
