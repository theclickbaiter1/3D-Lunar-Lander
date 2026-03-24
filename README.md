# 3D Lunar Lander - MuJoCo & DDPG

A 3D Lunar Lander simulation built using the MuJoCo physics engine and trained using Deep Deterministic Policy Gradient (DDPG) with Prioritized Experience Replay (PER).

> [!IMPORTANT]
> **Project Status:** This project is currently a work in progress. While the environment and agent architecture are fully implemented, the agent has not yet achieved a consistent "solved" state. Training is ongoing to refine the reward function and hyperparameters for successful autonomous landing.

## Project Overview
This project transitions the classic 2D Lunar Lander problem into a fully 3D environment. It uses MuJoCo for realistic physics and Gymnasium for the reinforcement learning interface.

### Key Features
- **3D Physics:** Realistic lander dynamics in a 3D coordinate system.
- **Continuous Control:** Uses DDPG to handle continuous action spaces.
- **Visual Effects:** Dynamic engine plumes that scale with thrust intensity.
- **Stabilization Logic:** Internal PD controllers helping the agent manage attitude (pitch/roll).

## Installation

### Prerequisites
- Python 3.8+
- MuJoCo (installed via `pip install mujoco`)

### Setup
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To start or resume training:
```bash
python train.py
```
Training progress is saved to `training_state.pkl` and model weights to `latest_actor.pth` / `latest_critic.pth`.

### Evaluation/Visualization
To watch the trained agent:
```bash
python enjoy.py
```

## Environment Details

### Observation Space (13 Dimensions)
- `pos`: x, y, z (3)
- `quat`: Orientation quaternion (4)
- `vel`: Linear velocities (3)
- `ang_vel`: Angular velocities (3)

### Action Space (3 Continuous Actions)
The agent outputs 3 values in the range `[-1, 1]`:
1. **Main Throttle:** Controls the main vertical engine.
2. **Target Pitch:** Sets the desired pitch angle.
3. **Target Roll:** Sets the desired roll angle.

*Note: The environment internally maps these pitch/roll targets to 4 side thrusters (North, South, East, West) using a PD controller.*

## Reward Function
The reward is designed to encourage a safe landing on the pad:
- **Distance Reward:** Based on proximity to the landing pad (center of the world).
- **Upright Reward:** Incentivizes keeping the lander vertical.
- **Speed Shaping:** Rewards slow descent and minimal horizontal drift when near the ground.
- **Fuel Penalty:** Small penalties for using main and side engines.
- **Time Penalty:** Encourages faster landings to prevent hovering.
- **Terminal Rewards:**
  - Safe Landing: High positive reward.
  - Crash/Hard Landing: Significant penalty.
  - Out of Bounds: Penalty.

*All rewards are scaled by a factor of 0.05 (1/20) for DDPG stability.*

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Algorithm | DDPG + PER |
| Max Episodes | 10,000 |
| Max Steps per Episode | 1,000 |
| Batch Size | 128 |
| Learning Rate (Actor) | 1e-4 |
| Learning Rate (Critic) | 1e-3 |
| Replay Buffer Size | 100,000 |
| Gamma (Discount Factor) | 0.99 |
| Tau (Soft Update) | 0.005 |
| Exploration Noise | 0.2 (decaying to 0.01) |
| PER Alpha | 0.6 |
| PER Beta | 0.4 |

## Storage Requirements
- **Weights:** `.pth` files are small (~300 KB each).
- **Training State:** `training_state.pkl` can grow significantly (e.g., ~13MB for 100k samples) as it contains the full replay buffer for resuming training. This file is ignored by default in `.gitignore`.

## Dependencies
- `numpy`
- `torch`
- `mujoco`
- `gymnasium`
- `mujoco-python-viewer`
