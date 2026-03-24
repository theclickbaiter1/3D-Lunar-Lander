import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class LunarLander3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.xml_path = "lunar_lander_3d.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Action space: 3 continuous actions
        # [main, target_pitch, target_roll] all mapped from [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 13 dimensions
        # pos (3), quat (4), vel (3), ang_vel (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Get geom ids for the plumes we added to the XML for visual effects
        self.plume_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) 
                          for name in ["main_plume", "north_plume", "south_plume", "east_plume", "west_plume"]]
        
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        a = np.clip(action, -1.0, 1.0)
        
        # Mapped action 0 to 1 for the main engine
        main_ctrl = (a[0] + 1.0) / 2.0
        
        # Target attitude (pitch and roll) from the RL agent
        # We limit the max tilt target to e.g. [-0.3, 0.3] rads
        target_pitch = a[1] * 0.3
        target_roll = a[2] * 0.3
        
        # Current attitude and angular velocity
        obs = self._get_obs()
        up_vector = self._get_up_vector(obs[3:7])
        
        # Extract Euler angles (roughly) from up_vector
        # uy roughly corresponds to pitch error (tilting around x-axis)
        # ux roughly corresponds to roll error (tilting around y-axis)
        current_pitch = np.arcsin(np.clip(up_vector[1], -1.0, 1.0))
        current_roll = np.arcsin(np.clip(up_vector[0], -1.0, 1.0))
        
        wx, wy, wz = obs[10], obs[11], obs[12]
        
        # PD Controller calculations
        # Kp = Proportional gain, Kd = Derivative gain
        Kp = 2.0
        Kd = 0.5
        
        # Error calculation
        error_pitch = target_pitch - current_pitch
        error_roll = target_roll - current_roll
        
        # PD signal
        # Note: we use wy for pitch (rotation around y axis) and wx for roll (rotation around x axis)
        pd_pitch = Kp * error_pitch - Kd * wx
        pd_roll = Kp * error_roll - Kd * wy
        
        # Map PD signals to the 4 side engines
        # North/South engines control pitch
        north_ctrl = np.clip(-pd_pitch, 0.0, 1.0)
        south_ctrl = np.clip(pd_pitch, 0.0, 1.0)
        
        # East/West engines control roll
        east_ctrl = np.clip(-pd_roll, 0.0, 1.0)
        west_ctrl = np.clip(pd_roll, 0.0, 1.0)
        
        # Combined control array for penalty calculation later
        ctrl = [main_ctrl, north_ctrl, south_ctrl, east_ctrl, west_ctrl]

        # Set controls: max force is defined in XML
        self.data.ctrl[0] = ctrl[0] * 300.0  # main engine
        self.data.ctrl[1] = ctrl[1] * 50.0   # north
        self.data.ctrl[2] = ctrl[2] * 50.0   # south
        self.data.ctrl[3] = ctrl[3] * 50.0   # east
        self.data.ctrl[4] = ctrl[4] * 50.0   # west
        
        # Update visual plume alpha (transparency) based on engine throttling
        for i, geom_id in enumerate(self.plume_ids):
            if geom_id != -1: 
                # Amplify the visual so it's easier to see even on low throttle
                alpha_val = min(1.0, float(ctrl[i]) * 2.5) 
                self.model.geom_rgba[geom_id][3] = alpha_val
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward(obs, ctrl)
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.current_step = 0
        
        # Randomize initial state slightly to ensure robust learning
        self.data.qpos[0:3] = [
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(5.0, 8.0)
        ]
        
        self.data.qvel[0:3] = [
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-1.0, 0.0)
        ]
        
        # Initial tilt
        tilt_axis = np.random.randn(3)
        tilt_axis /= np.linalg.norm(tilt_axis)
        tilt_angle = self.np_random.uniform(-0.3, 0.3)
        q = np.array([np.cos(tilt_angle/2), 
                      np.sin(tilt_angle/2)*tilt_axis[0], 
                      np.sin(tilt_angle/2)*tilt_axis[1], 
                      np.sin(tilt_angle/2)*tilt_axis[2]])
        self.data.qpos[3:7] = q
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel
        ]).astype(np.float32)

    def _compute_reward(self, obs, mapped_action):
        x, y, z = obs[0], obs[1], obs[2]
        vx, vy, vz = obs[7], obs[8], obs[9]
        wx, wy, wz = obs[10], obs[11], obs[12]
        
        reward = 0.0
        
        # Calculate helpful metrics
        drift = np.sqrt(vx**2 + vy**2)
        dist_to_pad = np.sqrt(x**2 + y**2 + (z - 0.5)**2)
        dist_xy = np.sqrt(x**2 + y**2)
        up_vector = self._get_up_vector(obs[3:7])
        tilt = np.arccos(np.clip(up_vector[2], -1.0, 1.0))
        ang_vel = np.sqrt(wx**2 + wy**2 + wz**2)
        
        # POSITIVE SHAPING REWARDS
        # DDPG learns much better from "getting closer" than "being punished less"
        
        # Reward for being close to pad (max 2.0 at pad, 0 at dist 10)
        shaping_dist = max(0, 10.0 - dist_to_pad) * 0.2
        reward += shaping_dist
        
        # Reward for being upright and stable
        shaping_upright = max(0, 1.0 - tilt) * 1.5
        reward += shaping_upright
        
        # Reward for moving slowly near the ground (but not hovering infinitely)
        if z < 5.0 and dist_xy < 3.0:
            shaping_speed = max(0, 2.0 - np.sqrt(vz**2 + drift**2)) * 1.0
            reward += shaping_speed
        
        # Small penalties (Fuel and Time)
        # Escalating time penalty: takes more points away the longer it hovers
        time_penalty = 0.1 + (self.current_step / 500.0)
        reward -= time_penalty
        reward -= mapped_action[0] * 0.3          # Main engine penalty
        reward -= sum(mapped_action[1:]) * 0.05   # Side engine penalties
        
        terminated = False
        
        # Ground collision zone
        if z < 0.8:
            if dist_xy > 2.0:
                reward -= 50.0 # Missed pad
                terminated = True
            elif np.abs(vz) > 1.5 or drift > 1.0 or tilt > 0.5:
                reward -= 50.0 # Hard landing
                terminated = True
            else:
                reward += 200.0 # Safe landing!
                terminated = True
                
        # Out of bounds
        if np.abs(x) > 15 or np.abs(y) > 15 or z > 20:
            reward -= 50.0
            terminated = True
            
        # DDPG CRITICAL FIX: Scale rewards to roughly [-1.0, 2.0] range
        reward = reward / 20.0
            
        return reward, terminated, False
        
    def _get_up_vector(self, q):
        qw, qx, qy, qz = q
        ux = 2 * (qx*qz + qw*qy)
        uy = 2 * (qy*qz - qw*qx)
        uz = qw*qw - qx*qx - qy*qy + qz*qz
        return np.array([ux, uy, uz])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco_viewer
                    self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                except ImportError:
                    print("Please install mujoco_viewer: pip install mujoco-python-viewer")
                    self.render_mode = None
                    return
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
