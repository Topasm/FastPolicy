import time
import gymnasium as gym
import gym_aloha  # Register the environment
import mujoco.viewer

import os
os.environ["MUJOCO_GL"] = "egl"


# Create the Gym environment (without needing Gym rendering)
env = gym.make('gym_aloha/AlohaInsertion-v0', render_mode="rgb_array")

# Access dm_control's Physics object
physics = env.unwrapped._env.physics

# Get raw model and data pointers
model = physics.model.ptr
data = physics.data.ptr


# Reset environment
env.reset()

# Launch MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        action = env.action_space.sample()

        # Apply action manually
        physics.data.ctrl[:14] = action
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

        step += 1
        if step % 100 == 0:
            print(f"Step {step}")

# Close Gym env
env.close()

os.environ["MUJOCO_GL"] = "egl"


# Create the Gym environment (without needing Gym rendering)
env = gym.make('gym_aloha/AlohaInsertion-v0', render_mode="rgb_array")

# Access dm_control's Physics object
physics = env.unwrapped._env.physics

# Get raw model and data pointers
model = physics.model.ptr
data = physics.data.ptr


# Reset environment
env.reset()

# Launch MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        action = env.action_space.sample()

        # Apply action manually
        physics.data.ctrl[:14] = action
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

        step += 1
        if step % 100 == 0:
            print(f"Step {step}")

# Close Gym env
env.close()
