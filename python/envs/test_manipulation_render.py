import time
import glfw
from envs.manipulation import Manipulation

env = Manipulation(
    render_mode="human",
    n_obstacles=2,
    max_episode_steps=2000,
)

obs, info = env.reset()

glfw.init()

action = env.action_space.sample()
      
for i in range(10000):
    obs, rew, term, trunc, info = env.step(action)
    time.sleep(0.002)                            # slow down so the window is watchable

    if term or trunc:
        print(f"episode ended at step {i}, success={info.get('is_success')}, "
              f"target_collision={info.get('target_collision')}, "
              f"obstacle_collision={info.get('obstacle_collision')}")
        obs, info = env.reset()

env.close()