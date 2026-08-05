import time
import glfw
from envs.manipulation import Manipulation

env = Manipulation(
    render_mode="human",
    n_obstacles=4,
    max_episode_steps=1000,
)

obs, info = env.reset()
print("obs shape:", obs.shape, "action space", env.action_space)

# glfw.init()

action = env.action_space.sample()

for i in range(5000):       
    obs, rew, term, trunc, info = env.step(action)
    time.sleep(0.002)                            # slow down so the window is watchable

    if term or trunc:
        print(f"episode ended at step {i}, success={info.get('is_success')}, "
              f"collision={info.get('collision')}")
        obs, info = env.reset()

env.close()