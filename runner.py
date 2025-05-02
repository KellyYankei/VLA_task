import os
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from MedicineSort_env import MedicineSortEnv
import sapien

def generate_videos(n_episodes=10, max_steps_per_episode=100, video_dir="medicine_sort_videos"):

    os.makedirs(video_dir, exist_ok=True)
    
    env = gym.make(
        "MedicineSortEnv-v1",
        obs_mode="state",
        render_mode="human",
        num_envs=1,
    )
    obs, _ = env.reset(options=dict(reconfigure=True))

    scene = env.unwrapped.scene
    camera = scene.add_camera(
        name="fixed_view",
        width=1280,
        height=720,
        pose=sapien.Pose([0.5, -1.0, 0.8], [0.9239, 0, 0.3827, 0]), 
        near=0.1, 
        far=100.0,  
        fovy=1.0,  
    )

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

if __name__ == "__main__":
    generate_videos(
        n_episodes=1,  
        max_steps_per_episode=200, 
        video_dir="medicine_sort" 
    )