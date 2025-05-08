from typing import Dict, Any
import numpy as np
import gymnasium as gym
import torch
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Panda
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat

@register_env("MedicineSortEnv-v1", max_episode_steps=100)
class MedicineSortEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args,  robot_uids="panda", **kwargs):
        # Medicine properties
        self.medicine_types = ["pill", "bottle", "syringe"]
        self.medicine_colors = {
            "pill": [0.8, 0.2, 0.2, 1], 
            "bottle": [0.2, 0.2, 0.8, 1],  
            "syringe": [0.2, 0.8, 0.2, 1],  
        }
        self.bin_locations = {
            "pill": [0.3, -0.2, 0],
            "bottle": [0.3, 0.0, 0],
            "syringe": [0.3, 0.2, 0],
        }
        super().__init__(*args,  robot_uids=robot_uids, **kwargs)


    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.medicines = []
        for med_type in self.medicine_types:
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
            if med_type == "pill":
                builder.add_capsule_visual(radius=0.01, half_length=0.02, material=sapien.render.RenderMaterial(base_color=self.medicine_colors[med_type]))
                builder.add_capsule_collision(radius=0.01, half_length=0.02)
            elif med_type == "bottle":
                builder.add_box_visual(half_size=[0.02, 0.02, 0.04],material=sapien.render.RenderMaterial(base_color=self.medicine_colors[med_type]))
                builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
            elif med_type == "syringe":
                builder.add_cylinder_visual(radius=0.01, half_length=0.05,material=sapien.render.RenderMaterial(base_color=self.medicine_colors[med_type]))
                builder.add_cylinder_collision(radius=0.01, half_length=0.05)
            self.medicines.append(builder.build(name=f"{med_type}_medicine"))

        self.bins = {}
        for med_type, loc in self.bin_locations.items():
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[0.05, 0.05, 0.02])
            builder.add_box_collision(half_size=[0.05, 0.05, 0.02])
            self.bins[med_type] = builder.build_static(name=f"{med_type}_bin")
            self.bins[med_type].set_pose(sapien.Pose(loc))


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        for med in self.medicines:
            med.set_pose(Pose.create_from_pq(
                p=torch.rand((len(env_idx), 3)) * torch.tensor([0.4, 0.4, 0]) + torch.tensor([-0.2, -0.2, 0.02]),
                q=euler2quat(0, 0, torch.rand(len(env_idx)) * 2 * np.pi)
            ))

    def evaluate(self):

        success = torch.ones(len(self.medicines), dtype=torch.bool)
        for med in self.medicines:
            med_type = med.name.split("_")[0]
            bin_center = torch.tensor(self.bin_locations[med_type])
            dist = torch.norm(med.pose.p[:, :2] - bin_center[:2], dim=1)
            success &= (dist < 0.05)  
        return {"success": success}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(len(self.medicines))
        for med in self.medicines:
            med_type = med.name.split("_")[0]
            bin_center = torch.tensor(self.bin_locations[med_type])
            dist = torch.norm(med.pose.p[:, :2] - bin_center[:2], dim=1)
            reward += 1 - torch.tanh(5 * dist)  
        reward[info["success"]] = 10.0  
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0
        return self.compute_dense_reward(obs, action, info) / max_reward

