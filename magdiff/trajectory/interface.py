from magdiff.trajectory.bezier import *
import mujoco_warp as MW
import warp as wp
import torch


class Trajectory:
    def __init__(self,base_model, warp_model, warp_data, armspec:list, nworld, nsteps, device="cuda"):
        self.base_model = base_model
        self.warp_model = warp_model
        self.warp_data = warp_data
        self.nworld = nworld
        self.nsteps = nsteps
        self.device = device
        self.left = armspec[0]
        self.right = armspec[1]
        
    def allocate(self):
        MW.step(self.warp_model, self.warp_data)

        xpos_torch = wp.to_torch(self.warp_data.xpos)
        xquat_torch = wp.to_torch(self.warp_data.xquat)

        mocap_pos_torch = wp.to_torch(self.warp_data.mocap_pos)
        mocap_quat_torch = wp.to_torch(self.warp_data.mocap_quat)



        start_pos_torch = xpos_torch[
            :,
            [self.left.body_id, self.right.body_id],
        ]

        start_quat_torch = xquat_torch[
            :,
            [self.left.body_id, self.right.body_id],
        ]

        goal_pos_torch = mocap_pos_torch[
            :,
            [self.left.mocap_id, self.right.mocap_id],
        ]

        goal_quat_torch = mocap_quat_torch[
            :,
            [self.left.mocap_id, self.right.mocap_id],
        ]



        self.start_pos_wp = wp.from_torch(
            start_pos_torch.contiguous(),
            dtype=wp.vec3,
        )

        self.start_rot_wp = wp.from_torch(
            start_quat_torch.contiguous(),
            dtype=wp.quat,
        )

        self.goal_pos_wp = wp.from_torch(
            goal_pos_torch.contiguous(),
            dtype=wp.vec3,
        )

        self.goal_rot_wp = wp.from_torch(
            goal_quat_torch.contiguous(),
            dtype=wp.quat,
        )

        
    def generate_random(self, profile="bezier",sigma=0.6):
        if profile == "bezier":
            self.traj_gen = BezierTrajectory(nworld=self.nworld, nsteps=self.nsteps, device=self.device)
        else:
            raise NotImplementedError(f"Trajectory profile {profile} not implemented")
        
        self.allocate()
        assert hasattr(self, "start_pos_wp"), "unallocated memory buffers"
        
        traj_pos, traj_rot = self.traj_gen.generate(
            self.start_pos_wp,
            self.goal_pos_wp,
            self.start_rot_wp,
            self.goal_rot_wp,
            sigma=sigma,
        )

        return torch.cat([traj_pos, traj_rot], dim=-1)
        


    

