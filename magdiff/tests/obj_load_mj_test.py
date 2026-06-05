import mujoco as MJ 
import torch 
from torch import nn
import numpy as np
import mujoco_warp as MW
import warp as wp
import mujoco.viewer as VW
import mediapy as media
import time 
from scipy.spatial.transform import Rotation as Rot

import h5py
import glob
import trimesh
import json
import numpy as np

idx = np.random.randint(0, len(glob.glob("../dataset/grasps/*")))
idx = 0

with open("../dataset/scales.json", "r") as f:
    scales = json.load(f)


grasp_files = glob.glob("../dataset/grasps/*")
obj_files = glob.glob("../dataset/meshes/*.obj")

grasp_file = h5py.File(grasp_files[idx], "r")
object_file = obj_files[obj_files.index("../dataset/meshes/" + grasp_files[idx].split("/")[-1].split(".")[0] + ".obj")]

grasps = grasp_file["grasps/grasps"][:]

grasp = grasps[np.random.randint(0, grasps.shape[0])]
grasp = grasps[-1]


scale = scales[grasp_files[idx].split("/")[-1].split(".")[0]]

user_scale = 0.5

mesh = trimesh.load(object_file)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(scale*user_scale)

mesh.export("../models/assets/temp.obj")
mesh = trimesh.load("../models/assets/temp.obj")

pos = mesh.centroid
bottom = mesh.bounds[0][2]


R_conv = np.array([
    [ 1,  0,  0],
    [ 0,  0,  1],
    [ 0, -1,  0],
])

Rz90 = Rot.from_euler("z", 90, degrees=True).as_matrix()

mj_frame_correction = np.eye(4)
mj_frame_correction[:3, :3] = R_conv @ Rz90

grasp1 = grasp[0] @ mj_frame_correction
grasp2 = grasp[1] @ mj_frame_correction

mesh_rot = Rot.from_euler("xz", [90, 90], degrees=True).as_matrix()
mesh_correction = np.eye(4)
mesh_correction[:3, :3] = mesh_rot

gripper_mesh = trimesh.load("../dataset/gripper.obj")
gripper_mesh.apply_translation(-gripper_mesh.centroid)
gripper_mesh.apply_transform(mesh_correction)

gripper_mesh.export("../models/assets/gripper_temp.obj")

# grasp1 = grasp[0]
# grasp2 = grasp[1]

g1_pos = grasp1[:3,3:]*user_scale
g2_pos = grasp2[:3,3:]*user_scale

g1_pos[2] -= bottom
g2_pos[2] -= bottom


g1_quaternion = Rot.from_matrix(grasp1[:3, :3]).as_quat(scalar_first=True)
g2_quaternion = Rot.from_matrix(grasp2[:3, :3]).as_quat(scalar_first=True)

xml_string = open("../models/world_parse_string.xml", "r").read().format(
    pos=pos,
    bottom=-bottom,
    g1_pos=g1_pos,
    g2_pos=g2_pos,
    g1_quaternion=g1_quaternion,
    g2_quaternion=g2_quaternion,
)

mj_model = MJ.MjModel.from_xml_string(xml_string)
mj_data = MJ.MjData(mj_model)

#*================================================================ CONTROL CLASSES ========================================================

class opspace:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu
        
        self.body_name = None
        self.offset = None
        self.mocap_id = None
        self.dof_ids = None
        
        self.dt = self.model.opt.timestep
        
        self.twist_int = np.zeros(6)
        
    def dyn(self):
        
        assert self.body_name is not None, "body_name must be set"
        assert self.dof_ids is not None, "dof_ids must be set"
        assert self.offset is not None, "offset must be set"
        assert self.mocap_id is not None, "mocap_id must be set"
        
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        MJ.mj_jacBody(
            self.model, self.data, jacp, jacr, self.model.body(self.body_name).id
        )
        jac6 = np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])

        # Mass matrix
        M_full = np.zeros((self.model.nv, self.model.nv))
        MJ.mj_fullM(self.model, M_full, self.data.qM)
        M_arm = M_full[np.ix_(self.dof_ids, self.dof_ids)] #!

        # Compute Minv_all via mj_solveM on identity
        I_nv = np.eye(self.model.nv)
        buf = np.zeros((self.model.nv, self.model.nv))
        MJ.mj_solveM(self.model, self.data, buf, I_nv)
        Minv_all = np.array(buf).reshape(self.model.nv, self.model.nv)
        Minv_arm = Minv_all[np.ix_(self.dof_ids, self.dof_ids)]
        Mx_inv_arm = jac6 @ Minv_arm @ jac6.T #! 

        # jacDot
        jacDotp = np.zeros((3, self.model.nv))
        jacDotr = np.zeros((3, self.model.nv))
        MJ.mj_jacDot(
            self.model,
            self.data,
            jacDotp,
            jacDotr,
            np.array([0,0,self.offset],dtype=np.float64),
            self.model.body(self.body_name).id,
        )
        jacDot6 = np.vstack([jacDotp[:, self.dof_ids], jacDotr[:, self.dof_ids]])

        # Bias forces
        h = self.data.qfrc_bias[self.dof_ids].copy()

        # End-effector position and quaternion
        self.ee_xpos = self.data.body(self.model.body(self.body_name).id).xpos.copy()
        self.ee_quat = self.data.body(self.model.body(self.body_name).id).xquat.copy()
        
        
        if self.offset != 0.0:
            ee_xmat = self.data.body(self.model.body(self.body_name).id).xmat.reshape(3, 3)
            ee_z_axis = ee_xmat[:, 2]
            self.ee_xpos = self.ee_xpos + self.offset * ee_z_axis

        # Mocap position/quaternion
        if self.mocap_id is not None:
            mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
            mocap_quat = self.data.mocap_quat[self.mocap_id].copy()
        else:
            mocap_pos = np.zeros(3)
            mocap_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Quaternion error -> rotation vector
        quat_conj = np.zeros(4)
        MJ.mju_negQuat(quat_conj, self.ee_quat)
        error_quat = np.zeros(4)
        MJ.mju_mulQuat(error_quat, mocap_quat, quat_conj)
        rot = Rot.from_quat(
            [error_quat[1], error_quat[2], error_quat[3], error_quat[0]]
        )
        rotvec = rot.as_rotvec()

        twist = np.zeros(6)
        twist[:3] = mocap_pos - self.ee_xpos
        twist[3:] = rotvec
        
        # Integral error update & anti-windup clamping
        self.twist_int += twist * self.dt
        if np.any(np.abs(self.twist_int) > 100.0):
            self.twist_int = np.zeros(6)
            
        return  Mx_inv_arm, M_arm, jac6, jacDot6, h, twist, self.twist_int


def compute_damping(Mx, K):
    def safe_matrix_sqrt(matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        sqrt_eigvals = np.sqrt(np.abs(eigvals))
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        return sqrt_matrix
    
    sqrt_Mx = safe_matrix_sqrt(Mx)
    sqrt_K = np.sqrt(K)  
    D = sqrt_Mx @ sqrt_K + sqrt_K @ sqrt_Mx
    return D
        
        

class arm7(opspace):
    def __init__(self, model, data, K, K0, KI):
        super().__init__(model, data)

        self.jnts = [f"L_joint{i}" for i in range(1,8)]    
        self.acts = [f"L_act{i}" for i in range(1,8)]
        
        self.dof_ids = [self.model.joint(jnt).id for jnt in self.jnts]
        self.act_ids = [self.model.actuator(act).id for act in self.acts]
        
        self.body_name = "xarm_L_gripper_base_link"
        self.offset = 0.1
        self.mocap_id = model.body("gripper2").mocapid[0]
        
        self.K = np.diag(K)
        self.KI = np.diag(KI)
        self.K0 = np.diag(K0)
        
    def step(self, debug=False):
        Mx_inv_arm, M_arm, jac, jacDot, h, twist, twist_int = self.dyn()
        
        try:
            Minv = np.linalg.pinv(M_arm, rcond=1e-6)
        except Exception:
            Minv = np.linalg.pinv(M_arm)

        try:
            Mx = np.linalg.inv(Mx_inv_arm) if abs(np.linalg.det(Mx_inv_arm)) > 1e-8 \
                else np.linalg.pinv(Mx_inv_arm, rcond=1e-3)
        except Exception:
            Mx = np.linalg.pinv(Mx_inv_arm, rcond=1e-3)
        
        D = compute_damping(Mx, self.K)
        
        q = self.data.qpos[self.dof_ids]
        qd = self.data.qvel[self.dof_ids]
        
        mu   = Mx @ (jac @ (Minv @ h) + jacDot @ qd)
        
        xd = jac @ qd
        
        F = -D @ xd + self.K @ twist + self.KI @ twist_int + mu
        
        tau = jac.T @ F
        
        if len(self.dof_ids) > 6:
            self.beta = 2.0 * (np.sqrt(np.diag(self.K0)) * (-qd)) + self.K0 @ (-q)
            jacBar = Minv @ jac.T @ Mx
            tau += (np.eye(len(self.dof_ids)) - jac.T @ jacBar.T) @ self.beta
    
        # Apply torques to actuators
        self.data.ctrl[self.act_ids] = tau
        
        if debug:
            print("Jacobian:\n", jac)
            print("Jacobian Dot:\n", jacDot)
            print("Mass Matrix (Arm):\n", M_arm)
            print("Inverse Mass Matrix (Arm):\n", Minv)
            print("Operational Space Mass Matrix:\n", Mx)
            print("Damping Matrix:\n", D)
            print("Bias Forces:\n", h)
            print("Twist:\n", twist)
            print("Integral of Twist:\n", twist_int)
            print("Control Force F:\n", F)
            print("Joint Torques tau:\n", tau)
            print("beta:\n", self.beta) if len(self.dof_ids) > 6 else "N/A"
            print("tau nullspace:\n", (np.eye(len(self.dof_ids)) - jac.T @ jacBar.T) @ self.beta) if len(self.dof_ids) > 6 else "N/A"
            print("="*50)
        

class arm6(arm7):
    def __init__(self, model, data, K, K0, KI):
        super().__init__(model, data, K, K0, KI)
        
        self.jnts = [f"R_joint{i}" for i in range(1,7)]    
        self.acts = [f"R_act{i}" for i in range(1,7)]
        
        self.dof_ids = [self.model.joint(jnt).id for jnt in self.jnts]
        self.act_ids = [self.model.actuator(act).id for act in self.acts]
        
        self.body_name = "R_gripper_body"
        self.offset = 0.06
        self.mocap_id = model.body("gripper1").mocapid[0]
        
        self.K = np.diag(K)
        self.KI = np.diag(KI)
        self.K0 = np.diag(K0)   
        

#*================================================================ PARALLELZATION  =======================================================

        
if __name__ == "__main__":
    Kp = [1000,1000,1000,100,100,100]
    Ki = [1,1,1,1,1,1]
    K0 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    
    arm7 = arm7(mj_model, mj_data, K=Kp, K0=K0, KI=Ki)
    arm6 = arm6(mj_model, mj_data, K=Kp, K0=K0, KI=Ki)
    
    viewer = VW.launch_passive(mj_model, mj_data)
    
    while viewer.is_running():
        arm7.step()
        arm6.step()
        MJ.mj_step(mj_model, mj_data)
        viewer.sync()
