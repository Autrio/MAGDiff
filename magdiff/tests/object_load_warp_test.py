import h5py
import glob
import mujoco as MJ
import mujoco_warp as MW
import trimesh
import json
from scipy.spatial.transform import Rotation as R
import numpy as np

import magdiff.controller as ctrl

import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument("-n", "--nworld", type=int, default=4, help="number of parallel worlds to simulate and render")
args = parser.parse_args()



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

user_scale = 0.5

scale = scales[grasp_files[idx].split("/")[-1].split(".")[0]]

mesh = trimesh.load(object_file)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(scale*user_scale)

mesh.export("temp.obj")
mesh = trimesh.load("temp.obj")

pos = mesh.centroid
bottom = mesh.bounds[0][2]


R_conv = np.array([
    [ 1,  0,  0],
    [ 0,  0,  1],
    [ 0, -1,  0],
])

rz90 = R.from_euler("z", 90, degrees=True).as_matrix()

mj_frame_correction = np.eye(4)
mj_frame_correction[:3, :3] = R_conv @ rz90

grasp1 = grasp[0] @ mj_frame_correction
grasp2 = grasp[1] @ mj_frame_correction

mesh_rot = R.from_euler("xz", [90, 90], degrees=True).as_matrix()
mesh_correction = np.eye(4)
mesh_correction[:3, :3] = mesh_rot

gripper_mesh = trimesh.load("../dataset/gripper.obj")
gripper_mesh.apply_translation(-gripper_mesh.centroid)
gripper_mesh.apply_transform(mesh_correction)

gripper_mesh.export("gripper_temp.obj")

# grasp1 = grasp[0]
# grasp2 = grasp[1]

g1_pos = grasp1[:3,3:]*user_scale
g2_pos = grasp2[:3,3:]*user_scale

g1_pos[2] -= bottom
g2_pos[2] -= bottom


g1_quaternion = R.from_matrix(grasp1[:3, :3]).as_quat(scalar_first=True)
g2_quaternion = R.from_matrix(grasp2[:3, :3]).as_quat(scalar_first=True)


