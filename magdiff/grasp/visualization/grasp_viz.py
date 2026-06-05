import h5py
import glob
import mujoco
from mujoco import viewer
import trimesh
import json
from scipy.spatial.transform import Rotation as R
import numpy as np

idx = np.random.randint(0, len(glob.glob("../dataset/grasps/*")))
# idx = 42

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


            # <joint type="free" name="object_joint" />

mj_xml = f"""<mujoco>
    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
		<camera name="front_camera" pos="0 2 2" xyaxes="-1 0 0 0 -0.70710678 0.70710678" resolution="640 480" output="rgb depth"/>
        
            <geom name="floor"
                type="plane" size="5 5 0.1" material="groundplane" contype="1"
                conaffinity="1" />
        <body name="object" pos="{pos[0]} {pos[1]} {-bottom}">
            <geom type="mesh" mesh="object_mesh" />
        </body>
        <body name="gripper1" pos="{g1_pos[0][0]} {g1_pos[1][0]} {g1_pos[2][0]}" quat="{g1_quaternion[0]} {g1_quaternion[1]} {g1_quaternion[2]} {g1_quaternion[3]}" mocap="true" >
            <geom type="mesh" mesh="gripper_mesh" rgba="1 0 0 1" />
        </body>
        <body name="gripper2" pos="{g2_pos[0][0]} {g2_pos[1][0]} {g2_pos[2][0]}" quat="{g2_quaternion[0]} {g2_quaternion[1]} {g2_quaternion[2]} {g2_quaternion[3]}" mocap="true" >
            <geom type="mesh" mesh="gripper_mesh"  rgba="0 1 0 1" />
        </body>
    </worldbody>
    <asset>
        <mesh name="object_mesh" file="temp.obj" />
        <mesh name="gripper_mesh" file="gripper_temp.obj" />
        <material name="white" rgba="1 1 1 1" />
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
			height="3072" />
		<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
			rgb2="0.1 0.2 0.3"
			markrgb="0.8 0.8 0.8" width="300" height="300" />
		<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
			reflectance="0.2" specular="0.2" shininess="0.1" />
    </asset>
</mujoco>"""    

model = mujoco.MjModel.from_xml_string(mj_xml)
data = mujoco.MjData(model)
model.vis.scale.framewidth = 0.01
model.vis.scale.framelength = 0.4

v = viewer.launch_passive(model, data)
v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

print(grasp_file)
print(object_file)

while v.is_running():
    mujoco.mj_step(model, data)
    v.sync()

