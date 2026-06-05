# Throw a ball at 100 different velocities.

import mujoco
import mujoco_warp as mjw
import warp as wp
import mediapy as media
import numpy as np

file="./xarm7/world_6_general.xml"
NWORLD = 1024

mjm = mujoco.MjModel.from_xml_path(file)
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=NWORLD)

CAM_RES = (128, 128)
rc = mjw.create_render_context(
 mjm,
 nworld=NWORLD,
 cam_res=CAM_RES,
 render_rgb=True,
 render_depth=True,
)

# Randomize the qpos of each world
qpos = d.qpos.numpy()
qpos[:, 0] = np.random.uniform(0, 2 * np.pi, size=NWORLD)
d.qpos = wp.array(qpos, dtype=wp.float32)

# Populate data fields for the current state
mjw.forward(m, d)

# Refit BVH updates the internal BVH using the current state. It is necessary to
# call this after the state has been updates and before rendering.
mjw.refit_bvh(m, d, rc)

# Render the current state
mjw.render(m, d, rc)
cam_index = 0
resolution = rc.cam_res.numpy()[cam_index]
rgb_data = wp.zeros((NWORLD, resolution[1], resolution[0]), dtype=wp.vec3)
mjw.get_rgb(rc, rgb_out=rgb_data, camera_index=cam_index)


wp.synchronize()

rgb_np = rgb_data.numpy()   # (N, H, W, 3) automatically

# grid (since NWORLD = 16)
rgb_grid = rgb_np.reshape(int(np.sqrt(NWORLD)), int(np.sqrt(NWORLD)), CAM_RES[1], CAM_RES[0], 3)
rgb_grid = rgb_grid.transpose(0, 2, 1, 3, 4)
rgb_grid = rgb_grid.reshape(int(np.sqrt(NWORLD)) * CAM_RES[0], int(np.sqrt(NWORLD)) * CAM_RES[1], 3)

media.write_image("rendered_worlds.png", rgb_grid)

# Extract depth data, which is scaled and clamped to [0, 1]
depth_data = wp.zeros((NWORLD, CAM_RES[1], CAM_RES[0]), dtype=float)
mjw.get_depth(rc, camera_index=0, depth_scale=3.5, depth_out=depth_data)

# Display our rendered worlds in a grid
depth_grid = depth_data.numpy().reshape(int(np.sqrt(NWORLD)), int(np.sqrt(NWORLD)), CAM_RES[1], CAM_RES[0])
depth_grid = depth_grid.transpose(0, 2, 1, 3)
depth_grid = depth_grid.reshape(int(np.sqrt(NWORLD)) * CAM_RES[0], int(np.sqrt(NWORLD)) * CAM_RES[1])
media.write_image("rendered_depth.png", depth_grid)