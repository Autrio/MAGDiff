import mujoco as MJ
import mujoco_warp as MW

from magdiff.controller.osc_parallel import ParallelOSC, ArmSpec
from magdiff.controller.render import WarpGridVideoRenderer
from magdiff.grasp import ObjectLoader
from magdiff.parse import NWORLD, NWORLD_REND

import numpy as np


loader = ObjectLoader(nworld=NWORLD,movable_object=True)
base_model, warp_model, warp_data = loader.build(randomize=True,seed=28)

left = ArmSpec(
        name="left",
        body_id=base_model.body("xarm_L_gripper_base_link").id,
        dof_ids=np.array([base_model.joint(f"L_joint{i}").id for i in range(1, 8)], dtype=np.int32),
        act_ids=np.array([base_model.actuator(f"L_act{i}").id for i in range(1, 8)], dtype=np.int32),
        mocap_id=int(base_model.body("gripper2").mocapid[0]),
        offset=0.15,
        K=np.array([1000, 1000, 1000, 100, 100, 100], dtype=np.float32)*10,
        KI=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
        K0=np.ones(7, dtype=np.float32),
    )

right = ArmSpec(
    name="right",
    body_id=base_model.body("R_gripper_body").id,
    dof_ids=np.array([base_model.joint(f"R_joint{i}").id for i in range(1, 7)], dtype=np.int32),
    act_ids=np.array([base_model.actuator(f"R_act{i}").id for i in range(1, 7)], dtype=np.int32),
    mocap_id=int(base_model.body("gripper1").mocapid[0]),
    offset=0.06,
    K=np.array([1000, 1000, 1000, 100, 100, 100], dtype=np.float32),
    KI=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
    K0=np.ones(6, dtype=np.float32),
)

osc = ParallelOSC(warp_model, warp_data, [left, right], nworld=NWORLD)

recorder = WarpGridVideoRenderer(
base_model,
warp_model,
warp_data,
camera_index=1,
nworld=NWORLD,
render_worlds=NWORLD_REND,
width=640,
height=480,
)

recorder.open("grasp_demo.mp4")

for _ in range(1000):
    osc.step()
    MW.step(warp_model, warp_data)
    recorder.capture()
    
recorder.close()