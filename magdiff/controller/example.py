from dataclasses import dataclass
import numpy as np
import mujoco as MJ
import mujoco_warp as MW
import warp as wp
from scipy.spatial.transform import Rotation as Rot
import argparse as ap
import magdiff.controller.osc_parallel as osc
import magdiff.controller.render as render
import os

NWORLD = 4


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--nworld", type=int, default=4, help="number of parallel worlds to simulate and render")
    args = parser.parse_args()
    NWORLD = args.nworld
    
    from pathlib import Path

    ROOT = Path(__file__).resolve()

    while not (ROOT / "pyproject.toml").exists():
        ROOT = ROOT.parent

    model = MJ.MjModel.from_xml_path(f"{ROOT}/magdiff/models/world_6_general.xml")
    data = MJ.MjData(model)

    warp_model = MW.put_model(model)
    warp_data = MW.make_data(model, nworld=NWORLD, njmax=300,nconmax=200)

    mocap_pos = warp_data.mocap_pos.numpy()

    mocap_pos += np.random.normal(
        loc=0.0,
        scale=0.2,
        size=mocap_pos.shape,
    )

    warp_data.mocap_pos = wp.array(mocap_pos, dtype=wp.vec3)
                
    left = osc.ArmSpec(
        name="left",
        body_id=model.body("xarm_L_gripper_base_link").id,
        dof_ids=np.array([model.joint(f"L_joint{i}").id for i in range(1, 8)], dtype=np.int32),
        act_ids=np.array([model.actuator(f"L_act{i}").id for i in range(1, 8)], dtype=np.int32),
        mocap_id=int(model.body("L_mocap_marker").mocapid[0]),
        offset=0.15,
        K=np.array([1000, 1000, 1000, 100, 100, 100], dtype=np.float32)*10,
        KI=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
        K0=np.ones(7, dtype=np.float32),
    )

    right = osc.ArmSpec(
        name="right",
        body_id=model.body("R_gripper_body").id,
        dof_ids=np.array([model.joint(f"R_joint{i}").id for i in range(1, 7)], dtype=np.int32),
        act_ids=np.array([model.actuator(f"R_act{i}").id for i in range(1, 7)], dtype=np.int32),
        mocap_id=int(model.body("R_mocap_marker").mocapid[0]),
        offset=0.06,
        K=np.array([1000, 1000, 1000, 100, 100, 100], dtype=np.float32),
        KI=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
        K0=np.ones(6, dtype=np.float32),
    )

    osc = osc.ParallelOSC(warp_model, warp_data, [left, right], nworld=NWORLD)


    recorder = render.WarpGridVideoRenderer(
    model,
    warp_model,
    warp_data,
    nworld=NWORLD,
    width=640,
    height=480,
    layout=render.GridLayout(np.ceil(np.sqrt(NWORLD)).astype(int), np.ceil(np.sqrt(NWORLD)).astype(int))
    )
    
    recorder.open("osc.mp4")
    for _ in range(1000):
        osc.step()
        MW.step(warp_model, warp_data)
        recorder.capture()
        
    recorder.close()


