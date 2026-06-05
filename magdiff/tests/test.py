import magdiff.controller as ctrl
import mujoco as MJ
import mujoco_warp as MW

if __name__ == "__main__":
    ctrl.recorder.open("osc.mp4")
    for _ in range(1000):
        ctrl.osc.step()
        MW.step(ctrl.warp_model, ctrl.warp_data)
        ctrl.recorder.capture()
        
    ctrl.recorder.close()

