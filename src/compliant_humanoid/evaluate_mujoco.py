"""Interactive MuJoCo transfer evaluation of a Genesis-trained G1 policy."""
import argparse
from contextlib import nullcontext
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco as mj
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .augmentation import CLIP, XML
from .controller import Controller
from .motion import load_clip


def scene_model():
    spec = mj.MjSpec.from_file(str(XML))
    spec.option.timestep = .005
    spec.option.disableflags |= mj.mjtDisableBit.mjDSBL_AUTORESET
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.add_texture(name="floor_checker", type=mj.mjtTexture.mjTEXTURE_2D,
                     builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER, width=128, height=128,
                     rgb1=[.32, .36, .40], rgb2=[.50, .55, .59])
    floor = spec.add_material(name="floor_material", texrepeat=[2, 2], texuniform=True)
    floor.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "floor_checker"
    spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], material="floor_material")
    for joint in spec.joints:
        if joint.type == mj.mjtJoint.mjJNT_HINGE:
            # Genesis 1.4.5 MJCF fills unspecified rotor inertia on one-DOF links.
            if joint.armature == 0:
                joint.armature = .1
            servo = spec.add_actuator(name=joint.name, target=joint.name,
                                      trntype=mj.mjtTrn.mjTRN_JOINT)
            servo.set_to_position(kp=60., kv=3.)
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected a floating G1 with 29 actuated joints")
    return model


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    policy = Controller(33)
    policy.load_state_dict(torch.load(args.policy, weights_only=True, map_location="cpu"))
    policy.eval()
    model = scene_model()
    data = mj.MjData(model)
    clip, reference = load_clip(args.clip, model)
    reference = torch.as_tensor(reference, dtype=torch.float32)
    wrist = model.body('left_wrist_yaw_link').id
    arm_ids = [i for i in range(model.nu) if model.joint(i+1).name.startswith(
        ("left_shoulder_", "left_elbow_", "left_wrist_"))]
    limits = model.jnt_range[np.asarray(arm_ids)+1]
    events = SimpleQueue()
    if args.headless:
        window = nullcontext(None)
    else:
        import mujoco.viewer as viewer
        window = viewer.launch_passive(model, data, key_callback=events.put)
    with window as view:
        if view:
            view.cam.lookat[:] = [0, 0, .8]
            view.cam.distance = 2.5
        reset = True
        elapsed = 0
        stiffness = args.stiffness
        pulse = 0
        mode = "play"
        arm_history = None
        while view is None or view.is_running():
            start = time.perf_counter()
            while not events.empty():
                key = events.get()
                if key == ord("R"):
                    reset = True
                elif key == ord("T") and not reset:
                    mode = "hold" if mode == "teach" else "teach"
                    arm_target = torch.tensor(np.clip(data.qpos[7:][arm_ids], *limits.T),
                                              dtype=torch.float32)
                    if mode == "teach":
                        stiffness = 50.
                    else:
                        stiffness_goal = 400.
                elif key == ord("F"):
                    pulse = 25
                elif key in (ord("1"), ord("2"), 321, 322):
                    stiffness = 100. if key in (ord("1"), 321) else 400.
                    stiffness_goal = stiffness
                    print(f"stiffness={stiffness:g} N/m", flush=True)
            if reset:
                mj.mj_resetData(model, data)
                row = clip[0]
                data.qpos[:] = np.r_[row[:3], row[6], row[3:6], row[7:]]
                mj.mj_forward(model, data)
                previous = torch.zeros(29)
                history = None
                pulse = 0
                frame = 0
                if mode != "play":
                    stiffness = args.stiffness
                mode = "play"
                arm_history = None
                reset = False
            orientation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
            # Floating-joint angular velocity is already in the local body frame.
            prop = torch.cat((torch.tensor(np.r_[data.qvel[3:6],
                              orientation.inv().apply([0, 0, -1]), data.qpos[7:], data.qvel[6:]],
                              dtype=torch.float32), previous))
            history = prop.repeat(10, 1) if history is None else torch.cat((history[1:], prop[None]))
            past = reference[np.maximum(0, frame + np.arange(-9, 1))]
            future = reference[np.minimum(len(clip)-1, frame + 2*np.arange(5))]
            if mode != "play":
                if arm_history is None:
                    arm_history = past[:, arm_ids].clone()
                if mode == "teach":
                    arm_target = torch.tensor(np.clip(data.qpos[7:][arm_ids], *limits.T),
                                              dtype=torch.float32)
                arm_history = torch.cat((arm_history[1:], arm_target[None]))
                past[:, arm_ids] = arm_history
                future[:, arm_ids] = arm_target
                if mode == "hold":
                    # Blend the command in log space over successive control ticks.
                    stiffness = float(np.exp(.9*np.log(last_stiffness) + .1*np.log(stiffness_goal)))
            last_stiffness = stiffness
            dist, _, _, _ = policy(history[None], past[None],
                                    torch.full((1, 10, 1), float(np.log(stiffness))), future[None])
            previous = dist.mean[0]
            data.ctrl[:] = model.qpos0[7:] + .5*previous.numpy()
            if args.test_pull and elapsed == 50:
                pulse = 25
            if view:
                view.set_texts((mj.mjtFontScale.mjFONTSCALE_100, mj.mjtGridPos.mjGRID_TOPRIGHT,
                                "Stiffness command (N/m)\nLeft wrist test force (N)\nLeft arm [T]",
                                f"{stiffness:.0f}\n{5 if pulse else 0}\n{mode}"))
                view.sync()
            if pulse:
                data.xfrc_applied[wrist, 0] += 5
            if not np.isfinite(data.ctrl).all():
                raise FloatingPointError(f"Non-finite policy target at t={data.time:.3f}s")
            for _ in range(4):
                mj.mj_step(model, data)
                for warning in (mj.mjtWarning.mjWARN_BADQACC, mj.mjtWarning.mjWARN_BADQVEL,
                                mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADCTRL):
                    stat = data.warning[warning]
                    if stat.number:
                        raise FloatingPointError(
                            f"{warning.name} at t={data.time:.3f}s, index={stat.lastinfo}, "
                            f"max_abs_target={np.abs(data.ctrl).max():.3f} rad"
                        )
            if pulse:
                data.xfrc_applied[wrist, 0] -= 5
                pulse -= 1
            elapsed += 1
            if elapsed == 75:
                mj.mj_forward(model, data)
                print(f"t=1.50 wrist_x_m={data.xpos[wrist, 0]:.8f}", flush=True)
            frame = min(frame + 1, len(clip)-1)
            if args.seconds and elapsed >= int(np.ceil(50*args.seconds)):
                break
            if view:
                time.sleep(max(0, .02-(time.perf_counter()-start)))
        print(f"steps={elapsed} root_z={data.qpos[2]:.3f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--stiffness", type=float, default=100.)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--test-pull", action="store_true", help="Apply +5 N at left wrist from 1 to 1.5 s")
    p.add_argument("--seconds", type=float, help="Stop after this many simulated seconds")
    args = p.parse_args()
    if not np.isfinite(args.stiffness) or args.stiffness <= 0:
        p.error("Stiffness must be positive and finite")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        p.error("Duration must be positive and finite")
    if args.headless and args.seconds is None:
        p.error("Specify --seconds in headless mode")
    run(args)


if __name__ == "__main__":
    main()
