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
    spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], rgba=[.38, .4, .42, 1])
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
        while view is None or view.is_running():
            start = time.perf_counter()
            while not events.empty():
                key = events.get()
                if key == ord("R"):
                    reset = True
                elif key in (ord("1"), ord("2")):
                    stiffness = 100. if key == ord("1") else 400.
                    print(f"stiffness={stiffness:g} N/m", flush=True)
            if reset:
                mj.mj_resetData(model, data)
                row = clip[0]
                data.qpos[:] = np.r_[row[:3], row[6], row[3:6], row[7:]]
                mj.mj_forward(model, data)
                previous = torch.zeros(29)
                history = None
                frame = 0
                reset = False
            orientation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
            # Floating-joint angular velocity is already in the local body frame.
            prop = torch.cat((torch.tensor(np.r_[data.qvel[3:6],
                              orientation.inv().apply([0, 0, -1]), data.qpos[7:], data.qvel[6:]],
                              dtype=torch.float32), previous))
            history = prop.repeat(10, 1) if history is None else torch.cat((history[1:], prop[None]))
            past = reference[np.maximum(0, frame + np.arange(-9, 1))]
            future = reference[np.minimum(len(clip)-1, frame + 2*np.arange(5))]
            dist, _, _, _ = policy(history[None], past[None],
                                    torch.full((1, 10, 1), float(np.log(stiffness))), future[None])
            previous = dist.mean[0]
            data.ctrl[:] = model.qpos0[7:] + .5*previous.numpy()
            if view:
                view.sync()
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
            elapsed += 1
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
