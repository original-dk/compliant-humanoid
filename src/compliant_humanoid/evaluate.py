"""Evaluate a deterministic policy under a finite world-frame force pulse."""
import argparse
import csv
from pathlib import Path

import genesis as gs
import mujoco as mj
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .augmentation import XML, CLIP
from .controller import Controller
from .motion import load_clip


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    policy = Controller(33)
    if args.policy:
        policy.load_state_dict(torch.load(args.policy, map_location="cpu", weights_only=True))
    policy.to(device).eval()
    gs.init(backend=gs.cuda if device.type == "cuda" else gs.cpu, seed=1)
    model = mj.MjModel.from_xml_path(str(XML))
    clip, reference = load_clip(args.clip, model)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.005), show_viewer=args.viewer)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(XML), align=False))
    scene.build()
    joints = [robot.get_joint(model.joint(i).name) for i in range(1, model.njnt)]
    indices = [j.dofs_idx_local[0] for j in joints]
    coords = [j.qs_idx_local[0] for j in joints]
    link = robot.get_link(args.link)
    robot.set_dofs_kp(np.full(29, 60.), indices)
    robot.set_dofs_kv(np.full(29, 3.), indices)

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    first = clip[0]
    pose = robot.get_qpos().clone()
    pose[:7] = tensor(np.r_[first[:3], first[6], first[3:6]])
    pose[coords] = tensor(first[7:])
    robot.set_qpos(pose, zero_velocity=True)
    previous = tensor(np.zeros(29))
    memory = None
    stiffness = tensor(np.full((1, 10, 1), np.log(args.stiffness)))
    with open(args.csv, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow("time_s fx_N fy_N fz_N link_x_m link_y_m link_z_m root_z_m fallen".split())
        for frame in range(min(len(clip), int(np.ceil(50*args.seconds)))):
            t = frame/50
            rotation = Rotation.from_quat(robot.get_quat().cpu().numpy()[[1, 2, 3, 0]]).inv()
            state = torch.cat((tensor(rotation.apply(robot.get_ang().cpu().numpy())),
                               tensor(rotation.apply([0., 0., -1.])), robot.get_dofs_position(indices),
                               robot.get_dofs_velocity(indices), previous))
            memory = state.repeat(10, 1) if memory is None else torch.cat((memory[1:], state[None]))
            past = tensor(reference[np.maximum(0, frame+np.arange(-9, 1))])
            future = tensor(reference[np.minimum(len(clip)-1, frame+2*np.arange(5))])
            dist, _, _, _ = policy(memory[None], past[None], stiffness, future[None])
            previous = dist.mean[0]
            target = tensor(clip[frame, 7:]) if args.pd else tensor(model.qpos0[7:]) + .5*previous
            robot.control_dofs_position(target, indices)
            force = args.force if args.start <= t < args.start + args.duration else np.zeros(3)
            for _ in range(4):
                link.apply_external_force(np.asarray(force))
                scene.step()
            height = robot.get_pos()[2].item()
            fallen = height < .35
            writer.writerow(((frame+1)/50, *force, *link.get_pos().cpu().tolist(), height, int(fallen)))
            if fallen:
                break
    print(f"{args.csv}: {(frame+1)/50:.2f} s, fallen={fallen}, root_z={height:.3f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--policy", type=Path)
    mode.add_argument("--pd", action="store_true", help="Track reference joint angles directly")
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--link", default="left_wrist_yaw_link")
    p.add_argument("--force", nargs=3, type=float, default=[5, 0, 0])
    p.add_argument("--start", type=float, default=.5)
    p.add_argument("--duration", type=float, default=.5)
    p.add_argument("--seconds", type=float, default=3)
    p.add_argument("--stiffness", type=float, default=100)
    p.add_argument("--csv", default="pull.csv")
    args = p.parse_args()
    if args.seconds <= 0 or args.start < 0 or args.duration < 0 or args.stiffness <= 0:
        p.error("Invalid duration, start time, or stiffness")
    run(args)


if __name__ == "__main__":
    main()
