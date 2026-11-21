"""Generate one compliant G1 pose from a retargeted LAFAN1 frame using offline FK."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

from .compliance import wrist_target

DATA = Path(__file__).resolve().parents[2] / "data"
XML = DATA / "g1_description/g1_29dof.xml"
CLIP = DATA / "demo/knee_bend.csv"


def adapt(model, reference, force, stiffness):
    data = mj.MjData(model)
    data.qpos[:] = reference
    mj.mj_forward(model, data)
    names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
             "right_ankle_roll_link"]
    ids = [model.body(name).id for name in names]
    targets = data.xpos[ids].copy()
    targets[0] = wrist_target(targets[0], force, stiffness)
    com = data.subtree_com[1].copy()
    limits = model.jnt_range[1:]

    def residual(joints):
        data.qpos[7:] = joints
        mj.mj_forward(model, data)
        errors = (data.xpos[ids] - targets) * np.array([[2], [1], [4], [4]])
        return np.r_[errors.ravel(), data.subtree_com[1] - com, 0.05 * (joints-reference[7:])]

    solution = least_squares(residual, np.clip(reference[7:], *limits.T), bounds=limits.T)
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.r_[reference[:7], solution.x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=CLIP)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--force", type=float, nargs=3, default=[5, 0, 0])
    parser.add_argument("--stiffness", type=float, default=100)
    args = parser.parse_args()
    row = np.loadtxt(args.clip, delimiter=",")[args.frame]
    reference = np.r_[row[:3], row[6], row[3:6], row[7:]]
    print(adapt(mj.MjModel.from_xml_path(str(XML)), reference, args.force, args.stiffness))


