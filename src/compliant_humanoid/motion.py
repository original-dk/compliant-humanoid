"""Resample readable-mimic G1 CSV clips and construct the 38-D reference."""
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_clip(path, model):
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Expected at least two finite G1 frames with 36 columns")
    source = np.arange(len(raw))/30
    times = np.arange(int(source[-1]*50)+1)/50
    rotations = Slerp(source, Rotation.from_quat(raw[:, 3:7]))(times)
    linear = np.column_stack([np.interp(times, source, col) for col in raw.T])
    linear[:, 3:7] = rotations.as_quat()
    velocity = rotations.inv().apply(np.gradient(linear[:, :3], .02, axis=0))
    yaw_rate = np.gradient(np.unwrap(rotations.as_euler("zyx")[:, 0]), .02)
    gravity = rotations.inv().apply([0., 0., -1.])
    feet = [model.body(side+"_ankle_roll_link").id for side in ("left", "right")]
    data = mj.MjData(model)
    contact = np.empty((len(linear), 2))
    for i, row in enumerate(linear):
        data.qpos[:] = np.r_[row[:3], row[6], row[3:6], row[7:]]
        mj.mj_forward(model, data)
        contact[i] = data.xpos[feet, 2] < .09
    return linear, np.column_stack((linear[:, 7:], velocity[:, :2], yaw_rate, linear[:, 2],
                                    gravity, contact))

