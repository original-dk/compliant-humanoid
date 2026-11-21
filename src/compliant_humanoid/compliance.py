"""WBC2 wrist targets and directional compliance. Inputs share a world frame."""
import numpy as np


def wrist_target(reference, force, stiffness):
    stiffness = np.asarray(stiffness)
    if np.any(stiffness <= 0):
        raise ValueError("Stiffness must be positive (N/m)")
    return np.asarray(reference) + np.asarray(force) / stiffness[..., None]


def book_stiffness(force, wrist, hinge, hinge_axis):
    tangent = np.cross(hinge_axis, np.subtract(wrist, hinge))
    denominator = np.dot(force, force) * np.dot(tangent, tangent)
    alignment = np.dot(force, tangent) ** 2 / denominator if denominator else 0.0
    return 50.0 + 350.0 * np.clip(alignment, 0.0, 1.0)


def axial_stiffness(force):
    """Appendix H plateau for a pelvis-X stiff axis, force in pelvis coordinates."""
    if force is None:
        return 50.0
    magnitude = np.linalg.norm(force)
    if magnitude < 1.0:
        return 50.0
    alignment = (force[0] / magnitude) ** 2
    return 50.0 + 350.0 * np.clip((alignment - 0.3) / 0.4, 0, 1)


def key_insertion(left, right):
    available = [f for f in (left, right) if f is not None]
    return axial_stiffness(max(available, key=np.linalg.norm, default=None))


def cabinet_pull(left):
    return axial_stiffness(left)

