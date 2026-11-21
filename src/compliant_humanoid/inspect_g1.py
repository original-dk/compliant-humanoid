"""Inspect a floating-base G1 MJCF and exercise Genesis at 50 Hz."""
import argparse
from pathlib import Path

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    gs.init()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(args.xml.resolve(strict=True))))
    scene.build()
    if robot.n_dofs != 35:
        raise ValueError(f"Expected 6 floating-base and 29 joint DOFs, got {robot.n_dofs}")
    for joint in robot.joints:
        print(joint.name, joint.dofs_idx_local)
    for _ in range(4):
        scene.step()


if __name__ == "__main__":
    main()

