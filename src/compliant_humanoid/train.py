"""Single-environment G1 PPO baseline tracking a wrist-load-adapted motion clip."""
import argparse
import numpy as np
import torch
import mujoco as mj
import genesis as gs
from scipy.spatial.transform import Rotation

from .augmentation import XML, CLIP, adapt
from .motion import load_clip
from .controller import Controller
from .ppo import advantages, update


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train(args):
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install the cuda extra and check the NVIDIA driver.")
    gs.init(backend=gs.cuda if args.device == "cuda" else gs.cpu)
    if gs.device.type != args.device:
        raise RuntimeError(f"Genesis initialized on {gs.device}, requested {args.device}")
    torch.set_default_device(gs.device)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.005), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(XML), align=False))
    scene.build()
    model = mj.MjModel.from_xml_path(str(XML))
    data = mj.MjData(model)
    joints = [robot.get_joint(model.joint(i).name) for i in range(1, model.njnt)]
    indices = [j.dofs_idx_local[0] for j in joints]
    coords = [j.qs_idx_local[0] for j in joints]
    wrist = robot.get_link("left_wrist_yaw_link")
    feet = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
    foot_links = [robot.get_link(model.body(i).name).idx_local for i in feet]
    robot.set_dofs_kp(np.full(29, 60.), indices)
    robot.set_dofs_kv(np.full(29, 3.), indices)
    clip, features = load_clip(args.clip, model)
    net = Controller(33)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    for iteration in range(args.iterations):
        start = rng.integers(max(1, len(clip)-24))
        row = clip[start]
        ref = np.r_[row[:3], row[6], row[3:6], row[7:]]
        data.qpos[:] = ref
        mj.mj_forward(model, data)
        gravity = np.array([0., 0., -1.])
        k = np.exp(rng.uniform(np.log(10), np.log(970)))
        force = rng.uniform(-5., 5., 3)
        target = tensor(adapt(model, ref, force, k)[7:])
        initial = robot.get_qpos().clone()
        initial[:7] = tensor(ref[:7])
        initial[coords] = tensor(ref[7:])
        robot.set_qpos(initial, zero_velocity=True)
        previous = torch.zeros(29)
        memory = None
        records = []

        stiffness = torch.full((10, 1), float(np.log(k)))
        for step in range(min(24, len(clip)-start)):
            frame = start + step
            row = clip[frame]
            ref = np.r_[row[:3], row[6], row[3:6], row[7:]]
            target = tensor(adapt(model, ref, force, k)[7:])
            ref_history = tensor(features[np.maximum(0, frame + np.arange(-9, 1))])
            future = tensor(features[np.minimum(len(clip)-1, frame + 2*np.arange(5))])
            qpos = robot.get_qpos().cpu().numpy()
            rot = Rotation.from_quat(qpos[[4, 5, 6, 3]])
            q = robot.get_dofs_position(indices)
            dq = robot.get_dofs_velocity(indices)
            angular = rot.inv().apply(robot.get_ang().cpu().numpy())
            prop = torch.cat((tensor(angular), tensor(rot.inv().apply(gravity)), q, dq, previous))
            memory = prop.repeat(10, 1) if memory is None else torch.cat((memory[1:], prop[None]))
            data.qpos[:] = np.r_[qpos[:7], q.cpu().numpy()]
            mj.mj_forward(model, data)
            com = data.subtree_com[1].copy()
            com_vel = (com-last_com)/.02 if step else np.zeros(3)
            last_com = com
            contact = robot.get_links_net_contact_force().cpu().numpy()[foot_links, 2] > 1
            balance = tensor(np.r_[rot.inv().apply(com-qpos[:3]), rot.inv().apply(com_vel),
                                   rot.inv().apply(robot.get_vel().cpu().numpy()), angular, qpos[2],
                                   contact, data.xpos[feet, 2], rot.inv().apply(com-qpos[:3] +
                                   com_vel*np.sqrt(max(com[2], .01)/9.81))[:2]])
            wrench = tensor(np.r_[rot.inv().apply(force), np.zeros(3), np.log(k),
                                  np.zeros(6), np.log(k)])
            obs = [memory, ref_history, stiffness, future, torch.cat((wrench, balance))]
            with torch.no_grad():
                dist, value, _, _ = net(*(x[None] for x in obs))
                action = dist.sample()[0]
                logp = dist.log_prob(action).sum()
            robot.control_dofs_position(.5 * action + tensor(model.qpos0[7:]), indices)
            for _ in range(4):
                wrist.apply_external_force(force)
                scene.step()
            error = (robot.get_dofs_position(indices)-target).square().mean()
            reward = 2 * torch.exp(-error/.09) - .1 * (action-previous).square().mean()
            fallen = robot.get_pos()[2].item() < .35
            records.append((obs, action, logp, value.item(), reward.item(), wrench, balance))
            previous = action
            if fallen:
                break
        obs, act, logp, values, rewards, wrench, balance = zip(*records)
        v = tensor(values)
        end = torch.zeros_like(v, dtype=torch.bool)
        end[-1] = True
        adv, ret = advantages(tensor(rewards), v, torch.cat((v[1:], v.new_zeros(1))), end, end)
        batch = [torch.stack(xs) for xs in zip(*obs)]
        for _ in range(5):
            for idx in torch.randperm(len(v)).chunk(min(4, len(v))):
                loss = update(net, optimizer, [x[idx] for x in batch], torch.stack(act)[idx],
                              torch.stack(logp)[idx], adv[idx], ret[idx], torch.stack(wrench)[idx],
                              torch.stack(balance)[idx])
        print(iteration, len(v), np.mean(rewards), loss, flush=True)
    torch.save(net.state_dict(), args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="standing.pt")
    train(p.parse_args())


if __name__ == '__main__':
    main()

