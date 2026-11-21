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
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.005),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )
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
    actor_critic, auxiliary = [], []
    for name, parameter in net.named_parameters():
        group = auxiliary if name.startswith(("force_", "balance", "wrench")) else actor_critic
        group.append(parameter)
    optimizer = torch.optim.Adam([{"params": actor_critic, "lr": args.lr},
                                  {"params": auxiliary, "lr": 3e-4}])
    # SoftMimic StaticStand uses Isaac Lab joint_pos_limits, with a 0.9 soft range.
    ranges = tensor(model.jnt_range[1:])
    center = ranges.mean(dim=1)
    half = .45 * (ranges[:, 1] - ranges[:, 0])
    soft_lower, soft_upper = center - half, center + half
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True, map_location=gs.device))


    def optimize(records, bootstrap, terminal):
        obs, act, logp, values, rewards, wrench, balance = zip(*records)
        v = tensor(values)
        end = torch.zeros_like(v, dtype=torch.bool)
        end[-1] = terminal
        adv, ret = advantages(tensor(rewards), v, torch.cat((v[1:], v.new_tensor([bootstrap]))), end, end)
        batch = [torch.stack(xs) for xs in zip(*obs)]
        with torch.no_grad():
            old_policy = net(*batch)[0]
            old_mean, old_std = old_policy.mean.clone(), old_policy.stddev.clone()
        for _ in range(5):
            for idx in torch.randperm(len(v)).chunk(min(4, len(v))):
                with torch.no_grad():
                    current = net(*(x[idx] for x in batch))[0]
                    original = torch.distributions.Normal(old_mean[idx], old_std[idx])
                    kl = torch.distributions.kl_divergence(original, current).sum(-1).mean().item()
                    # Multipliers and bounds are scheduler choices; Table A6 specifies target KL.
                    group = optimizer.param_groups[0]
                    if kl > .02:
                        group["lr"] = max(1e-5, group["lr"] / 1.5)
                    elif 0 < kl < .005:
                        group["lr"] = min(.01, group["lr"] * 1.5)
                loss = update(net, optimizer, [x[idx] for x in batch], torch.stack(act)[idx],
                              torch.stack(logp)[idx], adv[idx], ret[idx], torch.stack(wrench)[idx],
                              torch.stack(balance)[idx])
        return loss

    records = []
    iteration = 0
    while iteration < args.iterations:
        start = rng.integers(max(1, len(clip)-1500+1))
        row = clip[start]
        ref = np.r_[row[:3], row[6], row[3:6], row[7:]]
        data.qpos[:] = ref
        mj.mj_forward(model, data)
        gravity = np.array([0., 0., -1.])
        k = np.exp(rng.uniform(np.log(10), np.log(970)))
        force = rng.uniform(-args.force, args.force, 3)
        initial = robot.get_qpos().clone()
        initial[:7] = tensor(ref[:7])
        initial[coords] = tensor(ref[7:])
        robot.set_qpos(initial, zero_velocity=True)
        previous = torch.zeros(29)
        memory = None

        stiffness = torch.full((10, 1), float(np.log(k)))
        for step in range(min(1500, len(clip)-start) + 1):
            frame = min(start + step, len(clip)-1)
            row = clip[frame]
            ref = np.r_[row[:3], row[6], row[3:6], row[7:]]
            target = tensor(adapt(model, ref, force, k)[7:] if args.force else ref[7:])
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
            flags = torch.stack([torch.isfinite(x).all() for x in obs])
            if not flags.all():
                names = ("proprioception", "past_reference", "log_stiffness", "future_reference", "privileged")
                invalid = [name for name, ok in zip(names, flags.tolist()) if not ok]
                raise FloatingPointError(f"Non-finite {invalid} at update={iteration} frame={frame}")
            with torch.no_grad():
                dist, value, _, _ = net(*(x[None] for x in obs))
            if len(records) == 24:
                loss = optimize(records, value.item(), False)
                iteration += 1
                print(f"update={iteration} episode_steps={step} loss={loss:.3f}", flush=True)
                records.clear()
                if iteration >= args.iterations:
                    break
                with torch.no_grad():
                    dist, value, _, _ = net(*(x[None] for x in obs))
            if step == min(1500, len(clip)-start):
                if records:
                    # Time limit: bootstrap before resetting the simulator.
                    optimize(records, value.item(), False)
                    iteration += 1
                    records.clear()
                break
            with torch.no_grad():
                action = dist.sample()[0]
                logp = dist.log_prob(action).sum()
            robot.control_dofs_position(.5 * action + tensor(model.qpos0[7:]), indices)
            for _ in range(4):
                wrist.apply_external_force(force)
                scene.step()
            measured = robot.get_dofs_position(indices)
            limit_error = ((soft_lower - measured).relu() + (measured - soft_upper).relu()).sum()
            error = (measured-target).square().mean()
            reward = 2 * torch.exp(-error/.09) - .1 * (action-previous).square().mean()
            reward += .5 - 10.0 * limit_error
            fallen = robot.get_pos()[2].item() < .35
            height_error = robot.get_pos()[2] - row[2]
            up = Rotation.from_quat(robot.get_quat().cpu().numpy()[[1,2,3,0]]).as_matrix()[2,2]
            reward += torch.exp(-height_error.square()/.01) + float(up) - 5 * int(fallen)
            records.append((obs, action, logp, value.item(), reward.item(), wrench, balance))
            previous = action
            if fallen:
                loss = optimize(records, 0., True)
                iteration += 1
                print(f'update={iteration} fell_at={(step+1)/50:.2f}s loss={loss:.3f}', flush=True)
                records.clear()
                break
    torch.save(net.state_dict(), args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="standing.pt")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--force", type=float, default=5., help="Per-axis wrist load bound in N; 0 disables loads")
    p.add_argument("--init", help="Initialize weights from an earlier training stage")
    train(p.parse_args())


if __name__ == '__main__':
    main()

