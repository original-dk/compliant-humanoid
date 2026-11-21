"""Batched Genesis collection with per-environment resets and time-major GAE."""
import numpy as np
import torch

from .augmentation import adapt
from .ppo import advantages, update


def rotate(q, v):
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    cross = 2 * torch.linalg.cross(xyz, v)
    return v + q[..., :1]*cross + torch.linalg.cross(xyz, cross)


def train_batch(args, scene, robot, model, indices, coords, wrist, feet, clip, features,
                net, optimizer, soft_lower, soft_upper):
    n = args.num_envs
    device = robot.get_pos().device
    def floats(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32)
    rng = np.random.default_rng(args.seed)
    # Share offline IK trajectories, not simulator state, across environments.
    count = args.load_cases if args.force else 1
    forces = rng.uniform(-args.force, args.force, (count, 3))
    stiffness = np.exp(rng.uniform(np.log(10), np.log(970), count))
    targets = []
    for k, force in zip(stiffness, forces):
        if args.force:
            targets.append([adapt(model, np.r_[r[:3], r[6], r[3:6], r[7:]], force, k)[7:] for r in clip])
        else:
            targets.append(clip[:, 7:])
    targets = floats(np.asarray(targets))
    clip, features, forces = floats(clip), floats(features), floats(forces)
    log_bank = floats(np.log(stiffness))
    default = floats(model.qpos0[7:])
    frame = torch.zeros(n, device=device, dtype=torch.long)
    age = torch.zeros_like(frame)
    case = torch.zeros_like(frame)
    logk = floats(np.zeros(n))
    previous = torch.zeros(n, 29, device=device)
    history = torch.zeros(n, 10, 93, device=device)
    last_com = torch.zeros(n, 3, device=device)
    base_pose = robot.get_qpos().clone()
    mass = robot.get_links_mass().unsqueeze(-1)
    offset = robot.get_links_COM()
    gravity = floats([0, 0, -1])
    past_steps = torch.arange(-9, 1, device=device)
    future_steps = 2*torch.arange(5, device=device)

    def reset(ids):
        frame[ids] = torch.randint(max(1, len(clip)-1), (len(ids),), device=device)
        age[ids] = 0
        case[ids] = torch.randint(count, (len(ids),), device=device)
        logk[ids] = log_bank[case[ids]] if args.force else torch.empty(len(ids), device=device).uniform_(np.log(10), np.log(970))
        row = clip[frame[ids]]
        pose = base_pose[ids].clone()
        pose[:, :7] = row[:, [0, 1, 2, 6, 3, 4, 5]]
        pose[:, coords] = row[:, 7:]
        robot.set_qpos(pose, envs_idx=ids, zero_velocity=True)
        history[ids] = 0
        previous[ids] = 0

    def observe():
        quat = robot.get_quat()
        inv = torch.cat((quat[:, :1], -quat[:, 1:]), -1)
        ang = rotate(inv, robot.get_ang())
        state = torch.cat((ang, rotate(inv, gravity), robot.get_dofs_position(indices),
                           robot.get_dofs_velocity(indices), previous), -1)
        mem = torch.cat((history[:, 1:], state[:, None]), 1)
        mem = torch.where((age == 0)[:, None, None], state[:, None], mem)
        pos = robot.get_links_pos()
        com = ((pos + rotate(robot.get_links_quat(), offset))*mass).sum(1)/mass.sum(-2)
        vel = torch.where((age > 0)[:, None], (com-last_com)/.02, 0.)
        root = robot.get_pos()
        rel = rotate(inv, com-root)
        local_v = rotate(inv, vel)
        capture = rel + local_v*(com[:, 2:3].clamp_min(.01)/9.81).sqrt()
        balance = torch.cat((rel, local_v, rotate(inv, robot.get_vel()), ang, root[:, 2:3],
                             robot.get_links_net_contact_force()[:, feet, 2] > 1,
                             pos[:, feet, 2], capture[:, :2]), -1)
        z = torch.zeros_like(ang)
        k = logk[:, None]
        wrench = torch.cat((rotate(inv, forces[case]), z, k, z, z, k), -1)
        obs = (mem, features[(frame[:, None]+past_steps).clamp_min(0)],
               k[:, None].expand(-1, 10, -1),
               features[(frame[:, None]+future_steps).clamp_max(len(clip)-1)],
               torch.cat((wrench, balance), -1))
        return obs, com

    reset(torch.arange(n, device=device))
    for iteration in range(args.iterations):
        rows = []
        for _ in range(24):
            with torch.no_grad():
                obs, com = observe()
                dist, value, _, _ = net(*obs)
                action = dist.sample()
                old_logp = dist.log_prob(action).sum(-1)
                robot.control_dofs_position(default + .5*action, indices)
                for _ in range(4):
                    wrist.apply_external_force(forces[case])
                    scene.step()
                q = robot.get_dofs_position(indices)
                z = robot.get_pos()[:, 2]
                fallen = z < .35
                tilt = 1 - 2*robot.get_quat()[:, 1:3].square().sum(-1)
                reward = 2*torch.exp(-(q-targets[case, frame]).square().mean(-1)/.09)
                reward += .5 + tilt + torch.exp(-(z-clip[frame, 2]).square()/.01)
                reward -= .1*(action-previous).square().mean(-1) + 5*fallen
                reward -= 10*((soft_lower-q).relu()+(q-soft_upper).relu()).sum(-1)
                # Clone log stiffness before the per-environment reset mutates its storage.
                obs = (*obs[:2], obs[2].clone(), *obs[3:])
                history, previous, last_com = obs[0], action.clone(), com
                frame += 1
                age += 1
                done = fallen | (age >= 1500) | (frame >= len(clip))
                frame.clamp_max_(len(clip)-1)
                nxt = net(*observe()[0])[1].squeeze(-1)
                rows.append((*obs, action, old_logp, value.squeeze(-1), reward, nxt, fallen, done,
                             dist.mean, dist.stddev))
                ids = done.nonzero().flatten()
                if len(ids):
                    # History is stored in the rollout too.
                    history = history.clone()
                    reset(ids)
        stacked = [torch.stack(col) for col in zip(*rows)]
        adv, ret = advantages(stacked[8], stacked[7], stacked[9], stacked[10], stacked[11])
        fields = [x.flatten(0, 1) for x in stacked]
        adv, ret = adv.flatten(), ret.flatten()
        for _ in range(5):
            for idx in torch.randperm(24*n, device=device).split(min(args.batch_cap, 6*n)):
                sub = [x[idx] for x in fields[:5]]
                with torch.no_grad():
                    cur = net(*sub)[0]
                    orig = torch.distributions.Normal(fields[12][idx], fields[13][idx])
                    kl = torch.distributions.kl_divergence(orig, cur).sum(-1).mean().item()
                    group = optimizer.param_groups[0]
                    if kl > .02:
                        group["lr"] = max(1e-5, group["lr"]/1.5)
                    elif 0 < kl < .005:
                        group["lr"] = min(.01, group["lr"]*1.5)
                loss = update(net, optimizer, sub, fields[5][idx], fields[6][idx], adv[idx], ret[idx],
                              sub[4][:, :14], sub[4][:, 14:])
        print(f"update={iteration+1} samples={24*n} falls={stacked[10].sum().item()} loss={loss:.3f}", flush=True)
        del rows, fields, stacked
    torch.save(net.state_dict(), args.output)
