"""PPO minibatch update and generalized advantage estimates for WBC2."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def advantages(reward, value, next_value, terminated, episode_end):
    """Time-major tensors. next_value uses the pre-reset observation on timeouts."""
    delta = reward + 0.99 * (~terminated) * next_value - value
    result = torch.empty_like(reward)
    carry = torch.zeros_like(reward[-1])
    for i in reversed(range(len(reward))):
        carry = delta[i] + 0.99 * 0.95 * (~episode_end[i]) * carry
        result[i] = carry
    return result, result + value


def update(model, optimizer, observations, action, old_log_prob, advantage, returns,
           wrench_target, balance_target):
    """Balance fields 13:15 contain binary foot contacts; other fields are continuous."""
    policy, value, wrench, balance = model(*observations)
    log_ratio = policy.log_prob(action).sum(-1) - old_log_prob
    ratio = log_ratio.exp()
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    actor_loss = -surrogate.mean() - 0.005 * policy.entropy().sum(-1).mean()
    value_loss = F.mse_loss(value.squeeze(-1), returns)
    continuous = [*range(13), *range(15, 19)]
    balance_loss = F.mse_loss(balance[:, continuous], balance_target[:, continuous])
    balance_loss += F.binary_cross_entropy_with_logits(balance[:, 13:15], balance_target[:, 13:15])
    loss = actor_loss + 0.5 * value_loss + F.mse_loss(wrench, wrench_target) + balance_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach().item()

