"""WBC2 actor, asymmetric critic and supervised state estimators."""
import torch
from torch import nn
from torch.distributions import Normal


def mlp(*sizes):
    layers = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers.extend([nn.Linear(n_in, n_out), nn.ELU()])
    return nn.Sequential(*layers[:-1])


class Controller(nn.Module):
    def __init__(self, privileged_dim):
        super().__init__()
        self.force_projection = nn.Sequential(nn.Linear(132, 64), nn.ELU())
        self.force_gru = nn.GRU(64, 128, 2, batch_first=True)
        self.force_latent = nn.Linear(128, 96)
        self.wrench = mlp(96, 256, 128, 14)
        self.balance_latent = mlp(930, 256, 64)
        self.balance = mlp(64, 256, 19)
        self.reference = nn.Sequential(nn.Conv1d(38, 64, 3, padding=1), nn.ELU(),
                                       nn.AdaptiveAvgPool1d(1), nn.Flatten())
        features = 930 + 96 + 64 + 64
        self.actor = mlp(features, 512, 256, 128, 29)
        self.critic = mlp(features + privileged_dim, 512, 512, 512, 512, 1)
        self.log_std = nn.Parameter(torch.zeros(29))

    def forward(self, proprio, past_reference, log_k, future, privileged=None):
        history = torch.cat((proprio, past_reference, log_k), dim=-1)
        sequence, _ = self.force_gru(self.force_projection(history))
        force = self.force_latent(sequence[:, -1])
        body = self.balance_latent(proprio.flatten(1))
        reference = self.reference(future.transpose(1, 2))
        features = torch.cat((proprio.flatten(1), force.detach(), body.detach(), reference), -1)
        policy = Normal(self.actor(features), self.log_std.exp())
        value = None if privileged is None else self.critic(torch.cat((features, privileged), -1))
        return policy, value, self.wrench(force), self.balance(body)

