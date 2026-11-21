import unittest

import torch
from compliant_humanoid.controller import Controller
from compliant_humanoid.ppo import update


class PPOStabilityTest(unittest.TestCase):
    def test_extreme_finite_log_ratio(self):
        model = Controller(33)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        obs = [torch.zeros(3, 10, 93), torch.zeros(3, 10, 38),
               torch.zeros(3, 10, 1), torch.zeros(3, 5, 38), torch.zeros(3, 33)]
        action = torch.zeros(3, 29)
        with torch.no_grad():
            old_logp = model(*obs)[0].log_prob(action).sum(-1) - torch.tensor([120., 0., -120.])
        loss = update(model, optimizer, obs, action, old_logp, torch.tensor([-1., 1., 0.]),
                      torch.zeros(3), torch.zeros(3, 14), torch.zeros(3, 19))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
