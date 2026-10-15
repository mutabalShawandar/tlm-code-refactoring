import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms").
    Sign-based momentum update, single EMA buffer (~half the memory of AdamW).

    Update per step:
      c = beta1 * m + (1 - beta1) * grad
      theta = theta - lr * (sign(c) + weight_decay * theta)
      m = beta2 * m + (1 - beta2) * grad

    Because updates are sign(.), the effective step size is less sensitive to
    gradient magnitude than AdamW's -- published guidance is lr ~3-10x smaller
    than the AdamW lr you'd otherwise use, and weight_decay ~3-10x larger.
    Not yet validated for this project; A/B against the AdamW baseline before trusting.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
                p.add_(c.sign(), alpha=-lr)

                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
