"""DDPM (Ho et al., 2020) from scratch on 2-D spiral data.

Run:  python diffusion.py                (default 6000 steps)
      python diffusion.py --steps 500    (quick test)
Produces: forward.png, loss.png, reverse.png
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

T = 1000  # number of diffusion steps


# ---------------------------------------------------------------- data
def make_spiral(n=8000, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.sqrt(rng.uniform(0.05, 1, n)) * 3.2 * math.pi
    r = theta / (3.2 * math.pi)
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x += noise * rng.normal(size=x.shape)
    return torch.tensor(x * 2.0, dtype=torch.float32)  # scale ~[-2, 2]


# ---------------------------------------------------------------- noise schedule
betas = torch.linspace(1e-4, 0.02, T)             # beta_t (linear schedule)
alphas = 1.0 - betas                              # alpha_t
alpha_bar = torch.cumprod(alphas, dim=0)          # \bar{alpha}_t = prod_{s<=t} alpha_s


def q_sample(x0, t, eps=None):
    """Jump directly to timestep t:  x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps."""
    if eps is None:
        eps = torch.randn_like(x0)
    ab = alpha_bar[t].unsqueeze(-1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps


# ---------------------------------------------------------------- network eps_theta(x_t, t)
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same idea as Transformer positional encoding)."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredictor(nn.Module):
    def __init__(self, hidden=256, t_dim=64):
        super().__init__()
        self.t_emb = TimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_emb(t)], dim=-1))


# ---------------------------------------------------------------- reverse process
@torch.no_grad()
def p_sample_loop(model, n, snapshots=(999, 600, 300, 100, 0)):
    """x_{t-1} = 1/sqrt(a_t) * (x_t - b_t/sqrt(1-abar_t) * eps_theta(x_t,t)) + sigma_t z"""
    x = torch.randn(n, 2)
    out = {}
    for t in reversed(range(T)):
        tt = torch.full((n,), t, dtype=torch.long)
        eps_hat = model(x, tt)
        mean = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps_hat) / alphas[t].sqrt()
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = mean + betas[t].sqrt() * z  # sigma_t^2 = beta_t
        if t in snapshots:
            out[t] = x.clone()
    return out


def scatter(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="#39c5bb")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    data = make_spiral()

    # --- forward process: no learning, only a fixed schedule ---
    ts = [0, 50, 150, T - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(10, 2.8))
    for ax, t in zip(axes, ts):
        xt, _ = q_sample(data, torch.full((len(data),), t))
        scatter(ax, xt.numpy(), f"t = {t if t < T - 1 else T}")
    fig.suptitle("Forward process: a little Gaussian noise at every step")
    fig.tight_layout(); fig.savefig("forward.png", dpi=150)

    # --- training: L_simple = E || eps - eps_theta(x_t, t) ||^2 ---
    model = NoisePredictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    losses = []
    for step in range(1, args.steps + 1):
        x0 = data[torch.randint(len(data), (args.batch,))]
        t = torch.randint(0, T, (args.batch,))
        xt, eps = q_sample(x0, t)
        loss = ((eps - model(xt, t)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"step {step:5d} | L_simple {np.mean(losses[-500:]):.4f}")

    plt.figure(figsize=(6, 4))
    k = 50
    plt.plot(np.convolve(losses, np.ones(k) / k, mode="valid"))
    plt.xlabel("step"); plt.ylabel("L_simple (moving average)"); plt.tight_layout()
    plt.savefig("loss.png", dpi=150)

    # --- generation: from pure noise back to a spiral ---
    snaps = p_sample_loop(model, 3000)
    order = sorted(snaps.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(12, 2.8))
    for ax, t in zip(axes, order):
        scatter(ax, snaps[t].numpy(), f"t = {t + 1}")
    scatter(axes[-1], data[:3000].numpy(), "real data")
    fig.suptitle("Reverse process: from noise to new points")
    fig.tight_layout(); fig.savefig("reverse.png", dpi=150)


if __name__ == "__main__":
    main()
