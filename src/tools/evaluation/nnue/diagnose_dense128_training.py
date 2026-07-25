#!/usr/bin/env python3
"""Compare simple 128-input neural nets on the existing NNUE sample cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


BIT_SHIFTS = np.arange(64, dtype=np.uint64)
N_PHASES = 60
ACTIVATION_CLIP_FLOAT = 127.0 / 16.0


def make_feature_matrix(player: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    p = ((player[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    o = ((opponent[:, None] >> BIT_SHIFTS[None, :]) & 1).astype(np.float32, copy=False)
    return np.concatenate([p, o], axis=1)


class PlainDense(nn.Module):
    def __init__(self, phase_head: bool, clipped: bool):
        super().__init__()
        self.phase_head = phase_head
        self.clipped = clipped
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 32)
        if phase_head:
            self.out_w = nn.Parameter(torch.empty(N_PHASES, 32))
            self.out_b = nn.Parameter(torch.zeros(N_PHASES))
        else:
            self.out = nn.Linear(32, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        if self.phase_head:
            nn.init.normal_(self.out_w, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out.bias)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        if self.clipped:
            x = torch.clamp(x, 0.0, ACTIVATION_CLIP_FLOAT)
        return x

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        if self.phase_head:
            return (x * self.out_w[phase]).sum(dim=1) + self.out_b[phase]
        return self.out(x).squeeze(1)


class CurrentNNUEShape(nn.Module):
    def __init__(self, ft_dim: int):
        super().__init__()
        self.ft_bias = nn.Parameter(torch.zeros(ft_dim))
        self.ft_weight = nn.Parameter(torch.empty(128, ft_dim))
        self.hidden1 = nn.Linear(ft_dim * 2, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.out_w = nn.Parameter(torch.empty(N_PHASES, 32))
        self.out_b = nn.Parameter(torch.zeros(N_PHASES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ft_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.hidden1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.normal_(self.hidden2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.normal_(self.out_w, mean=0.0, std=0.02)

    @staticmethod
    def act(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, ACTIVATION_CLIP_FLOAT)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        stm = self.ft_bias + x @ self.ft_weight
        swapped = torch.cat([x[:, 64:], x[:, :64]], dim=1)
        non_stm = self.ft_bias + swapped @ self.ft_weight
        y = torch.cat([self.act(stm), self.act(non_stm)], dim=1)
        y = self.act(self.hidden1(y))
        y = self.act(self.hidden2(y))
        return (y * self.out_w[phase]).sum(dim=1) + self.out_b[phase]


@torch.no_grad()
def metrics(model: nn.Module, x: torch.Tensor, phase: torch.Tensor, target: torch.Tensor, batch_size: int) -> tuple[float, float]:
    model.eval()
    se = 0.0
    ae = 0.0
    seen = 0
    for start in range(0, target.numel(), batch_size):
        end = min(start + batch_size, target.numel())
        pred = model(x[start:end], phase[start:end])
        err = pred - target[start:end]
        se += float((err * err).sum().detach().cpu())
        ae += float(err.abs().sum().detach().cpu())
        seen += end - start
    return se / seen, ae / seen


def train_one(name: str, model: nn.Module, data: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    print(f"model {name}", flush=True)
    val_mse, val_mae = metrics(model, data["val_x"], data["val_phase"], data["val_y"], args.batch_size)
    print(f"initial val_mse {val_mse:.6f} val_mae {val_mae:.6f}", flush=True)
    n = data["train_y"].numel()
    best_mae = val_mae
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        train_loss_sum = 0.0
        seen = 0
        for start in range(0, n, args.batch_size):
            idx = torch.from_numpy(order[start:start + args.batch_size]).to(data["train_y"].device)
            pred = model(data["train_x"][idx], data["train_phase"][idx])
            loss = torch.mean((pred - data["train_y"][idx]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(idx.numel())
            train_loss_sum += float(loss.detach().cpu()) * batch_n
            seen += batch_n
        val_mse, val_mae = metrics(model, data["val_x"], data["val_phase"], data["val_y"], args.batch_size)
        best_mae = min(best_mae, val_mae)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch} train_epoch_mse {train_loss_sum / seen:.6f} "
                f"val_mse {val_mse:.6f} val_mae {val_mae:.6f} best_val_mae {best_mae:.6f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cache", default="model/nnue_records223plus_train10m_val1m_seed20260725_samples.npz")
    parser.add_argument("--train-samples", type=int, default=500_000)
    parser.add_argument("--val-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--phase", type=int, default=-1)
    parser.add_argument("--train-phase-min", type=int, default=-1)
    parser.add_argument("--train-phase-max", type=int, default=59)
    parser.add_argument("--val-phase", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    loaded = np.load(Path(args.sample_cache))
    if args.phase >= 0:
        train_idx = np.flatnonzero(loaded["train_phase"] == args.phase)[:args.train_samples]
        val_idx = np.flatnonzero(loaded["val_phase"] == args.phase)[:args.val_samples]
    else:
        if args.train_phase_min >= 0:
            train_mask = (loaded["train_phase"] >= args.train_phase_min) & (loaded["train_phase"] <= args.train_phase_max)
            train_idx = np.flatnonzero(train_mask)[:args.train_samples]
        else:
            train_idx = np.arange(min(args.train_samples, loaded["train_phase"].shape[0]))
        if args.val_phase >= 0:
            val_idx = np.flatnonzero(loaded["val_phase"] == args.val_phase)[:args.val_samples]
        else:
            val_idx = np.arange(min(args.val_samples, loaded["val_phase"].shape[0]))
    train_x = make_feature_matrix(loaded["train_player"][train_idx], loaded["train_opponent"][train_idx])
    val_x = make_feature_matrix(loaded["val_player"][val_idx], loaded["val_opponent"][val_idx])
    data = {
        "train_x": torch.from_numpy(train_x).to(device),
        "train_phase": torch.from_numpy(loaded["train_phase"][train_idx].astype(np.int64)).to(device),
        "train_y": torch.from_numpy(loaded["train_score"][train_idx].astype(np.float32)).to(device),
        "val_x": torch.from_numpy(val_x).to(device),
        "val_phase": torch.from_numpy(loaded["val_phase"][val_idx].astype(np.int64)).to(device),
        "val_y": torch.from_numpy(loaded["val_score"][val_idx].astype(np.float32)).to(device),
    }
    print(
        f"cache {args.sample_cache} train {data['train_y'].numel()} val {data['val_y'].numel()} "
        f"phase {args.phase} train_phase_min {args.train_phase_min} "
        f"train_phase_max {args.train_phase_max} val_phase {args.val_phase} "
        f"device {device} batch_size {args.batch_size} epochs {args.epochs}",
        flush=True,
    )
    torch.manual_seed(args.seed)
    train_one("plain_dense32_relu_shared_output", PlainDense(phase_head=False, clipped=False).to(device), data, args)
    torch.manual_seed(args.seed)
    train_one("plain_dense32_clipped_shared_output", PlainDense(phase_head=False, clipped=True).to(device), data, args)
    torch.manual_seed(args.seed)
    train_one("plain_dense32_relu_phase_output", PlainDense(phase_head=True, clipped=False).to(device), data, args)
    torch.manual_seed(args.seed)
    train_one("current_nnue_shape_ft32", CurrentNNUEShape(32).to(device), data, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
