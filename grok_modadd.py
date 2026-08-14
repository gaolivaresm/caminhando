"""Grokking en suma modular: (a + b) mod p, p = 97.

MLP de una capa oculta entrenado a full batch sobre el 40% de los 9409 pares.
El weight decay alto (1.0) es lo que separa la memorizacion de la
generalizacion: train acc llega a ~100% temprano, test acc se queda cerca de
cero un buen rato y despues salta.

Registra cada 100 pasos train/test loss y accuracy, y guarda la matriz de
embeddings (p x d) cada 1000 pasos para poder mirar como se reorganiza.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


P = 97
D_EMB = 128
D_HIDDEN = 256
TRAIN_FRAC = 0.4
LR = 1e-3
WEIGHT_DECAY = 1.0
STEPS = 30000
LOG_EVERY = 100
SNAP_EVERY = 1000
SEED = 0


class MLP(nn.Module):
    """Embeddings separados para a y b, concatenados -> hidden -> ReLU -> p."""

    def __init__(self, p=P, d_emb=D_EMB, d_hidden=D_HIDDEN):
        super().__init__()
        self.emb_a = nn.Embedding(p, d_emb)
        self.emb_b = nn.Embedding(p, d_emb)
        self.fc1 = nn.Linear(2 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, p)

    def forward(self, ab):
        x = torch.cat([self.emb_a(ab[:, 0]), self.emb_b(ab[:, 1])], dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


def make_data(p=P, train_frac=TRAIN_FRAC, seed=SEED):
    """Todos los pares (a, b), barajados con semilla fija y partidos."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)
    y = (x[:, 0] + x[:, 1]) % p

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(train_frac * x.shape[0])
    tr, te = perm[:n_train], perm[n_train:]
    return x[tr], y[tr], x[te], y[te]


def evaluate(model, x, y, loss_fn):
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="runs/base")
    ap.add_argument(
        "--snap-every",
        type=int,
        default=SNAP_EVERY,
        help="cada cuantos pasos guardar embeddings; con full batch y semilla fija "
        "la trayectoria es identica, asi que bajarlo solo densifica el muestreo",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "embeddings"), exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    x_tr, y_tr, x_te, y_te = make_data(train_frac=args.train_frac, seed=args.seed)
    print(f"train: {len(x_tr)} ejemplos | test: {len(x_te)} ejemplos")

    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for step in range(args.steps + 1):
        if step % args.snap_every == 0:
            np.save(
                os.path.join(args.outdir, "embeddings", f"emb_a_{step:06d}.npy"),
                model.emb_a.weight.detach().numpy(),
            )
            np.save(
                os.path.join(args.outdir, "embeddings", f"emb_b_{step:06d}.npy"),
                model.emb_b.weight.detach().numpy(),
            )

        if step % LOG_EVERY == 0:
            model.eval()
            tr_loss, tr_acc = evaluate(model, x_tr, y_tr, loss_fn)
            te_loss, te_acc = evaluate(model, x_te, y_te, loss_fn)
            model.train()
            history.append(
                {
                    "step": step,
                    "train_loss": tr_loss,
                    "test_loss": te_loss,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                }
            )
            if step % 1000 == 0:
                print(
                    f"step {step:6d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                    f"| test loss {te_loss:.4f} acc {te_acc:.4f}",
                    flush=True,
                )

        if step == args.steps:
            break

        opt.zero_grad()
        loss_fn(model(x_tr), y_tr).backward()
        opt.step()

    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "p": P,
                    "d_emb": D_EMB,
                    "d_hidden": D_HIDDEN,
                    "train_frac": args.train_frac,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "steps": args.steps,
                    "seed": args.seed,
                    "optimizer": "AdamW",
                    "batch": "full",
                },
                "history": history,
            },
            f,
            indent=2,
        )
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))
    print(f"listo -> {args.outdir}")


if __name__ == "__main__":
    main()
