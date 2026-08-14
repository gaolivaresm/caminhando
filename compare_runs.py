"""Compara trayectorias de participation ratio entre varias corridas.

Sirve para dos preguntas distintas:

  - control de etiquetas barajadas: si el PR baja igual con etiquetas
    permutadas, su caida mide presion del regularizador y no estructura
    emergente, y el titular se cae.
  - replicacion sobre K semillas independientes: si semillas distintas eligen
    frecuencias distintas pero convergen al mismo PR asintotico, eso es
    invariancia del *numero* efectivo de frecuencias bajo cambio de base.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#7c3aed", "#dc2626", "#059669", "#d97706", "#0891b2", "#be185d", "#4d7c0f"]


def spectrum(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0)).mean(axis=1)


def power_fractions(spec):
    power = spec[1:] ** 2
    return power / power.sum()


def participation_ratio(spec):
    frac = power_fractions(spec)
    return 1.0 / (frac**2).sum()


def load_run(rundir, top_k=6):
    emb_dir = os.path.join(rundir, "embeddings")
    snaps = sorted(
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    prs, steps = [], []
    for s in snaps:
        spec = spectrum(np.load(os.path.join(emb_dir, f"emb_a_{s:06d}.npy")))
        steps.append(s)
        prs.append(participation_ratio(spec))

    final_spec = spectrum(np.load(os.path.join(emb_dir, f"emb_a_{snaps[-1]:06d}.npy")))
    final_frac = power_fractions(final_spec)
    top = np.argsort(final_frac)[::-1][:top_k]

    with open(os.path.join(rundir, "history.json")) as f:
        blob = json.load(f)
    accs = {h["step"]: h["test_acc"] for h in blob["history"]}

    return {
        "name": os.path.basename(rundir.rstrip("/")),
        "config": blob["config"],
        "steps": np.array(steps),
        "prs": np.array(prs),
        "final_pr": float(prs[-1]),
        "final_test_acc": blob["history"][-1]["test_acc"],
        "top_freqs": sorted(int(f) + 1 for f in top),
        "top_share": float(final_frac[top].sum()),
        "test_acc": np.array([accs[s] for s in steps]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--out", default="runs/compare.png")
    ap.add_argument("--title", default="Trayectorias de participation ratio")
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    runs = [load_run(r, args.top_k) for r in args.rundirs]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for i, r in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(r["steps"], r["prs"], "-", lw=1.8, color=c, label=r["name"])
        axes[1].plot(r["steps"], r["test_acc"], "-", lw=1.8, color=c, label=r["name"])

    axes[0].set_ylabel("participation ratio")
    axes[0].set_title("Concentracion espectral")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Generalizacion")
    axes[1].axhline(1 / 97, ls="--", c="#666", alpha=0.7, lw=1)
    for ax in axes:
        ax.set_xlabel("paso")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"grafico -> {args.out}")

    print(f"\n{'corrida':<14} {'shuf':>5} {'PR final':>9} {'test acc':>9} {'pot.top':>8}  frecuencias")
    for r in runs:
        shuf = "si" if r["config"].get("shuffle_labels") else "no"
        print(
            f"{r['name']:<14} {shuf:>5} {r['final_pr']:9.2f} {r['final_test_acc']:9.4f} "
            f"{r['top_share']:8.3f}  {r['top_freqs']}"
        )

    prs = np.array([r["final_pr"] for r in runs])
    if len(prs) > 1:
        print(f"\nPR final: media {prs.mean():.2f}, sd {prs.std(ddof=1):.2f}, "
              f"cv {prs.std(ddof=1) / prs.mean() * 100:.1f}%, "
              f"rango [{prs.min():.2f}, {prs.max():.2f}]")

    summary = [
        {
            "name": r["name"],
            "seed": r["config"].get("seed"),
            "shuffle_labels": r["config"].get("shuffle_labels"),
            "final_pr": r["final_pr"],
            "final_test_acc": r["final_test_acc"],
            "top_freqs": r["top_freqs"],
            "top_share": r["top_share"],
        }
        for r in runs
    ]
    out_json = os.path.splitext(args.out)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"resumen -> {out_json}")


if __name__ == "__main__":
    main()
