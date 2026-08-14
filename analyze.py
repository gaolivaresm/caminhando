"""Curvas de grokking y FFT de los embeddings antes/despues del salto.

Detecta el salto como el primer paso donde test acc cruza 0.9, lo compara con
el paso donde train acc cruza 0.99, y hace FFT sobre las columnas de la matriz
de embeddings en dos momentos: memorizacion pura y post-salto.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = 97


def first_cross(hist, key, thresh):
    for h in hist:
        if h[key] >= thresh:
            return h["step"]
    return None


def plot_curves(hist, outpath, memo_step, grok_step):
    steps = np.array([h["step"] for h in hist])
    tr_acc = np.array([h["train_acc"] for h in hist])
    te_acc = np.array([h["test_acc"] for h in hist])
    tr_loss = np.array([h["train_loss"] for h in hist])
    te_loss = np.array([h["test_loss"] for h in hist])

    x = np.maximum(steps, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x, tr_acc, label="train", color="#2563eb")
    axes[0].plot(x, te_acc, label="test", color="#dc2626")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy")

    axes[1].plot(x, tr_loss, label="train", color="#2563eb")
    axes[1].plot(x, te_loss, label="test", color="#dc2626")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss")

    for ax in axes:
        ax.set_xlabel("paso (escala log)")
        if memo_step is not None:
            ax.axvline(max(memo_step, 1), ls="--", c="#2563eb", alpha=0.4)
        if grok_step is not None:
            ax.axvline(grok_step, ls="--", c="#dc2626", alpha=0.4)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(f"(a+b) mod {P} — memorizacion paso {memo_step}, grokking paso {grok_step}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    print(f"curvas -> {outpath}")


def spectrum(emb):
    """Magnitud media del FFT sobre el eje de los p numeros, por frecuencia.

    Centra cada columna (quita la media sobre los p tokens) para que el DC no
    domine, y promedia |FFT| sobre las 128 dimensiones.
    """
    centered = emb - emb.mean(axis=0, keepdims=True)
    mags = np.abs(np.fft.rfft(centered, axis=0))
    return mags.mean(axis=1)


def plot_fft(emb_dir, steps_to_plot, outpath):
    fig, axes = plt.subplots(1, len(steps_to_plot), figsize=(5.2 * len(steps_to_plot), 4.2))
    if len(steps_to_plot) == 1:
        axes = [axes]

    report = {}
    for ax, step in zip(axes, steps_to_plot):
        emb = np.load(os.path.join(emb_dir, f"emb_a_{step:06d}.npy"))
        spec = spectrum(emb)
        freqs = np.arange(len(spec))
        ax.bar(freqs[1:], spec[1:], color="#7c3aed")
        ax.set_title(f"paso {step}")
        ax.set_xlabel("frecuencia k")
        ax.set_ylabel("|FFT| medio")
        ax.grid(alpha=0.25, axis="y")

        s = spec[1:]
        power = s**2
        power = power / power.sum()
        # participation ratio: cuantas frecuencias participan de verdad
        pr = 1.0 / (power**2).sum()
        top = np.argsort(s)[::-1][:5] + 1
        report[step] = {
            "participation_ratio": float(pr),
            "top5_freqs": [int(f) for f in top],
            "top5_share_of_power": float(power[top - 1].sum()),
        }

    fig.suptitle("Espectro de la matriz de embeddings de a (media sobre 128 dims)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    print(f"fft -> {outpath}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/base")
    args = ap.parse_args()

    with open(os.path.join(args.rundir, "history.json")) as f:
        data = json.load(f)
    hist = data["history"]

    memo_step = first_cross(hist, "train_acc", 0.99)
    grok_step = first_cross(hist, "test_acc", 0.9)
    final = hist[-1]

    print(f"train acc >= 0.99 en paso: {memo_step}")
    print(f"test  acc >= 0.90 en paso: {grok_step}")
    print(f"final: train acc {final['train_acc']:.4f} | test acc {final['test_acc']:.4f}")

    emb_dir = os.path.join(args.rundir, "embeddings")
    plot_curves(hist, os.path.join(args.rundir, "curves.png"), memo_step, grok_step)

    snaps = sorted(
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    if grok_step is not None:
        before = max([s for s in snaps if s <= grok_step - 2000] or [snaps[1]])
        after = min([s for s in snaps if s >= grok_step + 2000] or [snaps[-1]])
    else:
        before, after = snaps[len(snaps) // 4], snaps[-1]
    final_snap = snaps[-1]
    print(f"snapshots comparados: antes={before}, despues={after}, final={final_snap}")

    to_plot = [before, after]
    if final_snap not in to_plot:
        to_plot.append(final_snap)
    report = plot_fft(emb_dir, to_plot, os.path.join(args.rundir, "fft.png"))
    summary = {
        "memorization_step": memo_step,
        "grokking_step": grok_step,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "fft_before_step": before,
        "fft_after_step": after,
        "fft": {str(k): v for k, v in report.items()},
    }
    with open(os.path.join(args.rundir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
