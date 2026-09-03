"""Dibuja los circulos del toro latente y el devanado de cada uno.

Cada panel es una frecuencia dominante: los 97 tokens proyectados sobre su
plano de Fourier, con la poligonal 0 -> 1 -> 2 -> ... -> 96 encima. Esa
poligonal es el poligono estrellado {p/k}: el orden de los enteros no es el
orden del circulo, salta k posiciones por vez.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = 97


def spectrum(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0)).mean(axis=1)


def project(emb, k, p=P):
    centered = emb - emb.mean(axis=0, keepdims=True)
    a = np.arange(p)
    f_k = centered.T @ np.exp(-2j * np.pi * k * a / p)
    u, v = np.real(f_k), np.imag(f_k)
    u = u / np.linalg.norm(u)
    v = v - (v @ u) * u
    v = v / np.linalg.norm(v)
    return centered @ u, centered @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/base")
    ap.add_argument("--n-freqs", type=int, default=6)
    ap.add_argument("--out", default="runs/circles.png")
    ap.add_argument("--winding", action="store_true", default=True)
    args = ap.parse_args()

    emb_dir = os.path.join(args.rundir, "embeddings")
    snaps = sorted(
        int(os.path.basename(q).split("_")[-1].split(".")[0])
        for q in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    emb = np.load(os.path.join(emb_dir, f"emb_a_{snaps[-1]:06d}.npy"))

    power = spectrum(emb)[1:] ** 2
    freqs = sorted((np.argsort(power)[::-1][: args.n_freqs] + 1).tolist())

    cols = 3
    rows = (len(freqs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    a = np.arange(P)
    for ax, k in zip(axes, freqs):
        x, y = project(emb, k)
        # la poligonal en orden de los enteros: ahi se ve el devanado
        ax.plot(np.append(x, x[0]), np.append(y, y[0]),
                "-", lw=0.5, color="#94a3b8", alpha=0.9, zorder=1)
        sc = ax.scatter(x, y, c=a, cmap="twilight", s=26, zorder=2,
                        edgecolors="white", linewidths=0.4)
        for tok in (0, 1, 2):
            ax.annotate(str(tok), (x[tok], y[tok]), fontsize=9, weight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=3)
        r = np.hypot(x, y)
        ax.set_title(f"k = {k}   (cv del radio {r.std(ddof=1) / r.mean():.3f})", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_alpha(0.25)

    for ax in axes[len(freqs):]:
        ax.axis("off")

    fig.colorbar(sc, ax=axes.tolist(), label="token a", shrink=0.6)
    fig.suptitle(
        f"El toro latente, circulo por circulo  ({os.path.basename(args.rundir)}, "
        f"paso {snaps[-1]})\nla poligonal une los enteros en orden 0,1,2,...: "
        f"cada circulo se recorre saltando k posiciones",
        fontsize=12,
    )
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"grafico -> {args.out}")


if __name__ == "__main__":
    main()
