"""Cuando se forma la estructura de Fourier: antes del salto o durante?

Recorre todos los snapshots de embeddings de una corrida y mide, paso a paso,
cuan concentrado esta el espectro. Superpone eso con la curva de test accuracy
para ver cual de las dos se mueve primero.

Dos medidas de concentracion, por si una enganara:

  - participation ratio: cuantas frecuencias participan de verdad. Baja de ~48
    (espectro plano) hacia unas pocas.
  - potencia en las frecuencias que terminan dominando: cuanta de la potencia
    total ya vive, en cada paso, en las k frecuencias top del estado final.

Para comparar tiempos usa el paso donde cada curva cruza la mitad de su propio
recorrido, que no depende de la escala de cada una.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOP_K = 6


def spectrum(emb):
    """Magnitud media del FFT sobre el eje de los p tokens, por frecuencia."""
    centered = emb - emb.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0)).mean(axis=1)


def power_fractions(spec):
    """Potencia normalizada por frecuencia, descartando el DC."""
    power = spec[1:] ** 2
    return power / power.sum()


def participation_ratio(spec):
    frac = power_fractions(spec)
    return 1.0 / (frac**2).sum()


def midpoint_crossing(steps, values, rising=True):
    """Primer paso donde la curva cruza la mitad de su propio rango."""
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return None
    half = lo + 0.5 * (hi - lo)
    hits = np.where(v >= half)[0] if rising else np.where(v <= half)[0]
    return int(steps[hits[0]]) if len(hits) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/base")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb_dir = os.path.join(args.rundir, "embeddings")
    snaps = sorted(
        int(os.path.basename(p).split("_")[-1].split(".")[0])
        for p in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    print(f"{len(snaps)} snapshots, de {snaps[0]} a {snaps[-1]}")

    specs = {s: spectrum(np.load(os.path.join(emb_dir, f"emb_a_{s:06d}.npy"))) for s in snaps}

    # las frecuencias que dominan al final, para rastrearlas hacia atras
    final_frac = power_fractions(specs[snaps[-1]])
    dominant = np.argsort(final_frac)[::-1][:args.top_k]
    print(f"frecuencias dominantes al final (k): {sorted(int(f) + 1 for f in dominant)}")

    steps = np.array(snaps)
    prs = np.array([participation_ratio(specs[s]) for s in snaps])
    shares = np.array([power_fractions(specs[s])[dominant].sum() for s in snaps])

    with open(os.path.join(args.rundir, "history.json")) as f:
        hist = json.load(f)["history"]
    acc_by_step = {h["step"]: h["test_acc"] for h in hist}
    test_acc = np.array([acc_by_step[s] for s in snaps])

    t_acc = midpoint_crossing(steps, test_acc, rising=True)
    t_pr = midpoint_crossing(steps, prs, rising=False)
    t_share = midpoint_crossing(steps, shares, rising=True)
    print(f"cruce del punto medio -> test acc: {t_acc} | PR: {t_pr} | potencia top-{args.top_k}: {t_share}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (l_pr,) = ax.plot(steps, prs, "o-", ms=3, color="#7c3aed", label="participation ratio")
    ax.set_xlabel("paso")
    ax.set_ylabel("participation ratio (frecuencias activas)", color="#7c3aed")
    ax.tick_params(axis="y", labelcolor="#7c3aed")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    (l_acc,) = ax2.plot(steps, test_acc, "-", lw=2, color="#dc2626", label="test accuracy")
    (l_sh,) = ax2.plot(steps, shares, "--", lw=2, color="#059669",
                       label=f"potencia en las top-{args.top_k} finales")
    ax2.set_ylabel("test accuracy  /  fraccion de potencia")
    ax2.set_ylim(-0.03, 1.03)

    for t, c in ((t_pr, "#7c3aed"), (t_share, "#059669"), (t_acc, "#dc2626")):
        if t is not None:
            ax.axvline(t, ls=":", c=c, alpha=0.7)

    handles = [l_pr, l_sh, l_acc]
    ax.legend(handles, [h.get_label() for h in handles], loc="center right")
    ax.set_title(
        f"Estructura espectral vs generalizacion  "
        f"(puntos medios: PR {t_pr}, potencia {t_share}, test acc {t_acc})"
    )
    fig.tight_layout()

    out = args.out or os.path.join(args.rundir, "pr_timeline.png")
    fig.savefig(out, dpi=130)
    print(f"grafico -> {out}")

    result = {
        "n_snapshots": len(snaps),
        "snapshot_stride": int(steps[1] - steps[0]) if len(steps) > 1 else None,
        "dominant_freqs_final": sorted(int(f) + 1 for f in dominant),
        "midpoint_test_acc": t_acc,
        "midpoint_participation_ratio": t_pr,
        "midpoint_top_power": t_share,
        "series": [
            {
                "step": int(s),
                "participation_ratio": float(pr),
                "top_power_share": float(sh),
                "test_acc": float(ta),
            }
            for s, pr, sh, ta in zip(steps, prs, shares, test_acc)
        ],
    }
    with open(os.path.join(args.rundir, "pr_timeline.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'paso':>6} {'PR':>7} {'pot.top':>8} {'test acc':>9}")
    for s, pr, sh, ta in zip(steps, prs, shares, test_acc):
        print(f"{s:6d} {pr:7.2f} {sh:8.3f} {ta:9.4f}")


if __name__ == "__main__":
    main()
