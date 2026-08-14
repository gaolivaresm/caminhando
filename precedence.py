"""Precedencia calibrada: cuantas sigmas sobre su propio ruido esta cada observable.

La comparacion ingenua ("PR cae 1% bajo su maximo" contra "test acc supera
1/97") no dice nada, porque cada umbral es el piso de ruido de un instrumento
distinto y esos pisos no estan calibrados entre si. Un proceso continuo visto
con dos instrumentos de distinta resolucion siempre "empieza antes" en el mas
fino.

Aca cada observable se convierte a z-score contra un nulo explicito:

  - test accuracy: binomial con n = tamaño del test y p = 1/p_clases. El error
    estandar es sqrt(p(1-p)/n).
  - participation ratio: ensemble de matrices gaussianas p x d, que da la
    distribucion muestral del estadistico cuando no hay ninguna estructura.
    Es el nulo correcto para "cuanto puede moverse el PR por puro azar".

Con las dos series en sigmas, la pregunta "quien se movio primero" tiene
respuesta: el primer paso donde cada una cruza el mismo umbral de z.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = 97
D_EMB = 128


def spectrum(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0)).mean(axis=1)


def participation_ratio(spec):
    power = spec[1:] ** 2
    frac = power / power.sum()
    return 1.0 / (frac**2).sum()


def pr_null(n_draws, p=P, d=D_EMB, seed=0):
    """Distribucion muestral del PR para matrices sin ninguna estructura."""
    rng = np.random.default_rng(seed)
    return np.array(
        [participation_ratio(spectrum(rng.standard_normal((p, d)))) for _ in range(n_draws)]
    )


def first_crossing(steps, z, thresh):
    hits = np.where(z >= thresh)[0]
    return int(steps[hits[0]]) if len(hits) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/dense")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--z-thresh", type=float, default=3.0)
    args = ap.parse_args()

    with open(os.path.join(args.rundir, "pr_timeline.json")) as f:
        series = json.load(f)["series"]
    with open(os.path.join(args.rundir, "history.json")) as f:
        cfg = json.load(f)["config"]

    steps = np.array([r["step"] for r in series])
    prs = np.array([r["participation_ratio"] for r in series])
    accs = np.array([r["test_acc"] for r in series])

    n_test = int(round((1 - cfg["train_frac"]) * P * P))
    p_chance = 1.0 / P
    acc_se = np.sqrt(p_chance * (1 - p_chance) / n_test)
    z_acc = (accs - p_chance) / acc_se

    null = pr_null(args.n_null)
    pr_mu, pr_sd = null.mean(), null.std(ddof=1)
    # el PR baja, asi que el z se mide hacia abajo para que crecer sea "mas señal"
    z_pr = (pr_mu - prs) / pr_sd

    print(f"nulo de test acc : n={n_test}, p={p_chance:.5f}, se={acc_se:.6f}")
    print(f"nulo de PR       : {args.n_null} matrices gaussianas {P}x{D_EMB}, "
          f"media {pr_mu:.3f}, sd {pr_sd:.3f}")
    print(f"PR observado en el paso 0: {prs[0]:.3f}  (z={z_pr[0]:+.2f})")

    t = args.z_thresh
    c_acc = first_crossing(steps, z_acc, t)
    c_pr = first_crossing(steps, z_pr, t)
    print(f"\nprimer cruce de z={t}:  test acc -> {c_acc} | PR -> {c_pr}")
    if c_acc is not None and c_pr is not None:
        lead = c_acc - c_pr
        quien = "el PR" if lead > 0 else "la test accuracy"
        print(f"ventaja: {quien} por {abs(lead)} pasos")
    for extra in (5.0, 10.0):
        print(f"  z={extra}: test acc -> {first_crossing(steps, z_acc, extra)} "
              f"| PR -> {first_crossing(steps, z_pr, extra)}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(steps, z_pr, "o-", ms=3, color="#7c3aed", label="participation ratio (z)")
    ax.plot(steps, z_acc, "-", lw=2, color="#dc2626", label="test accuracy (z)")
    ax.axhline(t, ls="--", c="#111", alpha=0.5, label=f"z = {t}")
    for c, col in ((c_pr, "#7c3aed"), (c_acc, "#dc2626")):
        if c is not None:
            ax.axvline(c, ls=":", c=col, alpha=0.7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("paso")
    ax.set_ylabel("sigmas sobre el nulo propio de cada observable")
    ax.set_title("Precedencia calibrada: ambas series en unidades de su propio ruido")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(args.rundir, "precedence.png")
    fig.savefig(out, dpi=130)
    print(f"grafico -> {out}")

    result = {
        "n_test": n_test,
        "acc_null": {"p_chance": p_chance, "se": float(acc_se)},
        "pr_null": {"n_draws": args.n_null, "mean": float(pr_mu), "sd": float(pr_sd)},
        "z_threshold": t,
        "crossing_test_acc": c_acc,
        "crossing_pr": c_pr,
        "series": [
            {"step": int(s), "z_pr": float(zp), "z_acc": float(za)}
            for s, zp, za in zip(steps, z_pr, z_acc)
        ],
    }
    with open(os.path.join(args.rundir, "precedence.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'paso':>5} {'z(PR)':>9} {'z(acc)':>9}")
    for s, zp, za in zip(steps, z_pr, z_acc):
        if s % 200 == 0 and s <= 4000:
            print(f"{s:5d} {zp:9.2f} {za:9.2f}")


if __name__ == "__main__":
    main()
