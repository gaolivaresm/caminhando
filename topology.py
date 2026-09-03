"""Topologia latente: son circulos de verdad, y en el orden correcto?

El espectro concentrado dice que quedan pocas frecuencias, pero no dice que la
variedad sea circular. Se prueban tres observables, y no los tres sirven:

  1. Uniformidad del radio. Proyectando los p tokens sobre el plano de Fourier
     de la frecuencia k, un circulo tiene todos los puntos a igual distancia
     del centro. El cv del radio DISCRIMINA fuerte: ~0.03 con estructura contra
     ~0.43 sin ella.

  2. Orden angular. No alcanza con caer en un circulo: hay que caer en el orden
     2*pi*k*a/p. El estadistico es el parametro de orden

         R = | (1/p) * sum_a exp( i * (theta_a -+ 2*pi*k*a/p) ) |

     OJO con el nulo. La tentacion es usar P(R >= r) = exp(-p*r^2), que supone
     angulos uniformes e independientes. Es el nulo equivocado: el plano de
     proyeccion se construye A PARTIR de la frecuencia k, asi que hasta una
     matriz gaussiana sale con R ~ 0.80. Contra cero, cualquier cosa da p = 0.
     Contra el nulo empirico, R discrimina, pero mucho menos de lo que parece.

  3. Degeneracion del espectro de PCA. Un circulo aporta dos direcciones de
     igual varianza, asi que C circulos deberian dar C pares de autovalores
     casi iguales. NO DISCRIMINA: los autovalores vecinos de una matriz
     aleatoria ya son parecidos de por si (0.95 en el nulo contra 0.99 con
     estructura). Se deja reportado como resultado negativo.

Todo se compara contra dos referencias: el control de etiquetas barajadas y un
ensemble gaussiano de la misma forma.
"""

import argparse
import glob
import json
import os

import numpy as np

P = 97


def spectrum(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    return np.abs(np.fft.rfft(centered, axis=0)).mean(axis=1)


def power_fractions(spec):
    power = spec[1:] ** 2
    return power / power.sum()


def participation_ratio(spec):
    frac = power_fractions(spec)
    return 1.0 / (frac**2).sum()


def pca_eigenvalues(emb):
    """Varianza por componente principal de la matriz centrada."""
    centered = emb - emb.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    return s**2 / (emb.shape[0] - 1)


def pair_degeneracy(eigs, n_pairs):
    """Cuan iguales son los autovalores dentro de cada par consecutivo.

    Devuelve la razon menor/mayor de cada par: 1.0 es degeneracion perfecta,
    que es la firma de una direccion circular.
    """
    out = []
    for i in range(n_pairs):
        a, b = eigs[2 * i], eigs[2 * i + 1]
        out.append(float(min(a, b) / max(a, b)))
    return out


def circle_test(emb, k, p=P):
    """Proyecta sobre el plano de Fourier de la frecuencia k y mide orden.

    Devuelve el parametro de orden R, su p-valor analitico, y el coeficiente
    de variacion de los radios (0 seria un circulo perfecto).
    """
    centered = emb - emb.mean(axis=0, keepdims=True)
    a = np.arange(p)
    phase = np.exp(-2j * np.pi * k * a / p)
    f_k = centered.T @ phase  # vector complejo de dimension d

    u, v = np.real(f_k), np.imag(f_k)
    # ortonormaliza el plano para que los angulos no queden sesgados por una
    # base oblicua, que inventaria elipticidad donde no la hay
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return None
    u = u / nu
    v = v - (v @ u) * u
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None
    v = v / nv

    x, y = centered @ u, centered @ v
    theta = np.arctan2(y, x)
    radii = np.hypot(x, y)

    # La orientacion del plano proyectado es una convencion, no un dato: con
    # u = Re(F_k) y v = Im(F_k) el recorrido sale en sentido horario, asi que
    # theta_a = -2*pi*k*a/p. Se evaluan los dos sentidos y se toma el mejor;
    # quedarse con uno solo mide el signo de la convencion, no la estructura.
    expected = 2 * np.pi * k * a / p
    cands = {
        s: float(np.abs(np.mean(np.exp(1j * (theta - s * expected)))))
        for s in (1, -1)
    }
    orientation = max(cands, key=cands.get)
    R = cands[orientation]
    # NO se reporta el p-valor analitico exp(-p*R^2): ese nulo supone angulos
    # uniformes e independientes, y aca no lo son. El plano de proyeccion se
    # construye A PARTIR de la frecuencia k, asi que hasta una matriz sin
    # ninguna estructura sale ordenada en esa frecuencia -- el nulo empirico da
    # R ~ 0.80, no 0. Comparar contra cero convertia un confundido en un
    # p-valor de cero. El nulo correcto es el ensemble gaussiano.
    return {
        "freq": int(k),
        "order_parameter": R,
        "orientation": int(orientation),
        "radius_cv": float(radii.std(ddof=1) / radii.mean()),
        "variance_share": float((x**2 + y**2).sum() / (centered**2).sum()),
    }


def analyze(rundir, n_freqs=6, verbose=True):
    emb_dir = os.path.join(rundir, "embeddings")
    snaps = sorted(
        int(os.path.basename(q).split("_")[-1].split(".")[0])
        for q in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    emb = np.load(os.path.join(emb_dir, f"emb_a_{snaps[-1]:06d}.npy"))

    with open(os.path.join(rundir, "history.json")) as f:
        cfg = json.load(f)["config"]

    spec = spectrum(emb)
    pr = participation_ratio(spec)
    frac = power_fractions(spec)
    top = np.argsort(frac)[::-1][:n_freqs] + 1

    eigs = pca_eigenvalues(emb)
    n_pairs = max(1, int(round(pr)))
    ratios = pair_degeneracy(eigs, n_pairs)
    tail = eigs[2 * n_pairs] / eigs[0] if len(eigs) > 2 * n_pairs else float("nan")

    circles = [c for c in (circle_test(emb, int(k)) for k in top) if c is not None]

    if verbose:
        name = os.path.basename(rundir.rstrip("/"))
        shuf = " [CONTROL barajado]" if cfg.get("shuffle_labels") else ""
        print(f"\n=== {name}{shuf}  (paso {snaps[-1]}, PR = {pr:.2f})")
        print(f"  PCA: {n_pairs} pares esperados; razon dentro de cada par "
              f"(1.0 = circulo): {[f'{r:.3f}' for r in ratios]}")
        print(f"  autovalor {2 * n_pairs + 1} relativo al primero: {tail:.4f}")
        print(f"  {'k':>4} {'R':>7} {'cv radio':>9} {'% var':>7}")
        for c in circles:
            print(f"  {c['freq']:4d} {c['order_parameter']:7.4f} "
                  f"{c['radius_cv']:9.3f} {c['variance_share'] * 100:6.1f}%")

    return {
        "name": os.path.basename(rundir.rstrip("/")),
        "shuffle_labels": bool(cfg.get("shuffle_labels")),
        "seed": cfg.get("seed"),
        "step": snaps[-1],
        "participation_ratio": float(pr),
        "n_pairs_expected": n_pairs,
        "pair_ratios": ratios,
        "eigenvalue_tail_ratio": float(tail),
        "circles": circles,
    }


def analyze_matrix(emb, name, n_freqs=6):
    """Mismo pipeline que analyze(), sobre una matriz suelta."""
    spec = spectrum(emb)
    pr = participation_ratio(spec)
    frac = power_fractions(spec)
    top = np.argsort(frac)[::-1][:n_freqs] + 1
    eigs = pca_eigenvalues(emb)
    n_pairs = max(1, int(round(pr)))
    return {
        "name": name,
        "shuffle_labels": False,
        "participation_ratio": float(pr),
        "n_pairs_expected": n_pairs,
        "pair_ratios": pair_degeneracy(eigs, min(n_pairs, len(eigs) // 2)),
        "circles": [c for c in (circle_test(emb, int(k)) for k in top) if c is not None],
    }


def null_ensemble(n_draws, n_freqs=6, p=P, d=128, seed=0):
    """El mismo analisis sobre matrices gaussianas, que no tienen estructura.

    Es el nulo que dice cuales de los tres observables discriminan de verdad y
    cuales reportan lo mismo con o sin estructura.
    """
    rng = np.random.default_rng(seed)
    return [
        analyze_matrix(rng.standard_normal((p, d)), f"null{i}", n_freqs)
        for i in range(n_draws)
    ]


def stats(group, key):
    return np.array([c[key] for r in group for c in r["circles"]])


def describe(group, label, null=None):
    if not group:
        return
    Rs = np.array([c["order_parameter"] for r in group for c in r["circles"]])
    cvs = np.array([c["radius_cv"] for r in group for c in r["circles"]])
    shares = np.array([c["variance_share"] for r in group for c in r["circles"]])
    pairs = np.array([x for r in group for x in r["pair_ratios"]])
    def z(vals, key, sign):
        if null is None:
            return ""
        ref = stats(null, key)
        return f"   z = {sign * (vals.mean() - ref.mean()) / ref.std(ddof=1):+7.1f}"

    print(f"\n{label}: {len(group)} corridas, {len(Rs)} frecuencias")
    print(f"  parametro de orden R : {Rs.mean():.4f}  "
          f"[{Rs.min():.4f}, {Rs.max():.4f}]{z(Rs, 'order_parameter', 1)}")
    print(f"  cv del radio         : {cvs.mean():.4f}  "
          f"[{cvs.min():.4f}, {cvs.max():.4f}]{z(cvs, 'radius_cv', -1)}")
    print(f"  varianza por frec.   : {shares.mean() * 100:.1f}%"
          f"{z(shares, 'variance_share', 1)}")
    print(f"  degeneracion PCA     : {pairs.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--n-freqs", type=int, default=6)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--out", default="runs/topology.json")
    args = ap.parse_args()

    results = [analyze(r, args.n_freqs) for r in args.rundirs]

    real = [r for r in results if not r["shuffle_labels"]]
    ctrl = [r for r in results if r["shuffle_labels"]]

    nulls = null_ensemble(args.n_null, args.n_freqs)

    print("\n--- resumen: que observable discrimina y cual no")
    describe(real, "semillas reales", null=nulls)
    describe(ctrl, "control barajado", null=nulls)
    describe(nulls, "nulo gaussiano")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresultados -> {args.out}")


if __name__ == "__main__":
    main()
