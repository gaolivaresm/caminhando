"""Como se completa el toro: de una vez o por partes?

Dos preguntas distintas sobre la misma variedad.

GEOMETRIA. El token a cae en el angulo 2*pi*k*a/p del circulo de frecuencia k.
Recorrer los enteros 0,1,2,... no recorre el circulo en orden: salta k
posiciones por vez. Como p es primo, todo k != 0 genera el ciclo completo, asi
que cada circulo se visita entero pero con el devanado de un poligono
estrellado {p/k} distinto. El orden angular real es la sucesion de multiplos
del inverso modular de k.

TIEMPO. Dos cosas podrian pasar y se distinguen midiendo:

  - los circulos aparecen todos juntos, o hay una secuencia entre ellos;
  - dentro de un circulo, los 97 puntos se acomodan a la vez, o algunos se
    fijan antes que otros.

Para lo segundo se mide, por token, el paso en que su desviacion radial baja
del umbral. Si todos se fijan en el mismo paso, el llenado es homogeneo. Si se
reparten, hay un orden y vale preguntar cual.
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


def project(emb, k, p=P):
    """Coordenadas de los p tokens en el plano de Fourier de la frecuencia k."""
    centered = emb - emb.mean(axis=0, keepdims=True)
    a = np.arange(p)
    f_k = centered.T @ np.exp(-2j * np.pi * k * a / p)
    u, v = np.real(f_k), np.imag(f_k)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return None
    u = u / nu
    v = v - (v @ u) * u
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None
    v = v / nv
    return centered @ u, centered @ v


def radial_deviation(emb, k, p=P):
    """Desviacion relativa del radio de cada token respecto de la mediana."""
    proj = project(emb, k, p)
    if proj is None:
        return None
    r = np.hypot(*proj)
    med = np.median(r)
    return np.abs(r - med) / med


def star_order(k, p=P):
    """Orden angular real de los tokens sobre el circulo de frecuencia k."""
    inv = pow(int(k), -1, p)  # inverso modular: quien cae en la posicion j
    return [(j * inv) % p for j in range(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="runs/dense")
    ap.add_argument("--reference", default="runs/base",
                    help="corrida larga de la que se toman las frecuencias finales")
    ap.add_argument("--n-freqs", type=int, default=6)
    ap.add_argument("--thresh", type=float, default=0.10,
                    help="desviacion radial relativa que cuenta como 'ya sobre el circulo'")
    ap.add_argument("--out", default="runs/sequence.json")
    args = ap.parse_args()

    # las frecuencias que terminan ganando, tomadas del final de la corrida larga
    ref_dir = os.path.join(args.reference, "embeddings")
    ref_snaps = sorted(
        int(os.path.basename(q).split("_")[-1].split(".")[0])
        for q in glob.glob(os.path.join(ref_dir, "emb_a_*.npy"))
    )
    ref_emb = np.load(os.path.join(ref_dir, f"emb_a_{ref_snaps[-1]:06d}.npy"))
    spec = spectrum(ref_emb)
    power = spec[1:] ** 2
    freqs = (np.argsort(power)[::-1][: args.n_freqs] + 1).tolist()
    print(f"frecuencias finales (paso {ref_snaps[-1]}): {sorted(freqs)}")

    print("\n--- geometria: orden angular sobre cada circulo")
    for k in sorted(freqs):
        seq = star_order(k)
        print(f"  k={k:2d}: salto de {k} por token; "
              f"orden angular = {seq[:8]}... (paso angular {pow(int(k), -1, P)})")

    emb_dir = os.path.join(args.dense, "embeddings")
    snaps = sorted(
        int(os.path.basename(q).split("_")[-1].split(".")[0])
        for q in glob.glob(os.path.join(emb_dir, "emb_a_*.npy"))
    )
    embs = {s: np.load(os.path.join(emb_dir, f"emb_a_{s:06d}.npy")) for s in snaps}

    # --- cuando se vuelve circular cada frecuencia
    print(f"\n--- tiempo: paso en que el cv del radio baja de {args.thresh}")
    per_freq = {}
    for k in sorted(freqs):
        cvs = []
        for s in snaps:
            proj = project(embs[s], k)
            r = np.hypot(*proj)
            cvs.append(float(r.std(ddof=1) / r.mean()))
        cvs = np.array(cvs)
        hit = np.where(cvs < args.thresh)[0]
        step = int(snaps[hit[0]]) if len(hit) else None
        per_freq[int(k)] = {"cv": cvs.tolist(), "circular_at": step}
        print(f"  k={k:2d}: {step}")

    hits = [v["circular_at"] for v in per_freq.values() if v["circular_at"] is not None]
    if len(hits) > 1:
        print(f"  -> rango entre la primera y la ultima: {max(hits) - min(hits)} pasos "
              f"(de {min(hits)} a {max(hits)})")

    # --- dentro de un circulo: se acomodan todos juntos?
    print(f"\n--- tiempo: por token, paso en que su desviacion radial baja de {args.thresh}")
    per_token = {}
    for k in sorted(freqs):
        devs = np.array([radial_deviation(embs[s], k) for s in snaps])  # (T, p)
        lock = []
        for a in range(P):
            # el primer cruce no sirve: al principio los embeddings son ruido y
            # un token puede quedar al radio mediano por accidente. Lo que
            # interesa es cuando se QUEDA, o sea el paso siguiente a la ultima
            # vez que estuvo por encima del umbral.
            above = np.where(devs[:, a] >= args.thresh)[0]
            if len(above) == 0:
                lock.append(int(snaps[0]))
            elif above[-1] + 1 < len(snaps):
                lock.append(int(snaps[above[-1] + 1]))
            else:
                lock.append(-1)
        lock = np.array(lock)
        ok = lock[lock > 0]
        per_token[int(k)] = lock.tolist()
        if len(ok):
            print(f"  k={k:2d}: mediana {int(np.median(ok))}, "
                  f"rango [{ok.min()}, {ok.max()}], "
                  f"sd {ok.std(ddof=1):.0f}, tokens sin fijar: {int((lock < 0).sum())}")

    # el orden de fijado, correlaciona con algo del token?
    print("\n--- el orden de fijado dentro de un circulo depende del token?")
    for k in sorted(freqs)[:3]:
        lock = np.array(per_token[k], dtype=float)
        valid = lock > 0
        if valid.sum() < 10:
            continue
        a = np.arange(P)[valid]
        l = lock[valid]
        # correlacion con el valor del token y con su posicion angular
        c_val = np.corrcoef(a, l)[0, 1]
        ang = (k * a) % P
        c_ang = np.corrcoef(ang, l)[0, 1]
        print(f"  k={k:2d}: corr con el entero a = {c_val:+.3f}, "
              f"corr con la posicion angular = {c_ang:+.3f}, "
              f"dispersion relativa = {l.std(ddof=1) / l.mean() * 100:.1f}%")

    with open(args.out, "w") as f:
        json.dump(
            {
                "freqs": sorted(freqs),
                "snapshots": snaps,
                "threshold": args.thresh,
                "circular_at": {str(k): v["circular_at"] for k, v in per_freq.items()},
                "cv_series": {str(k): v["cv"] for k, v in per_freq.items()},
                "token_lock_step": {str(k): v for k, v in per_token.items()},
                "star_order": {str(k): star_order(k) for k in sorted(freqs)},
            },
            f,
            indent=2,
        )
    print(f"\nresultados -> {args.out}")


if __name__ == "__main__":
    main()
