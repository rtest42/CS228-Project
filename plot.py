#!/usr/bin/env python3
import csv, math, re, sys
from pathlib import Path
import matplotlib.pyplot as plt

T = int(sys.argv[1]) if len(sys.argv) > 1 else 40   # match if score >= T

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def load(path):
    g, i = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s = int(r["score"]); g[s] = int(r["genuine"]); i[s] = int(r["impostor"])
    return g, i

def rates(g, i, t):
    G, I = sum(g.values()), sum(i.values())
    fn = sum(v for s, v in g.items() if s < t)
    fp = sum(v for s, v in i.items() if s >= t)
    return fn, G, fp, I

data, clean = {}, None
for p in sorted(Path("results").glob("*.csv")):
    g, i = load(p)
    m = re.fullmatch(r"CASIA-(crop|rwin|blur|noise)([\d.]+)", p.stem)
    if m:
        data.setdefault(m.group(1), []).append((float(m.group(2)), g, i))
    elif p.stem == "CASIA-FingerprintV5":
        clean = (g, i)

fam = [("crop", "Centered crop", "Area kept (%)", 100.0),
       ("rwin", "Random window", "Area kept (%)", 100.0),
       ("blur", "Gaussian blur", "Sigma (px)", 0.0),
       ("noise", "Gaussian noise", "Sigma (gray levels)", 0.0)]

# --- Figure 1: FNMR and FMR vs level, with a CSV summary ---
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
rows = [("family", "level", "FNMR", "FNMR_lo", "FNMR_hi", "FMR", "false_matches")]
styles = {}
for key, title, xlabel, clean_x in fam:
    pts = sorted(data.get(key, []))
    if clean:
        pts = pts + [(clean_x, *clean)]
        pts.sort()
    xs, fn_, lo, hi, fm = [], [], [], [], []
    for lvl, g, i in pts:
        fn, G, fp, I = rates(g, i, T)
        a, b = wilson(fn, G)
        xs.append(lvl); fn_.append(100 * fn / G); lo.append(100 * a); hi.append(100 * b); fm.append(fp)
        rows.append((key, lvl, fn / G, a, b, fp / I, fp))
    col = 0 if key in ("crop", "rwin") else (1 if key == "blur" else 2)
    for a_ in (ax[0][col], ax[1][col]):
        a_.set_xlabel(xlabel)
    ax[0][col].errorbar(xs, fn_, yerr=[[f - l for f, l in zip(fn_, lo)], [h - f for f, h in zip(fn_, hi)]],
                        marker="o", capsize=3, label=title)
    ax[1][col].plot(xs, [max(f, 0.5) for f in fm], marker="o", label=title)

names = ["Area-based (crop vs window)", "Blur", "Noise"]
for c in range(3):
    ax[0][c].set_title(f"FNMR at threshold {T}: {names[c]}")
    ax[0][c].set_ylabel("FNMR (%)"); ax[0][c].set_ylim(0, 100); ax[0][c].grid(alpha=.3); ax[0][c].legend()
    ax[1][c].set_title(f"False matches at threshold {T}"); ax[1][c].set_ylabel("count (0 drawn at 0.5)")
    ax[1][c].set_yscale("log"); ax[1][c].grid(alpha=.3); ax[1][c].legend()
for a_ in (ax[0][0], ax[1][0]):
    a_.invert_xaxis()
fig.tight_layout(); fig.savefig("curves.png", dpi=150)

with open("summary.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

# --- Figure 2: DET curves ---
fig2, axs = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
for a_, (key, title, _, _) in zip(axs, fam):
    for lvl, g, i in sorted(data.get(key, [])):
        G, I = sum(g.values()), sum(i.values())
        top = max(max(g), max(i)) + 2
        fnmr, fmr, fn_c, fp_c = [], [], 0, sum(i.values())
        for t in range(top):
            fn_c = sum(v for s, v in g.items() if s < t)
            fp_c = sum(v for s, v in i.items() if s >= t)
            fnmr.append(fn_c / G); fmr.append(max(fp_c / I, 1e-9))
        a_.plot(fmr, fnmr, label=f"{lvl:g}")
    a_.set_xscale("log"); a_.set_xlim(1e-7, 1); a_.set_title(title); a_.set_xlabel("FMR"); a_.grid(alpha=.3)
    a_.legend(fontsize=7)
axs[0].set_ylabel("FNMR")
fig2.tight_layout(); fig2.savefig("det.png", dpi=150)
print("wrote curves.png, det.png, summary.csv")
