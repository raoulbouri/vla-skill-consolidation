"""Stage 1 report (CPU only): influence, old/new interaction, clipping simulations and weight sensitivity from the
per-batch files written by scripts/cl_opd_grad_stage1.py.
  scripts/py-policy scripts/cl_opd_grad_report.py --in outputs/cl_opd/analysis/grad_stage1

Unit = trajectory (observation x noise chain), the gradient the trainer computes on its own. The batch gradient is
g = sum_i a_i g_i with a_i = (1 - w)/n_new for t063 trajectories and w/n_old for t035 ones (1/n_new without replay).
Inner products come from the Gram matrix of sketches with the exact squared norms on its diagonal.
  directional contribution  c_i = a_i <g_i, g> / |g|^2          (sums to 1 over the batch; > 1 or < 0 possible)
  leave-one-out angle       angle(g, g - a_i g_i)
  clipping                  b_i = a_i min(1, tau_i / |g_i|),  C = sum_i b_i g_i
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDENT = {"replay": "outputs/cl_opd/full_20261108T214604Z/k1_t063/student", "noreplay": "outputs/cl_opd/t063_noreplay_20261109T150326Z/k1_t063/student"}
CKPT_U = {"init": 0, "c100": 100, "c200": 200}
CLASSES = ("fresh", "replay", "old")
SCHEMES = [("none", None)] + [(f"batch_median_x{k}", k) for k in (1, 2, 3, 5)] + [(f"class_median_x{k}", k) for k in (2, 3, 5)] + [(f"fixed_p{p}", p) for p in (50, 75, 90, 95, 99)]


def deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def history(student):
    rows = {}
    for f in ("training_metrics.csv", "training_metrics_from_u100.csv"):
        rows.update({int(r["update"]): r for r in csv.DictReader(open(ROOT / student / f))})
    return rows


def load(f):
    z = np.load(f)
    s = json.loads(str(z["summary"]))
    G = np.array(z["gram"], np.float64)
    sq = np.array(z["sq_expert"], np.float64) + np.array(z["sq_heads"], np.float64)
    sketch_diag = np.diag(G).copy()
    np.fill_diagonal(G, sq)  # exact norms on the diagonal
    cls = np.array(["old" if o else "replay" if r else "fresh" for o, r in zip(z["is_old"], z["is_replay"])])
    return {"s": s, "G": G, "norm": np.sqrt(sq), "sketch_diag_rel_err": np.abs(sketch_diag - sq) / sq, "cls": cls, "phase": np.array(z["phase"]), "kl": np.array(z["kl_full"]), "obs": np.array(z["obs"])}


def weights(b, w):
    n_new, n_old = int(np.sum(b["cls"] != "old")), int(np.sum(b["cls"] == "old"))
    if n_old == 0 or w is None:
        return np.where(b["cls"] != "old", 1.0 / n_new, 0.0)
    return np.where(b["cls"] == "old", w / n_old, (1.0 - w) / n_new)


def influence(G, a):
    Ga = G @ a
    gg = float(a @ Ga)
    c = a * Ga / gg
    # |g - a_i g_i|^2 = gg - 2 a_i (Ga)_i + a_i^2 G_ii ;  <g, g - a_i g_i> = gg - a_i (Ga)_i
    loo_sq = gg - 2 * a * Ga + a * a * np.diag(G)
    loo_cos = (gg - a * Ga) / np.sqrt(gg * np.maximum(loo_sq, 1e-300))
    return gg, c, np.degrees(np.arccos(np.clip(loo_cos, -1, 1)))


def taus(b, scheme, k, fixed):
    n = b["norm"]
    if scheme == "none":
        return np.full_like(n, np.inf)
    if scheme.startswith("batch_median"):
        return np.full_like(n, k * np.median(n))
    if scheme.startswith("class_median"):
        t = np.empty_like(n)
        for c in CLASSES:
            m = b["cls"] == c
            if m.any():
                t[m] = k * np.median(n[m])
        return t
    return np.full_like(n, fixed[k])


def clip_eval(b, a, scheme, k, fixed):
    G, n = b["G"], b["norm"]
    t = taus(b, scheme, k, fixed)
    s = np.minimum(1.0, t / n)
    bw = a * s
    gg0, c0, _ = influence(G, a)
    gg, c, loo = influence(G, bw)
    i0 = int(np.argmax(c0))
    a_wo = a.copy()
    a_wo[i0] = 0.0
    cos_orig = float(bw @ G @ a) / np.sqrt(gg * gg0)
    cos_wo_top = float(bw @ G @ a_wo) / np.sqrt(gg * float(a_wo @ G @ a_wo))
    out = {"clip_frac": float(np.mean(s < 1)), "norm": float(np.sqrt(gg)), "norm_over_unclipped": float(np.sqrt(gg / gg0)),
           "max_contrib": float(c.max()), "max_loo_deg": float(loo.max()), "angle_to_unclipped_deg": deg(cos_orig), "angle_to_unclipped_without_top_deg": deg(cos_wo_top)}
    for cl in CLASSES:
        m = b["cls"] == cl
        if m.any():
            out[f"clip_frac_{cl}"] = float(np.mean(s[m] < 1))
            out[f"contrib_{cl}"] = float(c[m].sum())
    return out


def batch_metrics(b, w_actual, fixed):
    a = weights(b, w_actual)
    gg, c, loo = influence(b["G"], a)
    top = int(np.argmax(c))
    m = {"u": b["s"]["u"], "row": b["s"]["row"], "weights": b["s"]["weights"], "exact": b["s"]["exact"], "controller": "teacher" if b["s"]["chains"] == 2 else "student",
         "grad_norm_exact": b["s"]["recomputed"]["grad_norm"], "grad_norm_sketch": float(np.sqrt(gg)), "kl_mean": b["s"]["recomputed"]["kl_mean"],
         "max_contrib": float(c.max()), "top_class": str(b["cls"][top]), "top_phase": str(b["phase"][top]), "top_norm_over_median": float(b["norm"][top] / np.median(b["norm"])),
         "top_kl_over_median": float(b["kl"][top] / np.median(b["kl"])), "max_loo_deg": float(loo.max()), "top_rank_by_norm": int((b["norm"] > b["norm"][top]).sum()) + 1,
         "sketch_diag_rel_err_max": float(b["sketch_diag_rel_err"].max())}
    for cl in CLASSES:
        sel = b["cls"] == cl
        if sel.any():
            m[f"contrib_{cl}"] = float(c[sel].sum())
            m[f"norm_median_{cl}"] = float(np.median(b["norm"][sel]))
            m[f"norm_max_{cl}"] = float(b["norm"][sel].max())
    # exact old/new interaction from the exact sums: S_new = sum of new gradients, S_old = sum of old ones
    es = b["s"]["exact_sums"]
    nn_, oo, no = (sum(es[x].values()) for x in ("new_new", "old_old", "new_old"))
    if oo > 0:
        n_new_t, n_old_t = b["s"]["n_traj"], b["s"]["n_old_traj"]
        m["cos_new_old"] = no / np.sqrt(nn_ * oo)
        for w in (0.5, 0.2):
            cn, co = (1 - w) / n_new_t, w / n_old_t
            tot = cn * cn * nn_ + co * co * oo + 2 * cn * co * no
            m[f"w{w}_new_norm"], m[f"w{w}_old_norm"] = cn * np.sqrt(nn_), co * np.sqrt(oo)
            m[f"w{w}_share_new"] = (cn * cn * nn_ + cn * co * no) / tot  # <cn S_new, g>/|g|^2
            m[f"w{w}_share_old"] = (co * co * oo + cn * co * no) / tot
    for name, k in SCHEMES:
        r = clip_eval(b, a, name.split("_x")[0].split("_p")[0] if not name.startswith("fixed") else "fixed", k, fixed)
        m.update({f"{name}:{x}": v for x, v in r.items()})
    if oo > 0:
        a2 = weights(b, 0.2)
        for name, k in [("none", None), ("batch_median_x3", 3), ("class_median_x3", 3)]:
            r = clip_eval(b, a2, name.split("_x")[0], k, fixed)
            m.update({f"w0.2|{name}:{x}": v for x, v in r.items()})
    return m


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    root = ROOT / a.inp
    plan = json.loads((root / "plan.json").read_text())
    report = {"validation": {}, "batches": {}, "sensitivity": {}, "summary": {}}
    for run in plan:
        report["validation"][run] = json.loads((root / run / "validation.json").read_text())
        hist = history(STUDENT[run])
        spikes = set(plan[run]["spikes"])
        B = {f.stem: load(f) for f in sorted((root / run).glob("u*_*.npz"))}
        # fixed thresholds: percentiles of per-trajectory norms in the ordinary batches at the same weights
        fixed = {}
        for lab in CKPT_U:
            pool = np.concatenate([b["norm"] for b in B.values() if b["s"]["weights"] == lab and b["s"]["u"] in plan[run]["normal"]] or [np.array([np.nan])])
            fixed[lab] = {p: float(np.nanpercentile(pool, p)) for p in (50, 75, 90, 95, 99)}
        rows = []
        for key, b in B.items():
            w = b["s"]["old_weight"]
            m = batch_metrics(b, w, fixed[b["s"]["weights"]])
            h = hist[b["s"]["row"]]
            m["kind"] = "anchor" if b["s"]["u"] in (0, 100, 200) and b["s"]["exact"] else "spike" if b["s"]["u"] in spikes else "normal"
            m["logged_spike"] = b["s"]["u"] in spikes
            m["grad_norm_logged"] = float(h["grad_norm"])
            m["kl_mean_logged"] = float(h["kl_mean"])
            m["weights_distance"] = abs(b["s"]["u"] - CKPT_U[b["s"]["weights"]])
            rows.append(m)
        report["batches"][run] = {"fixed_thresholds": fixed, "rows": rows}
        # sensitivity: the same batch at two weights
        sens = []
        by_u = {}
        for key, b in B.items():
            by_u.setdefault(b["s"]["u"], {})[b["s"]["weights"]] = b
        for u, d in sorted(by_u.items()):
            labs = sorted(d)
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    x, y = d[labs[i]], d[labs[j]]
                    ax, ay = weights(x, x["s"]["old_weight"]), weights(y, y["s"]["old_weight"])
                    _, cx, _ = influence(x["G"], ax)
                    _, cy, _ = influence(y["G"], ay)
                    tx, ty = set(np.argsort(-cx)[:5]), set(np.argsort(-cy)[:5])
                    sens.append({"u": u, "pair": f"{labs[i]}-{labs[j]}", "spearman_norm": spearman(x["norm"], y["norm"]), "spearman_kl": spearman(x["kl"], y["kl"]),
                                 "top1_same": int(np.argmax(cx) == np.argmax(cy)), "top5_overlap": len(tx & ty) / 5, "max_contrib": [float(cx.max()), float(cy.max())],
                                 "grad_norm": [x["s"]["recomputed"]["grad_norm"], y["s"]["recomputed"]["grad_norm"]], "logged": float(hist[u + 1]["grad_norm"]), "spike": u in spikes})
        report["sensitivity"][run] = sens
    out = root / "report.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    summarize(report, root)


def q(v, p):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.percentile(v, p)) if v else float("nan")


def summarize(report, root):
    L = ["# Stage 1 gradient analysis: summary tables", ""]
    for run, val in report["validation"].items():
        L += [f"## Validation ({run}): logged vs recomputed at the exact weights", "", "| row | weights | metric | logged | recomputed | rel. diff |", "|---|---|---|---|---|---|"]
        for v in val:
            for k, c in v["compare"].items():
                L.append(f"| {v['row']} | {v['weights']} | {k} | {c['logged']:.5g} | {c['recomputed']:.5g} | {c['rel_diff']:.1e} |")
        L.append("")
    for run, d in report["batches"].items():
        R = d["rows"]
        L += [f"## {run}: influence by batch kind (weights c100/c200 for U >= 100, all labels pooled)", "", "| kind | weights | n | median grad norm (recomp.) | median max contribution | median max LOO angle | top obs class (fresh/replay/old) |", "|---|---|---|---|---|---|---|"]
        for kind in ("normal", "spike"):
            for lab in ("init", "c100", "c200"):
                r = [x for x in R if x["kind"] in (kind, "anchor") and (x["logged_spike"] == (kind == "spike")) and x["weights"] == lab]
                if not r:
                    continue
                tc = {c: sum(x["top_class"] == c for x in r) for c in CLASSES}
                L.append(f"| {kind} | {lab} | {len(r)} | {q([x['grad_norm_exact'] for x in r], 50):.0f} | {q([x['max_contrib'] for x in r], 50):.3f} | {q([x['max_loo_deg'] for x in r], 50):.1f} | {tc['fresh']}/{tc['replay']}/{tc['old']} |")
        L.append("")
        L += [f"## {run}: clipping schemes (medians over batches; spike vs normal)", "", "| scheme | kind | clip frac | max contribution | max LOO angle | angle to unclipped | angle to unclipped-without-top | |C| / |g| |", "|---|---|---|---|---|---|---|---|"]
        for name, _ in SCHEMES:
            for kind in ("normal", "spike"):
                r = [x for x in R if x["logged_spike"] == (kind == "spike")]
                L.append(f"| {name} | {kind} | {q([x[f'{name}:clip_frac'] for x in r], 50):.2f} | {q([x[f'{name}:max_contrib'] for x in r], 50):.3f} | {q([x[f'{name}:max_loo_deg'] for x in r], 50):.1f} | {q([x[f'{name}:angle_to_unclipped_deg'] for x in r], 50):.1f} | {q([x[f'{name}:angle_to_unclipped_without_top_deg'] for x in r], 50):.1f} | {q([x[f'{name}:norm_over_unclipped'] for x in r], 50):.3f} |")
        L.append("")
        if any("cos_new_old" in x for x in R):
            L += [f"## {run}: old/new interaction (exact sums)", "", "| kind | cos(new, old) median [p10, p90] | w=0.5 share new / old | w=0.2 share new / old | w=0.5 |new| / |old| |", "|---|---|---|---|---|"]
            for kind in ("normal", "spike"):
                r = [x for x in R if x["logged_spike"] == (kind == "spike") and "cos_new_old" in x]
                L.append(f"| {kind} | {q([x['cos_new_old'] for x in r], 50):.3f} [{q([x['cos_new_old'] for x in r], 10):.3f}, {q([x['cos_new_old'] for x in r], 90):.3f}] | {q([x['w0.5_share_new'] for x in r], 50):.2f} / {q([x['w0.5_share_old'] for x in r], 50):.2f} | {q([x['w0.2_share_new'] for x in r], 50):.2f} / {q([x['w0.2_share_old'] for x in r], 50):.2f} | {q([x['w0.5_new_norm'] / x['w0.5_old_norm'] for x in r], 50):.1f} |")
            L.append("")
        S = report["sensitivity"][run]
        L += [f"## {run}: sensitivity to the weights", "", "| pair | n | Spearman(norms) median | top-1 same | top-5 overlap | spikes reproduced (recomp. > 1e4) |", "|---|---|---|---|---|---|"]
        for pair in sorted({s["pair"] for s in S}):
            r = [s for s in S if s["pair"] == pair]
            sp = [s for s in r if s["spike"]]
            L.append(f"| {pair} | {len(r)} | {q([s['spearman_norm'] for s in r], 50):.2f} | {np.mean([s['top1_same'] for s in r]):.2f} | {np.mean([s['top5_overlap'] for s in r]):.2f} | {sum(min(s['grad_norm']) > 1e4 for s in sp)}/{len(sp)} at both |")
        L += ["", "| weights | distance |U - ckpt| | n | recomputed / logged grad norm median [p10, p90] | logged spikes reproduced |", "|---|---|---|---|---|"]
        for lab in ("init", "c100", "c200"):
            for lo, hi in ((0, 0), (1, 25), (26, 60), (61, 120), (121, 250)):
                r = [x for x in R if x["weights"] == lab and lo <= x["weights_distance"] <= hi]
                if not r:
                    continue
                ratio = [x["grad_norm_exact"] / x["grad_norm_logged"] for x in r]
                sp = [x for x in r if x["logged_spike"]]
                L.append(f"| {lab} | {lo}-{hi} | {len(r)} | {q(ratio, 50):.2f} [{q(ratio, 10):.2f}, {q(ratio, 90):.2f}] | {sum(x['grad_norm_exact'] > 1e4 for x in sp)}/{len(sp)} |")
        L.append("")
    (root / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
