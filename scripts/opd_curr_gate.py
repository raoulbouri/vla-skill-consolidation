"""Update-100 gate of the two-phase curriculum, on the development starts 960000-960039 (never the held-out demo_10..49).
Rule fixed before the run (preregistration.json): PASS if the update-100 student, driving alone, moves the door (<= -0.05 rad) in >= 10/40 episodes and reaches the handle (<= 0.12 m) in >= 20/40, OR finishes >= 50% of takeovers from the teacher's 25% point. Otherwise FAIL: stop and report.
  python3 scripts/opd_curr_gate.py <gate_dir> <student_id>"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

G, SID = Path(sys.argv[1]), sys.argv[2]
L = lambda f: [json.loads(l) for l in open(f)]  # noqa: E731


def rollouts(pid):
    E = L(G / "exp1" / pid / "episodes.jsonl")
    n = len(E)
    return {"episodes": n, "success": sum(e["success"] for e in E), "near": sum(e["first_near"] is not None for e in E), "contact": sum(e["first_contact"] is not None for e in E), "door_moved": sum(e["door_onset"] is not None for e in E), "median_min_dist": float(np.median([e["min_dist"] for e in E])), "door_min_mean": float(np.mean([e["door_min"] for e in E]))}


def takeovers(pid):
    c = defaultdict(lambda: [0, 0])
    for r in L(G / "exp2b" / pid / "takeovers.jsonl"):
        c[str(r["fraction"])][0] += r["success"]
        c[str(r["fraction"])][1] += 1
    return {k: {"success": v[0], "n": v[1]} for k, v in sorted(c.items())}


R = {"student": SID, "rollouts": {p: rollouts(p) for p in ("teacher10d", SID)}, "takeover": {p: takeovers(p) for p in (SID, "teacher10d_resume")}}
A = L(G / "scores" / "exp2a_actions.jsonl")
ck = [k[4:-9] for k in A[0] if k.startswith("ckpt") and k.endswith("_rmse_xyz")]
R["action_error_teacher_states"] = {ph: {f"ckpt{c}_xyz": float(np.mean([r[f"ckpt{c}_rmse_xyz"] for r in A if ph == "all" or r["phase"] == ph])) for c in ck} | {"n": sum(ph == "all" or r["phase"] == ph for r in A)} for ph in ("approach", "near", "opening", "all")}
for s in ("b_teacher_states", "a_student_states"):
    X = L(G / "scores" / f"exp3_{s}.jsonl")
    R[f"kl_{s}"] = {c: {k: float(np.mean([r["kl"][k] for r in X if r["ckpt"] == c])) for k in ("full", "executed", "unexecuted", "padded")} | {"act_xyz": float(np.mean([r["act"]["rmse_xyz"] for r in X if r["ckpt"] == c]))} for c in sorted({r["ckpt"] for r in X})}
s, t25 = R["rollouts"][SID], R["takeover"][SID].get("0.25", {"success": 0, "n": 0})
R["rule"] = {"door_moved_ge_10": s["door_moved"] >= 10, "near_ge_20": s["near"] >= 20, "takeover25_ge_half": t25["n"] > 0 and t25["success"] >= 0.5 * t25["n"]}
R["rule"]["pass"] = (R["rule"]["door_moved_ge_10"] and R["rule"]["near_ge_20"]) or R["rule"]["takeover25_ge_half"]
R["reference_previous_student500_diagnostic_set"] = {"near": 12, "door_moved": 3, "takeover25": "6/20", "note": "starts 950000-950039, not the same starts"}
(G / "gate.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
