"""Does piecewise-uniform frame spacing bias the angular-kinematics maxima?

Run: ``uv run python -m evals.frame_spacing_probe``

Clip frame spacing is constant within a phase but per-phase, with a 2.0-2.5x
straddle gap at every phase handover. ``analysis/gesture.py`` divides by real
``frame.time_s`` deltas so it is not naively wrong, but a finite difference over
an interval 2.4x wider than its neighbours is a biased estimate, and jerk is the
third derivative.

Discriminate estimator bias from a real physical transient.

groundtruth's point: a phase handover is also where motion genuinely changes
direction, so peak jerk sitting on the widest interval is expected under BOTH
hypotheses. The discriminator is to resample the clip onto genuinely uniform
time spacing (slerp between the same rotations) and recompute. If the peak drops
materially it was estimator bias; if it survives it is real motion.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from evals.corpus import load_corpus, compile_case

def kin(times, quats):
    """quats: (F, B, 4). Returns max velocity, acceleration, jerk."""
    v, vt = [], []
    for i in range(len(times) - 1):
        dt = max(float(times[i+1] - times[i]), 1e-8)
        m = 0.0
        for b in range(quats.shape[1]):
            a = Rotation.from_quat(quats[i, b]); c = Rotation.from_quat(quats[i+1, b])
            m = max(m, float((a.inv() * c).magnitude()) / dt)
        v.append(m); vt.append((times[i+1] + times[i]) / 2)
    acc, at = [], []
    for i in range(len(v) - 1):
        dt = max(vt[i+1] - vt[i], 1e-8); acc.append(abs(v[i+1] - v[i]) / dt)
        at.append((vt[i+1] + vt[i]) / 2)
    jk = [abs(acc[i+1] - acc[i]) / max(at[i+1] - at[i], 1e-8) for i in range(len(acc) - 1)]
    return max(v, default=0.0), max(acc, default=0.0), max(jk, default=0.0)

def main() -> None:
    print(f"{'case':32s} {'jerk(real dt)':>13s} {'jerk(uniform)':>13s} {'change':>8s} {'nonunif':>8s}")
    rows = []
    for case in load_corpus():
        cid = case.entry.id
        if case.entry.intent not in ("gesture", "strike", "composite"): continue
        hand = "left" if "left" in cid else "right"
        clip = compile_case(case)
        frames = clip.frames
        bones = [f"{hand}UpperArm", f"{hand}LowerArm", f"{hand}Hand"]
        if not all(b in frames[0].bones for b in bones) or len(frames) < 8: continue
        times = np.asarray([f.time_s for f in frames], float)
        quats = np.asarray([[f.bones[b].rotation.as_list() for b in bones] for f in frames], float)
        _, _, j_real = kin(times, quats)
        # resample onto uniform spacing over the same span, same frame count
        uni = np.linspace(times[0], times[-1], len(times))
        res = np.empty_like(quats)
        for b in range(quats.shape[1]):
            res[:, b] = Slerp(times, Rotation.from_quat(quats[:, b]))(uni).as_quat()
        _, _, j_uni = kin(uni, res)
        dts = np.diff(times)
        nonunif = float(dts.max() / np.median(dts))
        pct = (j_uni - j_real) / j_real * 100 if j_real else 0.0
        rows.append((cid, j_real, j_uni, pct, nonunif))
        print(f"{cid:32s} {j_real:13.1f} {j_uni:13.1f} {pct:+7.1f}% {nonunif:8.2f}")
    if rows:
        import statistics
        print(f"\nn = {len(rows)}   median change {statistics.median(r[3] for r in rows):+.1f}%   "
              f"mean {statistics.mean(r[3] for r in rows):+.1f}%")


if __name__ == "__main__":
    main()
