"""Per-image rotation error of an estimated pose file against hidden/true_poses.json.

The estimate is defined up to a global yaw (and the scorer aligns it anyway), so the
best global rotation is removed first (Kabsch on the camera axes), then each image's
residual angle is reported.  Tilt of the estimated up axis is printed as well.

Usage: python pose_error.py --set sets/t01 --poses out/t01_poses.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("ERP_COMMON", "."))
from common import pose_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--poses", required=True)
a = ap.parse_args()
true = json.load(open(f"{a.set}/hidden/true_poses.json"))
est = json.load(open(a.poses))
Rt = {s["image"]: pose_to_R(s["yaw_deg"], s["pitch_deg"], s["roll_deg"]) for s in true["shots"]}
names = [n for n in est["R_cam2world"] if n in Rt]
Re = {n: np.array(est["R_cam2world"][n]) for n in names}
# global G with  Rt ~ G Re   (stack all camera axes)
A = np.concatenate([Re[n] for n in names], 1); B = np.concatenate([Rt[n] for n in names], 1)
U, _, Vt = np.linalg.svd(B @ A.T); G = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
errs = {}
for n in names:
    D = (G @ Re[n]).T @ Rt[n]
    errs[n] = float(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))))
tilt = float(np.degrees(np.arccos(np.clip(G[1, 1], -1, 1))))
print(f"placed {len(names)}/{len(Rt)}  hfov est {est['hfov_deg']:.2f} / true {true['hfov_deg']:.2f}")
print(f"rotation error deg: median {np.median(list(errs.values())):.3f}  max {max(errs.values()):.3f}")
print(f"tilt of estimated frame: {tilt:.2f} deg (score 0 if >= 5)")
missing = sorted(set(Rt) - set(names))
if missing:
    print("missing:", " ".join(missing))
