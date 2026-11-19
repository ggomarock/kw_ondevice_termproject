"""Jetson ERP challenge - rotation-only baseline (24 pinhole photos -> 2048x1024 ERP).

Pipeline
  1. features + matching on every pair      (--matcher sift | lightglue)
  2. RANSAC homography per pair -> inlier correspondences
  3. shared focal: 1-D search over HFOV, Kabsch rotation per pair, pick min residual
  4. rotation init: maximum spanning tree on inlier counts
  5. bundle adjustment: 24 rotations + 1 focal (scipy least_squares, robust loss)
  6. levelling: up = normal of the horizon ring, forward = one horizon camera
  7. photometric: vignetting model + per-image/per-channel gain from overlaps
  8. compositing: centre-weighted blend (weights ^ --sharp)

Conventions follow common.py of the challenge (x right, y up, z forward;
camera pixel v grows downward).  Images that do not connect to the main
component are left out (reported as "placed n/24").

Usage
  python erp_baseline.py --images sets/t01/public/images --out out/t01.png [--matcher lightglue]
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--matcher", default="sift", choices=["sift", "lightglue"])
ap.add_argument("--min_inliers", type=int, default=25)
ap.add_argument("--max_pts", type=int, default=150, help="correspondences per pair used in BA")
ap.add_argument("--sharp", type=float, default=4.0, help="blend weight exponent (higher = less averaging)")
ap.add_argument("--vignette", type=float, default=0.18, help="assumed vignetting coefficient (make_set default)")
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--poses_out", default=None, help="write estimated camera-to-world rotations (json)")
a = ap.parse_args()

T0 = time.perf_counter()
def log(msg):
    print(f"[{time.perf_counter() - T0:6.1f}s] {msg}", flush=True)

paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
imgs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
N = len(imgs)
h, w = imgs[0].shape[:2]
cx, cy = (w - 1) / 2, (h - 1) / 2
log(f"{N} images {w}x{h}")

# ------------------------------------------------------------------ 1. features + matching
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
raw = {}
if a.matcher == "sift":
    sift = cv2.SIFT_create(nfeatures=4000)
    feats = []
    for im in imgs:
        k, d = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        if d is None:
            feats.append((np.zeros((0, 2), np.float32), None)); continue
        d = d / (d.sum(1, keepdims=True) + 1e-7)          # RootSIFT
        d = np.sqrt(d).astype(np.float32)
        feats.append((np.float32([kp.pt for kp in k]), d))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for i, j in pairs:
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 2 or len(dj) < 2:
            continue
        m = [x for x, y in bf.knnMatch(di, dj, k=2) if x.distance < 0.8 * y.distance]
        if len(m) >= 8:
            raw[(i, j)] = (pi[[x.queryIdx for x in m]], pj[[x.trainIdx for x in m]])
else:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    ext = SuperPoint(max_num_keypoints=2048).eval().to(dev)
    lg = LightGlue(features="superpoint").eval().to(dev)
    feats = []
    for im in imgs:
        t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(dev)
        feats.append(ext({"image": t}))
    log(f"superpoint done ({dev})")
    for i, j in pairs:
        out = rbd(lg({"image0": feats[i], "image1": feats[j]}))
        m = out["matches"].cpu().numpy()
        if len(m) >= 8:
            k0 = feats[i]["keypoints"][0].cpu().numpy(); k1 = feats[j]["keypoints"][0].cpu().numpy()
            raw[(i, j)] = (k0[m[:, 0]].astype(np.float32), k1[m[:, 1]].astype(np.float32))
log(f"matching done: {len(raw)} candidate pairs")

# ------------------------------------------------------------------ 2. RANSAC homography
edges = {}
for (i, j), (p, q) in raw.items():
    H, inl = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.9999)
    if H is None:
        continue
    inl = inl.ravel().astype(bool)
    if inl.sum() >= a.min_inliers:
        edges[(i, j)] = (p[inl], q[inl])
log(f"verified pairs: {len(edges)}")


def bearings(pts, f):
    d = np.stack([(pts[:, 0] - cx) / f, -(pts[:, 1] - cy) / f, np.ones(len(pts))], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def kabsch(A, B):
    """R minimising |R A - B| for row-vector sets (B ~ A @ R.T)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ------------------------------------------------------------------ 3. focal search
best = None
for hfov in np.arange(68.0, 82.01, 0.25):
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    res = []
    for (i, j), (p, q) in edges.items():
        bi, bj = bearings(p, f), bearings(q, f)
        R = kabsch(bi, bj)                      # bj ~ R bi   (R = R_j^T R_i)
        res.append(np.median(np.linalg.norm(bi @ R.T - bj, axis=1)))
    s = float(np.median(res)) if res else 1e9
    if best is None or s < best[0]:
        best = (s, hfov, f)
_, hfov0, f0 = best
log(f"focal search: hfov {hfov0:.2f} deg (f {f0:.1f})")

# drop pairs that do not fit a pure rotation at the chosen focal
for e in list(edges):
    p, q = edges[e]
    bi, bj = bearings(p, f0), bearings(q, f0)
    R = kabsch(bi, bj)
    err = np.linalg.norm(bi @ R.T - bj, axis=1) * f0            # ~ pixels
    keep = err < 4.0
    if keep.sum() < a.min_inliers:
        del edges[e]
    else:
        edges[e] = (p[keep], q[keep])
log(f"rotation-consistent pairs: {len(edges)}")

# ------------------------------------------------------------------ 4. spanning-tree init
deg = np.zeros(N)
for (i, j), (p, _) in edges.items():
    deg[i] += len(p); deg[j] += len(p)
root = int(np.argmax(deg))
Rw = {root: np.eye(3)}                            # camera -> world
rel = {}
for (i, j), (p, q) in edges.items():
    rel[(i, j)] = kabsch(bearings(p, f0), bearings(q, f0))   # R_j^T R_i
order = sorted(edges, key=lambda e: -len(edges[e][0]))
changed = True
while changed:
    changed = False
    for (i, j) in order:
        if (i in Rw) == (j in Rw):
            continue
        Rij = rel[(i, j)]
        if i in Rw:
            Rw[j] = Rw[i] @ Rij.T                # R_j = R_i R_ij^T
        else:
            Rw[i] = Rw[j] @ Rij                  # R_i = R_j R_ij
        changed = True
        break
placed = sorted(Rw)
log(f"placed {len(placed)}/{N} images in the main component")

# ------------------------------------------------------------------ 5. bundle adjustment
idx = {c: k for k, c in enumerate(placed)}
rng = np.random.default_rng(0)
obs = []
for (i, j), (p, q) in edges.items():
    if i in idx and j in idx:
        sel = rng.permutation(len(p))[: a.max_pts]
        obs.append((idx[i], idx[j], p[sel], q[sel]))
n_cam = len(placed)


def unpack(x):
    rv = x[:-1].reshape(n_cam, 3)
    return Rotation.from_rotvec(rv).as_matrix(), x[-1]


def residuals(x):
    Rs, f = unpack(x)
    out = []
    for ci, cj, p, q in obs:
        di = bearings(p, f) @ Rs[ci].T
        dj = bearings(q, f) @ Rs[cj].T
        out.append(((di - dj) * f0).ravel())       # ~ pixel units (fixed scale: no focal bias)
    return np.concatenate(out)


x0 = np.concatenate([Rotation.from_matrix(np.stack([Rw[c] for c in placed])).as_rotvec().ravel(), [f0]])
n_res = sum(3 * len(o[2]) for o in obs)
S = lil_matrix((n_res, len(x0)), dtype=int)
r0 = 0
for ci, cj, p, _ in obs:
    n = 3 * len(p)
    S[r0:r0 + n, 3 * ci:3 * ci + 3] = 1
    S[r0:r0 + n, 3 * cj:3 * cj + 3] = 1
    S[r0:r0 + n, -1] = 1
    r0 += n
sol = least_squares(residuals, x0, jac_sparsity=S, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=60)
Rs, f = unpack(sol.x)
rms = np.sqrt(np.median(residuals(sol.x).reshape(-1, 3) ** 2 @ np.ones(3)))
log(f"BA: f {f:.1f} (hfov {np.degrees(2 * np.arctan(w / 2 / f)):.2f}), median residual {rms:.2f} px")

# ------------------------------------------------------------------ 6. levelling
ys = Rs[:, :, 1]                      # image-up of every camera in world
zs = Rs[:, :, 2]                      # optical axes
up = ys.sum(0); up /= np.linalg.norm(up)
nr = 0
for _ in range(3):                    # refine with the horizon ring (axes perpendicular to up)
    ring = zs[np.abs(zs @ up) < 0.35]
    nr = len(ring)
    if len(ring) < 4:
        break
    n = np.linalg.svd(ring - 0 * ring.mean(0))[2][-1]
    up = n if n @ up > 0 else -n
fwd = zs[np.argmin(np.abs(zs @ up))]
fwd = fwd - (fwd @ up) * up; fwd /= np.linalg.norm(fwd)
right = np.cross(up, fwd)
A = np.stack([right, up, fwd])        # old world -> new world
Rs = np.einsum("ab,nbc->nac", A, Rs)
log(f"levelling: horizon ring {nr} cams")
if a.poses_out:
    import json
    json.dump({"hfov_deg": float(np.degrees(2 * np.arctan(w / 2 / f))),
               "R_cam2world": {os.path.basename(paths[c]): Rs[k].tolist() for k, c in enumerate(placed)}},
              open(a.poses_out, "w"), indent=1)

# ------------------------------------------------------------------ 7. photometric + 8. compositing
W, H = a.W, a.W // 2
u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
lon, lat = np.meshgrid((u - 0.5) * 2 * np.pi, (0.5 - v) * np.pi)
D = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1).astype(np.float32)

xs, ys_ = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h) * h / w)
devig = 1.0 / (1.0 - a.vignette * (xs ** 2 + ys_ ** 2))


def project(Ri, scale=1):
    dc = D[::scale, ::scale] @ Ri             # world -> camera (R^T d, row form)
    z = dc[..., 2]
    ok = z > 1e-3
    zz = np.where(ok, z, 1)
    mx = (cx + f * dc[..., 0] / zz).astype(np.float32)
    my = (cy - f * dc[..., 1] / zz).astype(np.float32)
    ok &= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    wt = np.clip(1 - np.abs(mx - cx) / (w / 2), 0, 1) * np.clip(1 - np.abs(my - cy) / (h / 2), 0, 1)
    return mx, my, ok, wt * ok


lin = [im.astype(np.float32) * devig[..., None] for im in imgs]

# per-image, per-channel gain from overlaps at low resolution (log domain least squares)
sc = 8
lows = []
for k, c in enumerate(placed):
    mx, my, ok, _ = project(Rs[k], sc)
    s = cv2.remap(lin[c], mx, my, cv2.INTER_LINEAR)
    lows.append((s, ok))
rows, rhs = [], []
for a_ in range(n_cam):
    for b_ in range(a_ + 1, n_cam):
        m = lows[a_][1] & lows[b_][1]
        if m.sum() < 200:
            continue
        sa, sb = lows[a_][0][m], lows[b_][0][m]
        good = (sa.min(1) > 8) & (sb.min(1) > 8) & (sa.max(1) < 245) & (sb.max(1) < 245)
        if good.sum() < 100:
            continue
        rows.append((a_, b_, good.sum()))
        rhs.append(np.log(np.median(sb[good], 0)) - np.log(np.median(sa[good], 0)))
G = np.zeros((n_cam, 3))
if rows:
    M = np.zeros((len(rows) + 1, n_cam))
    for r, (a_, b_, _) in enumerate(rows):
        M[r, a_], M[r, b_] = 1, -1             # g_a - g_b = log(b) - log(a)
    M[-1, :] = 1                               # mean gain = 1
    G = np.linalg.lstsq(M, np.vstack([np.array(rhs), np.zeros(3)]), rcond=None)[0]
gains = np.exp(G)

acc = np.zeros((H, W, 3), np.float64)
wsum = np.zeros((H, W), np.float64)
for k, c in enumerate(placed):
    mx, my, ok, wt = project(Rs[k])
    s = cv2.remap(lin[c], mx, my, cv2.INTER_CUBIC) * gains[k][None, None, :]
    wt = wt ** a.sharp
    acc += s * wt[..., None]
    wsum += wt
erp = acc / np.maximum(wsum, 1e-8)[..., None]
hole = wsum < 1e-8
if hole.any():   # fill holes from the nearest valid pixel (cheap inpaint)
    erp8 = np.clip(erp, 0, 255).astype(np.uint8)
    erp = cv2.inpaint(erp8, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float64)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
cv2.imwrite(a.out, np.clip(erp + 0.5, 0, 255).astype(np.uint8))
log(f"wrote {a.out}  ({W}x{H}, holes {hole.mean() * 100:.2f}%)")
