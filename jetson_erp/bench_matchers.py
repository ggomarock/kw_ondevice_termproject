"""Benchmark feature matchers on the Jetson ERP training sets using ground-truth poses.

Because every shot is a pure rotation about one centre, the true pixel mapping between two
shots is the homography  H_ij = K S R_j^T R_i S K^-1  (S = diag(1,-1,1) flips image y to camera y).
So each matcher can be judged without building the stitching pipeline.

Per pair (only pairs whose true views overlap by >= --min_overlap):
  n_match     matches returned by the matcher
  n_correct   matches whose reprojection error under the true H is < --px px
  precision   n_correct / n_match
  n_inlier    RANSAC homography inliers (what a real pipeline would see; no GT used)
  h_err       median error (px) of the RANSAC H vs the true H on the overlap grid
  ok          n_inlier >= --min_inl and h_err < --px   (pair usable for stitching)
Per set: connected components of the graph of ok pairs -> are all 24 shots placeable?

Usage:
  python bench_matchers.py --sets "<...>/Jetson ERP/sets" --out results --matchers sift lightglue loftr roma
  python bench_matchers.py ... --only t01            # quick check
GT (hidden/true_poses.json) is used for evaluation only.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sets", required=True)
ap.add_argument("--out", default="results")
ap.add_argument("--matchers", nargs="+", default=["sift", "lightglue", "loftr", "roma"])
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--min_overlap", type=float, default=0.10)
ap.add_argument("--px", type=float, default=3.0)
ap.add_argument("--min_inl", type=int, default=30)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, H = 864, 648
S = np.diag([1.0, -1.0, 1.0])


# ---------------------------------------------------------------- geometry (same conventions as common.py)
def rot(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def K_of(hfov):
    f = (W / 2) / np.tan(np.radians(hfov) / 2)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])


def gt_H(K, Ri, Rj):
    return K @ S @ Rj.T @ Ri @ S @ np.linalg.inv(K)


GRID = np.stack(np.meshgrid(np.linspace(0, W - 1, 40), np.linspace(0, H - 1, 30)), -1).reshape(-1, 2)


def project(Hm, pts):
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def overlap_pts(K, Ri, Rj):
    """Grid points of image i that are visible in image j (in front of camera j and inside it)."""
    rays = (np.c_[GRID, np.ones(len(GRID))] @ np.linalg.inv(K).T) @ S
    dj = rays @ (Rj.T @ Ri).T
    q, _ = project(gt_H(K, Ri, Rj), GRID)
    vis = (dj[:, 2] > 0) & (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    return GRID[vis]


def group(idx):  # capture_plan order: 0-7 horizon, 8-14 up, 15-21 down, 22 zenith, 23 nadir
    return "H" if idx < 8 else "U" if idx < 15 else "D" if idx < 22 else "Z" if idx == 22 else "N"


# ---------------------------------------------------------------- matchers: each returns (k0, k1) Nx2 pixel arrays
class SIFT:
    def __init__(self):
        self.det = cv2.SIFT_create(nfeatures=4000)
        self.cache = {}

    def feats(self, path, img):
        if path not in self.cache:
            k, d = self.det.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            if d is None:
                k, d = [], np.zeros((0, 128), np.float32)
            d = np.sqrt(d / (np.abs(d).sum(1, keepdims=True) + 1e-7))  # RootSIFT
            self.cache[path] = (np.float32([p.pt for p in k]).reshape(-1, 2), d.astype(np.float32))
        return self.cache[path]

    def __call__(self, p0, p1, i0, i1):
        k0, d0 = self.feats(p0, i0); k1, d1 = self.feats(p1, i1)
        if len(d0) < 2 or len(d1) < 2:
            return np.zeros((0, 2)), np.zeros((0, 2))
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
        return k0[[g.queryIdx for g in good]], k1[[g.trainIdx for g in good]]


class LightGlueM:
    def __init__(self):
        from lightglue import LightGlue, SuperPoint
        self.ext = SuperPoint(max_num_keypoints=2048).eval().to(dev)
        self.lg = LightGlue(features="superpoint").eval().to(dev)
        self.cache = {}

    @torch.no_grad()
    def __call__(self, p0, p1, i0, i1):
        from lightglue.utils import rbd
        f = []
        for p, im in ((p0, i0), (p1, i1)):
            if p not in self.cache:
                t = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(dev)
                self.cache[p] = self.ext.extract(t)
            f.append(self.cache[p])
        out = rbd(self.lg({"image0": f[0], "image1": f[1]}))
        m = out["matches"].cpu().numpy()
        k0 = rbd(f[0])["keypoints"].cpu().numpy(); k1 = rbd(f[1])["keypoints"].cpu().numpy()
        return k0[m[:, 0]], k1[m[:, 1]]


class LoFTRM:
    def __init__(self):
        import kornia.feature as KF
        # "indoor_new" returns almost no matches in kornia 0.9 (13 on a 39%-overlap pair) -> use "indoor".
        # Run at the training resolution 640x480 (2.7x more matches than 864x648) and scale back.
        self.m = KF.LoFTR(pretrained="indoor").eval().to(dev)
        self.size = (640, 480)

    @torch.no_grad()
    def __call__(self, p0, p1, i0, i1):
        g = lambda im: torch.from_numpy(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), self.size,
                                                   interpolation=cv2.INTER_AREA)).float().div(255)[None, None].to(dev)
        out = self.m({"image0": g(i0), "image1": g(i1)})
        keep = out["confidence"] > 0.2
        sc = np.array([W / self.size[0], H / self.size[1]])
        return out["keypoints0"][keep].cpu().numpy() * sc, out["keypoints1"][keep].cpu().numpy() * sc


class RoMaM:
    def __init__(self):
        from romatch import roma_indoor
        self.m = roma_indoor(device=dev)

    @torch.no_grad()
    def __call__(self, p0, p1, i0, i1):
        warp, cert = self.m.match(p0, p1, device=dev)
        matches, c = self.m.sample(warp, cert, num=5000)
        k0, k1 = self.m.to_pixel_coordinates(matches, H, W, H, W)
        keep = (c > 0.05).cpu().numpy()
        return k0.cpu().numpy()[keep], k1.cpu().numpy()[keep]


MATCHERS = {"sift": SIFT, "lightglue": LightGlueM, "loftr": LoFTRM, "roma": RoMaM}


# ---------------------------------------------------------------- evaluation
def eval_pair(k0, k1, Hgt, ov):
    r = {"n_match": len(k0), "n_correct": 0, "precision": 0.0, "n_inlier": 0, "h_err": float("inf")}
    if len(k0) == 0:
        return r
    q, z = project(Hgt, k0)
    err = np.linalg.norm(q - k1, axis=1)
    r["n_correct"] = int(((err < a.px) & (z > 0)).sum())
    r["precision"] = r["n_correct"] / len(k0)
    if len(k0) >= 4:
        Hest, inl = cv2.findHomography(k0.astype(np.float64), k1.astype(np.float64), cv2.USAC_MAGSAC, a.px)
        if Hest is not None:
            r["n_inlier"] = int(inl.sum())
            qe, _ = project(Hest, ov); qg, _ = project(Hgt, ov)
            r["h_err"] = float(np.median(np.linalg.norm(qe - qg, axis=1)))
    return r


def components(n, edges):
    par = list(range(n))
    find = lambda x: x if par[x] == x else find(par[x])
    for i, j in edges:
        par[find(i)] = find(j)
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return sorted(comp.values(), key=len, reverse=True)


os.makedirs(a.out, exist_ok=True)
ids = sorted(d for d in os.listdir(a.sets) if os.path.isdir(os.path.join(a.sets, d, "hidden")))
if a.only:
    ids = [i for i in ids if i in a.only]

# pairs + images per set (shared by all matchers)
data = {}
for sid in ids:
    tp = json.load(open(os.path.join(a.sets, sid, "hidden", "true_poses.json")))
    K = K_of(tp["hfov_deg"])
    shots = tp["shots"]
    Rs = [rot(s["yaw_deg"], s["pitch_deg"], s["roll_deg"]) for s in shots]
    paths = [os.path.join(a.sets, sid, "public", "images", s["image"]) for s in shots]
    imgs = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]  # imread fails on non-ASCII paths (Windows)
    pairs = []
    for i in range(24):
        for j in range(i + 1, 24):
            ov = overlap_pts(K, Rs[i], Rs[j])
            if len(ov) / len(GRID) >= a.min_overlap:
                pairs.append((i, j, len(ov) / len(GRID), gt_H(K, Rs[i], Rs[j]), ov))
    data[sid] = (paths, imgs, pairs)
    print(f"{sid}: hfov {tp['hfov_deg']:.2f}, {len(pairs)} overlapping pairs", flush=True)

rows, summary = [], []
for name in a.matchers:
    print(f"\n=== {name}", flush=True)
    try:
        M = MATCHERS[name]()
    except Exception as e:
        print(f"  cannot load {name}: {e!r}", flush=True)
        continue
    for sid in ids:
        paths, imgs, pairs = data[sid]
        if dev.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ok_edges, t_tot = [], 0.0
        for i, j, ovr, Hgt, ov in pairs:
            t0 = time.perf_counter()
            k0, k1 = M(paths[i], paths[j], imgs[i], imgs[j])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            t_tot += dt
            r = eval_pair(np.asarray(k0, np.float64), np.asarray(k1, np.float64), Hgt, ov)
            ok = r["n_inlier"] >= a.min_inl and r["h_err"] < a.px
            if ok:
                ok_edges.append((i, j))
            rows.append({"matcher": name, "set": sid, "i": i, "j": j, "pair_group": "-".join(sorted(group(i) + group(j))),
                         "overlap": round(ovr, 3), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                         "ok": int(ok), "sec": round(dt, 4)})
        comp = components(24, ok_edges)
        iso = [f"{k}{group(k)}" for c in comp[1:] for k in c]
        mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
        s = {"matcher": name, "set": sid, "pairs": len(pairs), "ok_pairs": len(ok_edges),
             "largest_comp": len(comp[0]), "n_comp": len(comp), "not_in_largest": " ".join(iso),
             "sec_total": round(t_tot, 2), "sec_per_pair": round(t_tot / max(len(pairs), 1), 3), "peak_mem_MB": round(mem)}
        summary.append(s)
        print(f"  {sid}: ok {len(ok_edges):3d}/{len(pairs)}  largest {len(comp[0]):2d}/24  "
              f"{t_tot:6.1f}s  {mem:6.0f}MB  {('outside: ' + ' '.join(iso)) if iso else ''}", flush=True)
    del M
    if dev.type == "cuda":
        torch.cuda.empty_cache()

with open(os.path.join(a.out, "pairs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

# group table: pair success rate and median correct matches per pair group
print("\n=== pair success rate by group (ok / pairs, median n_correct)")
grp = sorted({r["pair_group"] for r in rows})
print("matcher    " + "".join(f"{g:>16s}" for g in grp))
for name in a.matchers:
    line = f"{name:10s} "
    for g in grp:
        rr = [r for r in rows if r["matcher"] == name and r["pair_group"] == g]
        if rr:
            line += f"{sum(r['ok'] for r in rr):5d}/{len(rr):<4d}({int(np.median([r['n_correct'] for r in rr])):4d})"
        else:
            line += " " * 16
    print(line)
print(f"\nwrote {a.out}/pairs.csv, {a.out}/summary.csv")
