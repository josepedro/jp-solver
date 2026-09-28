#!/usr/bin/env python3
# Quick GL-free preview of the 3D TGV AMR run: a max-intensity projection of the Q-criterion per frame
# with the dynamic refinement blocks overlaid as rectangles. Not the final 3D render (that is ParaView);
# this is a viewable mp4 to check the run and see the refinement track the vortices. Needs numpy,
# matplotlib and ffmpeg only.
import glob, os, re, struct, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--prefix", default="tgv")
ap.add_argument("--out", default=None)
ap.add_argument("--axis", type=int, default=0, help="projection axis (0=z,1=y,2=x)")
args = ap.parse_args()
outdir = args.out or os.path.join(args.dir, "preview")
os.makedirs(outdir, exist_ok=True)

def read_vti_Q(path):
    with open(path, "rb") as f:
        raw = f.read()
    m = re.search(rb'WholeExtent="0 (\d+)', raw); N = int(m.group(1)) + 1
    a = raw.find(b'<AppendedData'); us = raw.find(b'_', a) + 1
    (nb1,) = struct.unpack_from("<Q", raw, us)
    q = np.frombuffer(raw, dtype="<f4", count=nb1 // 4, offset=us + 8)
    return q.reshape(N, N, N), N  # [k,j,i]

def read_boxes(path):
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
    mp = re.search(r'<Points>.*?format="ascii">(.*?)</DataArray>', txt, re.S)
    if not mp: return []
    nums = np.array(mp.group(1).split(), dtype=float)
    if nums.size < 24: return []
    pts = nums.reshape(-1, 3)
    boxes = []
    for b in range(pts.shape[0] // 8):
        c = pts[b*8:(b+1)*8]
        boxes.append((c[:,0].min(), c[:,0].max(), c[:,1].min(), c[:,1].max(), c[:,2].min(), c[:,2].max()))
    return boxes

vtis = sorted(glob.glob(os.path.join(args.dir, args.prefix + "_*.vti")))
vtps = sorted(glob.glob(os.path.join(args.dir, args.prefix + "_*.vtp")))
if not vtis: sys.exit("no vti frames")
print("frames:", len(vtis))

# fix the colour scale from a late frame so growth over time is honest
qL, N = read_vti_Q(vtis[min(len(vtis)-1, int(len(vtis)*0.8))])
projL = np.clip(qL, 0, None).max(axis=args.axis)
vmax = np.percentile(projL[projL > 0], 99.0) if np.any(projL > 0) else 1.0
ax_names = {0: ("x", "y", 2, 1), 1: ("x", "z", 2, 0), 2: ("y", "z", 1, 0)}  # axis -> (xlabel,ylabel,box_x,box_y)
xl, yl, bxi, byi = ax_names[args.axis]

for fi, vp in enumerate(vtis):
    q, N = read_vti_Q(vp)
    proj = np.clip(q, 0, None).max(axis=args.axis)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(proj, origin="lower", extent=[0, N, 0, N], cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear")
    bxs = read_boxes(vtps[fi]) if fi < len(vtps) else []
    lo = {0: (2, 3), 1: (0, 1), 2: (0, 1)}  # unused; explicit below
    for bb in bxs:
        xlo, xhi = bb[bxi*2], bb[bxi*2+1]; ylo, yhi = bb[byi*2], bb[byi*2+1]
        ax.add_patch(Rectangle((xlo, ylo), xhi-xlo, yhi-ylo, fill=False, edgecolor=(1.0,0.55,0.1), lw=0.6, alpha=0.55))
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TGV3D GLO-AMR  Q max-projection + refinement blocks  frame %03d/%d" % (fi, len(vtis)), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "f_%04d.png" % fi)); plt.close(fig)
    if fi % 20 == 0: print("rendered", fi)
print("PNG sequence in", outdir)
