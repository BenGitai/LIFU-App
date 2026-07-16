"""
lifu_pipeline.py  --  CT/atlas-based transcranial LIFU planning + k-Wave sim.

Runs on Penn's Brains cluster. Reproduces the STEP-notebook outputs but starting
from the subject's real CT/MRI + subcortical atlas, with a virtual craniotomy.

Pipeline (each step saves arrays + a figure into --out):
  1  load + co-register all volumes (CT, MRI, skull, brain, subcortical atlas)
  2  find the STN target (subcortical label 91), auto-pick the clearer side
  3  skull surface + beam entry point + local normal
  4  register the device (outer_base + inner_adapter STL) flush on the skull
  5  cut the craniotomy hole in the skull under the device opening
  6  HU-based acoustic maps (c, rho) for skull/brain/device/water
  7  aim the 19 mm transducer through the window at the target
  8  phase-plate focusing delays to the target
  9  crop + resample the sim domain to isotropic dx_sim, save the bundle
 10  run k-Wave on GPU (record p_max on planes through the target)
 11  visualise the pressure field + anatomy

Dependencies (Brains): numpy, scipy, nibabel, trimesh, matplotlib, and the CuPy
k-Wave build (+ cupy) used on Della.  cadquery is NOT needed -- ship the .stl
files made by convert_step_to_stl.py.

    python lifu_pipeline.py --data . --out ./pipeline_out --dx-sim 0.3 --run-sim
    python lifu_pipeline.py --data . --out ./pipeline_out            # geometry only
"""
import os, sys, json, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import trimesh
from scipy import ndimage as ndi

# ── acoustic constants ────────────────────────────────────────────────────────
C_WATER, RHO_WATER = 1500.0, 1000.0
C_BRAIN, RHO_BRAIN = 1540.0, 1040.0
C_BONE,  RHO_BONE  = 3100.0, 2200.0     # dense cortical bone (porosity interpolates to water)
C_SIL,   RHO_SIL   = 1000.0, 1030.0     # inner_adapter silicone
C_TI,    RHO_TI    = 6070.0, 4540.0     # outer_base titanium
C_LENS             = 2500.0             # 3D-print resin sound speed (for the focusing lens)
LENS_BASE_MM       = 2.0                # flat backing-plate thickness of the printed lens
BACK_LEN_MM        = 12.0               # absorbing backing cylinder length behind the transducer

STN_LABEL   = 91          # subthalamic_nucleus (STh) in subcortical.nii
TX_DIAM_MM  = 19.0
FREQ_HZ     = 1e6
P0_PA       = 1e5
N_CYCLES    = 30          # near-CW: enough cycles for the Fresnel-wrapped lens to focus coherently
PML         = 12

FILES = dict(ct="BeanCTReg2MRI.nii", mri="BeanMRI.nii", skull="Bean_Skull.nii",
             brain="Bean_Brain.nii", subcort="subcortical.nii", cort="cortical.nii")
STL = dict(adapter="inner_adapter.stl", base="outer_base_craniotomy.stl")


# ── small helpers ─────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)


def load_nii(path):
    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(np.float32), img.affine


def resolve_files(data):
    """Auto-detect which file is which inside a folder (any filenames, .nii/.nii.gz).
    Masks/atlases by keyword; CT vs MRI by content (CT has strongly negative HU)."""
    import glob
    low = lambda p: os.path.basename(p).lower()
    niis = sorted(glob.glob(os.path.join(data, "*.nii")) + glob.glob(os.path.join(data, "*.nii.gz")))
    pick = lambda *ks, ex=(): next((p for p in niis if all(k in low(p) for k in ks)
                                    and not any(e in low(p) for e in ex)), None)
    subcort = pick("subcort"); cort = pick("cort", ex=("subcort",))
    skull = pick("skull"); brain = pick("brain")
    used = {subcort, cort, skull, brain}
    rest = [p for p in niis if p not in used]
    ct = mri = None
    if rest:
        mins = {}
        for p in rest:
            try: mins[p] = float(np.asarray(nib.load(p).dataobj).min())
            except Exception: mins[p] = 0.0
        ct = min(rest, key=lambda p: mins[p])                 # most-negative HU = CT
        others = [p for p in rest if p != ct]
        mri = min(others, key=lambda p: "mri" not in low(p)) if others else ct
    stls = glob.glob(os.path.join(data, "*.stl")) + glob.glob(os.path.join(data, "*.STL"))
    slow = lambda p: os.path.basename(p).lower()
    spick = lambda *ks: next((p for p in stls if all(k in slow(p) for k in ks)), None)
    return {"ct": ct, "mri": mri, "skull": skull, "brain": brain, "subcort": subcort,
            "cort": cort, "adapter": spick("adapter"), "base": spick("base"),
            "skull_craniotomy": spick("skull", "cranio")}


def w2v(xyz, aff):
    inv = np.linalg.inv(aff)
    xyz = np.atleast_2d(np.asarray(xyz, float))
    return (xyz @ inv[:3, :3].T) + inv[:3, 3]


def v2w(ijk, aff):
    ijk = np.atleast_2d(np.asarray(ijk, float))
    return (ijk @ aff[:3, :3].T) + aff[:3, 3]


def resample_like(src, src_aff, ref_shape, ref_aff, order=0):
    """Resample src volume onto the ref grid (nearest for masks/labels)."""
    if src.shape == ref_shape and np.allclose(src_aff, ref_aff, atol=1e-3):
        return src
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in ref_shape], indexing="ij")
    ref_ijk = np.stack([ii, jj, kk, np.ones_like(ii)], -1).reshape(-1, 4)
    world = ref_ijk @ ref_aff.T
    src_ijk = (world @ np.linalg.inv(src_aff).T)[:, :3].T
    out = ndi.map_coordinates(src, src_ijk, order=order, mode="constant", cval=0.0)
    return out.reshape(ref_shape)


def voxelize_mesh_on_grid(mesh, aff, shape):
    """Voxelize a world-space mesh (mm) onto the reference grid -> bool mask."""
    inv = np.linalg.inv(aff)
    vi = (mesh.vertices @ inv[:3, :3].T) + inv[:3, 3]          # -> voxel-index coords
    mi = trimesh.Trimesh(vertices=vi, faces=mesh.faces, process=False)
    vox = mi.voxelized(pitch=1.0).fill()
    pts = np.round(vox.points).astype(int)
    ok = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
    pts = pts[ok]
    m = np.zeros(shape, bool)
    m[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    return m


def ortho(vol, center, path, title="", cmap="gray", overlays=None, pts=None, proj=None):
    """3 orthogonal slices through `center` (voxel idx). overlays=(mask,color) drawn as
    the slice contour; proj=(mask,color) projected (any voxel) along the through-axis,
    dashed -- use proj for thin off-plane parts (device) so the full footprint shows."""
    c = np.round(center).astype(int)
    c = np.clip(c, [0, 0, 0], np.array(vol.shape) - 1)
    planes = [(2, "XY", 0, 1), (1, "XZ", 0, 2), (0, "YZ", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (fx, nm, a0, a1) in zip(axes, planes):
        sl = [slice(None)] * 3; sl[fx] = c[fx]
        ax.imshow(vol[tuple(sl)].T, origin="lower", cmap=cmap, aspect="equal")
        for msk, col in (overlays or []):
            mm = msk[tuple(sl)].T
            if mm.any():
                ax.contour(mm, levels=[0.5], colors=[col], linewidths=0.9)
        for msk, col in (proj or []):
            mm = msk.any(axis=fx).T
            if mm.any():
                ax.contour(mm, levels=[0.5], colors=[col], linewidths=1.1, linestyles="--")
        if pts is not None:
            for p, col in pts:
                pc = np.round(p).astype(int)
                ax.plot(pc[a0], pc[a1], "o", color=col, ms=6, mfc="none", mew=2)
        ax.set_title(f"{nm} @ {'XYZ'[fx]}={c[fx]}")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def kabsch(A, B):
    """Rigid transform mapping A onto B (both (N,3)); returns (R, t)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def _proper(R):
    U, _, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def umeyama(A, B, with_scale=False):
    """Similarity transform mapping A->B: B ~= s*R@A + t. Returns (s, R, t)."""
    n = len(A); muA = A.mean(0); muB = B.mean(0)
    A0, B0 = A - muA, B - muB
    Sig = (B0.T @ A0) / n
    U, D, Vt = np.linalg.svd(Sig)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = U @ Sgn @ Vt
    s = float(np.trace(np.diag(D) @ Sgn) / ((A0 ** 2).sum() / n)) if with_scale else 1.0
    return s, R, muB - s * (R @ muA)


def register_surface(src, dst, n_iter=40, sub=6000, seed=0):
    """Global ICP aligning source points onto dst (rigid + one PCA global scale).
    Tries all 24 axis-aligned orientations of the PCA frames (escapes the symmetric-
    skull flips), short-ICPs each, keeps the best, then refines. A point maps as
    (s0*p) @ R.T + t.  Returns (s0, R, t, rms)."""
    import itertools
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(src) > sub: src = src[rng.choice(len(src), sub, replace=False)]
    if len(dst) > sub: dst = dst[rng.choice(len(dst), sub, replace=False)]
    cs, cd = src.mean(0), dst.mean(0)
    evs, Us = np.linalg.eigh(np.cov((src - cs).T))
    evd, Ud = np.linalg.eigh(np.cov((dst - cd).T))
    s0 = float(np.sqrt(max(evd.sum(), 1e-9) / max(evs.sum(), 1e-9)))   # global scale
    src_s = s0 * src; cs_s = s0 * cs
    tree = cKDTree(dst)
    inits = []                                     # 24 proper signed-permutation frames
    for perm in itertools.permutations(range(3)):
        Pm = np.eye(3)[:, list(perm)]
        for sg in itertools.product((1, -1), repeat=3):
            M = Pm * np.array(sg)
            if np.linalg.det(M) > 0:
                inits.append(_proper(Ud @ M @ Us.T))
    best = None
    for R0 in inits:
        R, t = R0, cd - R0 @ cs_s
        for _ in range(8):
            d, idx = tree.query(src_s @ R.T + t)
            _, R, t = umeyama(src_s, dst[idx], with_scale=False)
        d, _ = tree.query(src_s @ R.T + t)
        rms = float(np.sqrt((d ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, R, t)
    _, R, t = best
    for _ in range(n_iter):
        d, idx = tree.query(src_s @ R.T + t)
        _, R, t = umeyama(src_s, dst[idx], with_scale=False)
    d, _ = tree.query(src_s @ R.T + t)
    return s0, R, t, float(np.sqrt((d ** 2).mean()))


# ── STEP 1: load + co-register ────────────────────────────────────────────────
def step1_load(data, out, S):
    log("[1] loading volumes ...")
    F = S.get("files") or resolve_files(data)
    S["files"] = F
    missing = [k for k in ("ct", "mri", "skull", "brain", "subcort", "cort") if not F.get(k)]
    if missing:
        raise FileNotFoundError(f"could not find in {data}: {', '.join(missing)} "
                                f"(need CT, MRI, skull/brain masks, subcortical + cortical atlases)")
    log("    " + "  ".join(f"{k}={os.path.basename(F[k])}" for k in
                           ("ct", "mri", "skull", "brain", "subcort", "cort")))
    ct, aff = load_nii(F["ct"])
    S["aff"] = aff; S["shape"] = ct.shape; S["ct"] = ct
    for key in ("mri", "skull", "brain", "subcort", "cort"):
        v, a = load_nii(F[key])
        order = 0 if key in ("skull", "brain", "subcort", "cort") else 1
        S[key] = resample_like(v, a, ct.shape, aff, order=order)
    S["skull"] = S["skull"] > 0.5
    S["brain"] = S["brain"] > 0.5
    S["vmm"] = float(np.mean(np.linalg.norm(aff[:3, :3], axis=0)))   # mean voxel size (mm)
    S["skull_c"] = np.argwhere(S["skull"]).mean(0)                    # skull centroid (voxel)
    zooms = np.linalg.norm(aff[:3, :3], axis=0)     # true per-axis voxel size (column norms)
    log(f"    grid {ct.shape}  voxel {np.round(zooms,3)} mm  CT HU [{ct.min():.0f},{ct.max():.0f}]")
    ctr = np.array(ct.shape) // 2
    ortho(S["mri"], ctr, os.path.join(out, "step01_inputs.png"), "MRI + skull(cyan)/brain(yellow)",
          overlays=[(S["skull"], "cyan"), (S["brain"], "yellow")])
    np.save(os.path.join(out, "step01_skull.npy"), S["skull"])
    np.save(os.path.join(out, "step01_brain.npy"), S["brain"])
    return S


# ── STEP 2: STN target (auto side) ────────────────────────────────────────────
def step2_target(data, out, S):
    label = int(S.get("target_label", STN_LABEL))
    atlas = S.get("atlas", "subcort")                        # "subcort" or "cort"
    name = S.get("target_name") or f"label {label}"
    log(f"[2] locating target '{name}' (label {label}) in {atlas} atlas ...")
    stn = np.isclose(S[atlas], label)
    if not stn.any():
        raise ValueError(f"target label {label} not found in {atlas} atlas")
    lab, n = ndi.label(stn)
    comps = [(lab == i) for i in range(1, n + 1)]
    comps = sorted(comps, key=lambda m: -m.sum())[:2]        # two biggest = L/R STN
    brain_c = np.argwhere(S["brain"]).mean(0)
    cand = []
    for m in comps:
        t = np.argwhere(m).mean(0)
        u = t - brain_c; u = u / (np.linalg.norm(u) + 1e-9)   # outward ray
        E = _ray_exit(t, u, S["skull"])
        n_hat = _skull_normal(E, S["skull"], brain_c)
        obliq = np.degrees(np.arccos(np.clip(abs(u @ n_hat), 0, 1)))   # ray vs normal
        cand.append(dict(target=t, entry=E, normal=n_hat, obliq=obliq, mask=m))
    best = min(cand, key=lambda c: c["obliq"])
    off = np.asarray(S.get("target_offset", [0, 0, 0]), float)      # user nudge (mm, scan/world frame)
    if np.any(off):
        best["target"] = best["target"] + np.linalg.inv(S["aff"])[:3, :3] @ off
        u = best["target"] - brain_c; u = u / (np.linalg.norm(u) + 1e-9)
        best["entry"] = _ray_exit(best["target"], u, S["skull"])
        best["normal"] = _skull_normal(best["entry"], S["skull"], brain_c)
        log(f"    nudged target by {off} mm (world)")
    S.update(target=best["target"], entry=best["entry"], normal=best["normal"], stn=best["mask"])
    tw = v2w(best["target"], S["aff"])[0]
    log(f"    picked STN side: target vox {np.round(best['target'],1)} = {np.round(tw,1)} mm, "
        f"path obliquity {best['obliq']:.0f} deg")
    ortho(S["mri"], best["target"], os.path.join(out, "step02_target.png"),
          f"'{name}' structure (red) + focus point (cyan)", overlays=[(best["mask"], "red"), (S["brain"], "yellow")],
          pts=[(best["target"], "cyan")])
    json.dump({"label": label, "atlas": atlas, "name": name, "voxels": int(best["mask"].sum()),
               "target_vox": best["target"].tolist(), "target_mm": tw.tolist(),
               "obliquity_deg": float(best["obliq"])},
              open(os.path.join(out, "step02_target.json"), "w"), indent=2)
    return S


def _ray_exit(target, u, skull, step=0.5, max_t=300):
    last = target.copy()
    for t in np.arange(0, max_t, step):
        p = target + u * t
        ijk = np.round(p).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(skull.shape)):
            break
        if skull[tuple(ijk)]:
            last = p
    return last


def _skull_normal(E, skull, brain_c):
    sm = ndi.gaussian_filter(skull.astype(np.float32), 2.0)
    g = np.array(np.gradient(sm))                      # points toward more bone (inward)
    ijk = np.clip(np.round(E).astype(int), 0, np.array(skull.shape) - 1)
    n = -np.array([g[a][tuple(ijk)] for a in range(3)])   # outward
    if np.linalg.norm(n) < 1e-6:
        n = E - brain_c
    n = n / (np.linalg.norm(n) + 1e-9)
    if n @ (E - brain_c) < 0:
        n = -n
    return n


# ── STEP 3: entry-point figure ────────────────────────────────────────────────
def step3_entry(data, out, S):
    log("[3] beam entry point + skull normal ...")
    E, nrm = S["entry"], S["normal"]
    log(f"    entry vox {np.round(E,1)}  outward normal {np.round(nrm,2)}")
    ortho(S["skull"].astype(float), E, os.path.join(out, "step03_entry.png"),
          "skull with entry (green) + target (red)",
          overlays=[(S["stn"], "red")], pts=[(E, "lime"), (S["target"], "red")])
    return S


# ── STEP 4: register device flush on skull ────────────────────────────────────
def step4_register(data, out, S):
    log("[4] placing device via the STEP<->NIfTI world correspondence (direct, no ICP) ...")
    # The SolidWorks STEP parts were built in the scan's physical (world-mm) frame, so we
    # voxelize them straight through the NIfTI affine -- the affine IS the correspondence.
    F = S["files"]
    if not F.get("base") or not F.get("adapter"):
        raise FileNotFoundError(f"missing device STL in {data}: need an *adapter*.stl and an *base*.stl")
    base = trimesh.load(F["base"], process=True, force="mesh")
    adap = trimesh.load(F["adapter"], process=True, force="mesh")
    for nm, m in (("base", base), ("adapter", adap)):
        if m.vertices.size == 0:
            raise ValueError(f"{nm} STL loaded empty ({os.path.basename(F[nm])}) — re-export it")
    S["base_mask"] = voxelize_mesh_on_grid(base, S["aff"], S["shape"])
    S["adap_mask"] = voxelize_mesh_on_grid(adap, S["aff"], S["shape"])
    P = adap.vertices - adap.vertices.mean(0)
    _, evec = np.linalg.eigh(np.cov(P.T))
    S["dev_axis_w"] = evec[:, 0] / np.linalg.norm(evec[:, 0])
    np.save(os.path.join(out, "step04_base_mask.npy"), S["base_mask"])
    np.save(os.path.join(out, "step04_adap_mask.npy"), S["adap_mask"])

    # correspondence check: report world spans, and Dice of CAD skull vs CT skull
    lo = v2w([0, 0, 0], S["aff"])[0]; hi = v2w(np.array(S["shape"]) - 1, S["aff"])[0]
    log(f"    NIfTI world span  {np.round(np.minimum(lo,hi),1)} .. {np.round(np.maximum(lo,hi),1)} mm")
    log(f"    device world span {np.round(base.bounds[0],1)} .. {np.round(base.bounds[1],1)} mm")
    cad_path = F.get("skull_craniotomy")
    if cad_path and os.path.exists(cad_path):
        cad = trimesh.load(cad_path, process=True, force="mesh")
        reg = voxelize_mesh_on_grid(cad, S["aff"], S["shape"])
        dice = 2.0 * float((reg & S["skull"]).sum()) / max(int(reg.sum()) + int(S["skull"].sum()), 1)
        log(f"    CAD skull vs CT skull Dice = {dice:.2f}  (high => correspondence is correct)")
        ortho(S["skull"].astype(float), S["skull_c"], os.path.join(out, "step04_registration.png"),
              f"CAD skull (orange, proj) vs CT skull (white)  Dice={dice:.2f}", proj=[(reg, "orange")])
    ortho(S["skull"].astype(float), S["skull_c"], os.path.join(out, "step04_device.png"),
          "device footprint (proj): base(cyan) adapter(magenta) STN(red)",
          proj=[(S["base_mask"], "cyan"), (S["adap_mask"], "magenta"), (S["stn"], "red")],
          pts=[(S["target"], "red")])
    return S


def _rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _skull_surface_world(skull, aff, center_w, radius_mm=40):
    surf = skull & ~ndi.binary_erosion(skull)
    pts = np.argwhere(surf)
    w = v2w(pts, aff)
    keep = np.linalg.norm(w - center_w, axis=1) <= radius_mm
    return w[keep] if keep.any() else w


# ── STEP 5: cut the craniotomy hole ───────────────────────────────────────────
def step5_cut(data, out, S):
    # The CAD skull surface != the CT skull surface, so subtracting the whole holed
    # CAD skull just captures surface disagreement everywhere.  Instead cut the hole
    # geometrically: the craniotomy = skull inside the adapter's BORE, along the
    # device axis.  The adapter is placed correctly via the affine, so this is exact.
    log("[5] cutting craniotomy = skull inside the adapter bore ...")
    vmm = S["vmm"]
    ax = np.linalg.inv(S["aff"])[:3, :3] @ S["dev_axis_w"]       # bore axis (voxel space)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    adap = np.argwhere(S["adap_mask"]).astype(float)
    c0 = adap.mean(0)
    d = adap - c0
    r_ad = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1)      # adapter radial extent
    r_in = max(np.percentile(r_ad, 5), (TX_DIAM_MM / 2) / vmm)   # inner bore radius (voxels)
    log(f"    adapter bore radius ~= {r_in * vmm:.1f} mm  (outer ~= {np.percentile(r_ad, 95) * vmm:.1f} mm)")

    # skull voxels inside that bore cylinder, kept near the adapter (near-side only)
    sk = np.argwhere(S["skull"]).astype(float)
    ds = sk - c0
    ax_s = ds @ ax
    r_s = np.linalg.norm(ds - np.outer(ax_s, ax), axis=1)
    in_cyl = (r_s <= r_in) & (np.abs(ax_s) <= 30.0 / vmm)
    hole = np.zeros(S["shape"], bool)
    p = np.round(sk[in_cyl]).astype(int)
    hole[p[:, 0], p[:, 1], p[:, 2]] = True
    lab, n = ndi.label(hole)                                     # keep the piece nearest the adapter
    if n > 1:
        coms = ndi.center_of_mass(hole, lab, range(1, n + 1))
        near = int(np.argmin([np.linalg.norm(np.array(cm) - c0) for cm in coms]))
        hole = lab == (near + 1)

    skull_cut = S["skull"] & ~hole
    S["skull_cut"] = skull_cut
    c = np.argwhere(hole).mean(0) if hole.any() else S["skull_c"]
    log(f"    removed {int(hole.sum()):,} skull voxels for the craniotomy")
    np.save(os.path.join(out, "step05_skull_cut.npy"), skull_cut)
    ortho(skull_cut.astype(float), c, os.path.join(out, "step05_hole.png"),
          "skull AFTER craniotomy  (hole=green, adapter=magenta dashed)",
          overlays=[(hole, "lime")], proj=[(S["adap_mask"], "magenta")], pts=[(S["target"], "red")])
    return S


# ── STEP 6: acoustic maps ─────────────────────────────────────────────────────
def step6_acoustics(data, out, S):
    log("[6] building HU-based acoustic maps ...")
    shape = S["shape"]
    c = np.full(shape, C_WATER, np.float32); rho = np.full(shape, RHO_WATER, np.float32)
    c[S["brain"]] = C_BRAIN; rho[S["brain"]] = RHO_BRAIN
    sk = S["skull_cut"]
    hu = np.clip(S["ct"], 0, None)
    hu_max = np.percentile(hu[sk], 99.5) if sk.any() else 1.0
    phi = np.clip(1.0 - hu / max(hu_max, 1.0), 0.0, 1.0)     # porosity (0 dense .. 1 water)
    c[sk] = (C_WATER * phi + C_BONE * (1 - phi))[sk]
    rho[sk] = (RHO_WATER * phi + RHO_BONE * (1 - phi))[sk]
    c[S["base_mask"]] = C_TI;  rho[S["base_mask"]] = RHO_TI
    c[S["adap_mask"]] = C_SIL; rho[S["adap_mask"]] = RHO_SIL
    S["c"] = c; S["rho"] = rho
    log(f"    c [{c.min():.0f},{c.max():.0f}] m/s  rho [{rho.min():.0f},{rho.max():.0f}] kg/m^3")
    ortho(c, S["entry"], os.path.join(out, "step06_soundspeed.png"), "sound speed (m/s)", cmap="viridis")
    np.save(os.path.join(out, "step06_c.npy"), c); np.save(os.path.join(out, "step06_rho.npy"), rho)
    return S


# ── STEP 7: aim transducer through the window ─────────────────────────────────
def step7_transducer(data, out, S):
    log("[7] aiming 19 mm transducer through the adapter at the target ...")
    vmm = S["vmm"]
    adap_vox = np.argwhere(S["adap_mask"]).astype(float)
    # transducer sits at the OUTER (skull-side) face of the adapter, aimed inward at target.
    out_dir = np.linalg.inv(S["aff"])[:3, :3] @ S["dev_axis_w"]        # device axis in voxel space
    out_dir = out_dir / (np.linalg.norm(out_dir) + 1e-9)
    brain_c = np.argwhere(S["brain"]).mean(0)
    if out_dir @ (adap_vox.mean(0) - brain_c) < 0:
        out_dir = -out_dir                                            # point outward (away from brain)
    pr = (adap_vox - adap_vox.mean(0)) @ out_dir
    outer = adap_vox[pr > np.percentile(pr, 60)]
    tx_center = outer.mean(0)
    aim = S["target"] - tx_center; aim = aim / (np.linalg.norm(aim) + 1e-9)   # aim axis (voxel space)
    tx = _disc_mask(tx_center, aim, (TX_DIAM_MM / 2) / vmm, S["shape"])
    S["tx_center"] = tx_center; S["tx_axis"] = aim; S["tx_mask"] = tx
    log(f"    transducer center vox {np.round(tx_center,1)}  aim {np.round(aim,2)}  ({int(tx.sum())} vox)")

    # save transducer pose + target (voxel and world mm)
    aim_w = S["aff"][:3, :3] @ aim; aim_w = aim_w / (np.linalg.norm(aim_w) + 1e-9)
    pose = {"transducer_center_vox": np.round(tx_center, 2).tolist(),
            "transducer_center_mm":  np.round(v2w(tx_center, S["aff"])[0], 2).tolist(),
            "aim_axis_vox": np.round(aim, 4).tolist(), "aim_axis_mm": np.round(aim_w, 4).tolist(),
            "target_vox": np.asarray(S["target"]).round(2).tolist(),
            "target_mm":  np.round(v2w(S["target"], S["aff"])[0], 2).tolist(),
            "diam_mm": TX_DIAM_MM, "freq_hz": FREQ_HZ}
    json.dump(pose, open(os.path.join(out, "step07_transducer.json"), "w"), indent=2)
    np.save(os.path.join(out, "step07_tx_mask.npy"), tx)
    ortho(S["c"], tx_center, os.path.join(out, "step07_transducer.png"),
          "transducer (cyan) + adapter (magenta proj) + STN (red proj)", cmap="viridis",
          overlays=[(tx, "cyan")], proj=[(S["adap_mask"], "magenta"), (S["stn"], "red")],
          pts=[(tx_center, "cyan"), (S["target"], "red")])
    return S


def _disc_mask(center, normal, R_vox, shape, thick=None):
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    thick = 0.5 * float(np.abs(normal).sum()) if thick is None else thick
    L = int(np.ceil(R_vox)) + 2
    c = np.round(center).astype(int)
    lo = np.clip(c - L, 0, np.array(shape) - 1); hi = np.clip(c + L + 1, 0, np.array(shape))
    gx, gy, gz = np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij")
    P = np.stack([gx - center[0], gy - center[1], gz - center[2]], -1)
    dperp = P @ normal
    r2 = (P ** 2).sum(-1) - dperp ** 2
    local = (np.abs(dperp) <= thick) & (r2 <= R_vox ** 2)
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = local
    return m


def _revolve(rr, top, nth=180):
    """Watertight surface-of-revolution solid from a radial top-profile top(rr),
    flat base at z=0, axis = +z (printable orientation)."""
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False); cs, sn = np.cos(th), np.sin(th)
    M = len(rr)
    V = [(0, 0, top[0]), (0, 0, 0.0)]                              # 0=top apex, 1=bottom centre
    for i in range(1, M):
        for j in range(nth): V.append((rr[i] * cs[j], rr[i] * sn[j], top[i]))
    off = 2 + (M - 1) * nth
    for i in range(1, M):
        for j in range(nth): V.append((rr[i] * cs[j], rr[i] * sn[j], 0.0))
    Tp = lambda i, j: 2 + (i - 1) * nth + (j % nth)
    Bp = lambda i, j: off + (i - 1) * nth + (j % nth)
    F = []
    for j in range(nth):
        F.append((0, Tp(1, j), Tp(1, j + 1)))                     # top apex fan
        F.append((1, Bp(1, j + 1), Bp(1, j)))                     # bottom centre fan
    for i in range(1, M - 1):
        for j in range(nth):
            F += [(Tp(i, j), Tp(i + 1, j), Tp(i + 1, j + 1)), (Tp(i, j), Tp(i + 1, j + 1), Tp(i, j + 1))]
            F += [(Bp(i, j), Bp(i + 1, j + 1), Bp(i + 1, j)), (Bp(i, j), Bp(i, j + 1), Bp(i + 1, j + 1))]
    i = M - 1
    for j in range(nth):                                          # outer rim wall
        F += [(Tp(i, j), Bp(i, j), Bp(i, j + 1)), (Tp(i, j), Bp(i, j + 1), Tp(i, j + 1))]
    m = trimesh.Trimesh(vertices=np.array(V, float), faces=np.array(F, int), process=True)
    m.fix_normals()
    return m


def _heightfield_solid(xs, ys, Ztop, valid):
    """Watertight solid from a masked height field: variable top surface z=Ztop[i,j],
    flat base at z=0, vertical side walls on the mask boundary. This realises a
    'constant-thickness, variable-LENGTH' lens -- each aperture point is a resin column
    whose length (Ztop) sets its acoustic delay; longer = more delay/advance."""
    nx, ny = Ztop.shape
    idx = -np.ones((nx, ny), int); V = []; k = 0
    for i in range(nx):
        for j in range(ny):
            if valid[i, j]:
                idx[i, j] = k; k += 1
                V.append((float(xs[i]), float(ys[j]), float(Ztop[i, j])))    # top vertex
    nt = k
    for i in range(nx):
        for j in range(ny):
            if valid[i, j]:
                V.append((float(xs[i]), float(ys[j]), 0.0))                  # base vertex
    Tv = lambda i, j: idx[i, j]
    Bv = lambda i, j: idx[i, j] + nt
    solid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if not solid[i, j]:
                continue
            F += [(Tv(i, j), Tv(i + 1, j), Tv(i + 1, j + 1)), (Tv(i, j), Tv(i + 1, j + 1), Tv(i, j + 1))]
            F += [(Bv(i, j), Bv(i + 1, j + 1), Bv(i + 1, j)), (Bv(i, j), Bv(i, j + 1), Bv(i + 1, j + 1))]
    def wall(a, b):
        F.append((Tv(*a), Tv(*b), Bv(*b))); F.append((Tv(*a), Bv(*b), Bv(*a)))
    for i in range(nx - 1):                                    # x-edges: bounded by solid[i,j-1] & solid[i,j]
        for j in range(ny):
            q0 = solid[i, j - 1] if j - 1 >= 0 else False
            q1 = solid[i, j] if j < ny - 1 else False
            if (q0 ^ q1) and valid[i, j] and valid[i + 1, j]:
                wall((i, j), (i + 1, j))
    for i in range(nx):                                        # y-edges: bounded by solid[i-1,j] & solid[i,j]
        for j in range(ny - 1):
            q0 = solid[i - 1, j] if i - 1 >= 0 else False
            q1 = solid[i, j] if i < nx - 1 else False
            if (q0 ^ q1) and valid[i, j] and valid[i, j + 1]:
                wall((i, j), (i, j + 1))
    m = trimesh.Trimesh(vertices=np.array(V, float), faces=np.array(F, int), process=True)
    try:
        m.fix_normals()                       # needs networkx only when winding is inconsistent
    except Exception as e:
        log(f"    (skipping fix_normals: {e}; slicers repair face normals)")
    return m


def _pose_block(S):
    """Shared header + transducer pose (scan/world mm) for transducer_spec.txt."""
    aff = S["aff"]
    txc_mm = np.asarray(v2w(S["tx_center"], aff)[0], float)
    tgt_mm = np.asarray(v2w(S["target"], aff)[0], float)
    aim_w = aff[:3, :3] @ S["tx_axis"]; aim_w = aim_w / (np.linalg.norm(aim_w) + 1e-9)
    az = float(np.degrees(np.arctan2(aim_w[1], aim_w[0])))
    el = float(np.degrees(np.arcsin(np.clip(aim_w[2], -1, 1))))
    F = float(np.linalg.norm(tgt_mm - txc_mm))
    name = S.get("target_name") or "label " + str(S.get("target_label", ""))
    txt = (
        f"TRANSDUCER ORIENTATION & LENS SPEC  (target: {name})\n"
        "================================================================\n"
        f"frequency                 {FREQ_HZ/1e6:.3f} MHz\n"
        f"aperture diameter         {TX_DIAM_MM:.1f} mm\n\n"
        "-- pose in scan/world frame (mm) --\n"
        f"transducer face centre    ({txc_mm[0]:.2f}, {txc_mm[1]:.2f}, {txc_mm[2]:.2f})\n"
        f"target                    ({tgt_mm[0]:.2f}, {tgt_mm[1]:.2f}, {tgt_mm[2]:.2f})\n"
        f"aim axis (unit, ->target) ({aim_w[0]:.4f}, {aim_w[1]:.4f}, {aim_w[2]:.4f})\n"
        f"azimuth / elevation       {az:.1f} deg / {el:.1f} deg\n"
        f"focal distance            {F:.2f} mm\n\n")
    return txt, F, az, el


# ── STEP 7b: focusing lens (printable STL) + orientation spec ─────────────────
def step7b_lens(data, out, S):
    log("[7b] preview lens (geometric water-path) + orientation spec ...")
    pose, F, az, el = _pose_block(S)
    R = TX_DIAM_MM / 2.0

    # PREVIEW-ONLY axisymmetric lens: converts the flat aperture into a wavefront focusing
    # at F through a HOMOGENEOUS water path (ignores the skull). This is what the geometry-only
    # (--preview, no GPU) run can produce. When the full sim runs, build_lens_from_delays()
    # OVERWRITES lens.stl with the freeform, skull-aberration-corrected lens.
    inv = 1.0 / C_LENS - 1.0 / C_WATER
    rr = np.linspace(0.0, R, 72)
    tt = np.sqrt(rr ** 2 + F ** 2) / 1e3 / C_WATER               # medium travel time per radius (s)
    T0 = tt.max() if inv > 0 else tt.min()                        # pick sign -> h >= 0
    h = (T0 - tt) / inv * 1e3                                     # lens length profile (mm)
    h = h - h.min()
    lens = _revolve(rr, LENS_BASE_MM + h)
    lens.export(os.path.join(out, "lens.stl"))

    spec = pose + (
        "-- printable lens (lens.stl) : PREVIEW, axisymmetric geometric (water path only) --\n"
        f"material sound speed      {C_LENS:.0f} m/s (coupling medium {C_WATER:.0f} m/s)\n"
        f"clear aperture            {2*R:.1f} mm\n"
        f"base plate thickness      {LENS_BASE_MM:.1f} mm\n"
        f"length range              {h.min():.2f}..{h.max():.2f} mm\n"
        f"lens volume               {lens.volume/1e3:.2f} cm^3   watertight={lens.is_watertight}\n\n"
        "NOTE: this preview lens corrects only the geometric water path -- it does NOT\n"
        "correct skull aberration. Run the full sim (--run-sim): a BACKWARD time-reversal\n"
        "sim then designs the real freeform lens and a FORWARD sim confirms the true focus.\n"
    )
    with open(os.path.join(out, "transducer_spec.txt"), "w", encoding="utf-8") as f:
        f.write(spec)
    log(f"    preview lens.stl  length {h.min():.2f}-{h.max():.2f} mm, "
        f"watertight={lens.is_watertight};  focal {F:.1f} mm, az/el {az:.0f}/{el:.0f} deg")
    write_preview_viewer(out, S)                 # geometry-only 3-D viewer to confirm placement
    return S


# ── STEP 8: phase-plate delays ────────────────────────────────────────────────
def step8_delays(data, out, S):
    log("[8] phase-plate focusing delays to the target ...")
    S["focus_target"] = S["target"]           # in the FULL grid; remapped after crop
    log("    (delays computed on the resampled sim grid in step 9/10)")
    return S


# ── STEP 9: crop + resample to isotropic dx_sim ───────────────────────────────
def step9_crop(data, out, S, dx_sim_mm):
    log(f"[9] cropping beam corridor + resampling to {dx_sim_mm} mm iso ...")
    # bound just the transducer -> target corridor (+ margin), NOT the whole head
    corridor = np.vstack([np.argwhere(S["tx_mask"]), np.atleast_2d(S["target"])])
    lo = np.floor(corridor.min(0)).astype(int); hi = np.ceil(corridor.max(0)).astype(int)
    mgn = int(np.ceil((TX_DIAM_MM * 0.7 + 12) / S["vmm"]))   # lateral/depth margin (voxels)
    lo = np.maximum(lo - mgn, 0); hi = np.minimum(hi + mgn + 1, np.array(S["shape"]))
    sl = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
    zooms = np.linalg.norm(S["aff"][:3, :3], axis=0)   # true voxel size per axis (mm);
    zoom = zooms / dx_sim_mm                            # diag() is wrong for a rotated affine
    c_c = ndi.zoom(S["c"][sl],   zoom, order=1)
    r_c = ndi.zoom(S["rho"][sl], zoom, order=1)
    tx_c = ndi.zoom(S["tx_mask"][sl].astype(np.float32), zoom, order=1) > 0.5
    br_c = ndi.zoom(S["brain"][sl].astype(np.float32), zoom, order=1) > 0.5
    st_c = ndi.zoom(S["stn"][sl].astype(np.float32),  zoom, order=1) > 0.5
    # FFT-friendly padding
    def smooth(n):
        while True:
            m = n
            for p in (2, 3, 5, 7):
                while m % p == 0: m //= p
            if m == 1: return n
            n += 1
    pad = [(0, smooth(s) - s) for s in c_c.shape]
    c_c = np.pad(c_c, pad, constant_values=C_WATER)
    r_c = np.pad(r_c, pad, constant_values=RHO_WATER)
    tx_c = np.pad(tx_c, pad); br_c = np.pad(br_c, pad); st_c = np.pad(st_c, pad)
    tgt_c = np.round(np.argwhere(st_c).mean(0)).astype(int) if st_c.any() else np.array(c_c.shape) // 2
    S.update(c_crop=c_c.astype(np.float32), rho_crop=r_c.astype(np.float32), tx_crop=tx_c,
             brain_crop=br_c, stn_crop=st_c, target_crop=tgt_c, dx=dx_sim_mm * 1e-3)
    log(f"    sim grid {c_c.shape}  ({np.array(c_c.shape)*dx_sim_mm} mm)  target {tgt_c.tolist()}")
    np.savez_compressed(os.path.join(out, "step09_sim_input.npz"),
                        c_crop=S["c_crop"], rho_crop=S["rho_crop"], tx_crop=tx_c.astype(np.uint8),
                        brain_crop=br_c.astype(np.uint8), stn_crop=st_c.astype(np.uint8),
                        target=tgt_c, dx=np.float64(S["dx"]), freq=np.float64(FREQ_HZ),
                        P0=np.float64(P0_PA), N_CYCLES=np.int64(N_CYCLES))
    ortho(S["c_crop"], tgt_c, os.path.join(out, "step09_simgrid.png"),
          "sim sound-speed + tx(cyan)/brain(yellow)/target(red)", cmap="viridis",
          overlays=[(tx_c, "cyan"), (br_c, "yellow"), (st_c, "red")], pts=[(tgt_c, "red")])
    return S


def _focus_medium(c, rho, tx, T, dx):
    """Shared medium for the backward + forward focus sims: smoothed sound-speed/density
    plus an absorbing backing cylinder just behind the transducer (forward-only radiation).
    Using the SAME medium in both sims keeps time-reversal reciprocity valid."""
    from kwave.kmedium import kWaveMedium
    c_sm = ndi.gaussian_filter(c.astype(np.float64), 0.6).astype(np.float32)
    r_sm = ndi.gaussian_filter(rho.astype(np.float64), 0.6).astype(np.float32)
    txc = np.argwhere(tx).mean(0)
    bdir = txc - T; bdir = bdir / (np.linalg.norm(bdir) + 1e-9)       # points away from target
    rel = np.indices(c.shape, dtype=np.float32).reshape(3, -1).T - txc
    axc = rel @ bdir
    rad = np.linalg.norm(rel - np.outer(axc, bdir), axis=1)
    R_b = (TX_DIAM_MM / 2 + 1.5) / (dx * 1e3); L_b = BACK_LEN_MM / (dx * 1e3)
    back = ((axc >= 1.0) & (axc <= L_b) & (rad <= R_b)).reshape(c.shape) & ~tx
    alpha = np.zeros(c.shape, np.float32); alpha[back] = 40.0         # dB/(MHz^y cm)
    medium = kWaveMedium(sound_speed=c_sm, density=r_sm, alpha_coeff=alpha, alpha_power=1.5)
    return medium, back


# ── STEP 9b: BACKWARD (time-reversal) design sim ──────────────────────────────
def step9b_backward(data, out, S):
    """Put a point source AT the target, propagate through the real skull, and record the
    pressure arriving on the transducer aperture. By reciprocity the phase-conjugate of what
    arrives is what a single transducer + lens must emit to refocus there THROUGH that skull.
    Yields a per-aperture emission delay -> the lens (step 10c) and the forward source use it."""
    log("[9b] backward time-reversal design sim (point source at target -> aperture) ...")
    from kwave.kgrid import kWaveGrid
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.utils.signals import tone_burst
    from scipy.signal import fftconvolve
    c, rho, tx = S["c_crop"], S["rho_crop"], S["tx_crop"]
    dx = S["dx"]; T = S["target_crop"].astype(float)
    npz = os.path.join(out, "step09b_delays.npz")
    if os.path.exists(npz):                      # reuse if aperture + target match (skip the GPU sim)
        d = np.load(npz)
        saved_tgt = d["target"] if "target" in d.files else T   # older files predate the target key
        if d["ap_coords"].shape[0] == int(tx.sum()) and np.allclose(saved_tgt, T, atol=1.0):
            S["delays_sec"] = d["delays_sec"]; S["ap_coords"] = d["ap_coords"]
            log(f"    reusing saved delays ({d['ap_coords'].shape[0]} pts; delete {os.path.basename(npz)} to force re-run)")
            return S
    Nx, Ny, Nz = c.shape
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    t_end = 1.8 * (max(c.shape) * dx) / max(float(c.min()), C_WATER)
    kgrid.makeTime(float(c.max()), cfl=0.3, t_end=t_end)             # same dt as the forward sim
    medium, _ = _focus_medium(c, rho, tx, T, dx)
    base = tone_burst(1 / kgrid.dt, FREQ_HZ, N_CYCLES).flatten().astype(np.float32) * P0_PA
    # point source at the target (a small ball; every voxel emits the same burst)
    V = np.round(T).astype(int)
    pm = np.zeros(c.shape, np.uint8)
    sl = tuple(slice(max(0, V[a] - 1), min(c.shape[a], V[a] + 2)) for a in range(3))
    pm[sl] = 1
    src = kSource(); src.p_mask = pm; src.p = np.tile(base, (int(pm.sum()), 1))
    sen = kSensor(); sen.mask = tx.astype(np.uint8); sen.record = ["p"]   # full time-history on aperture
    ff = np.flatnonzero(tx.ravel(order="F"))
    Sxyz = np.array(np.unravel_index(ff, tx.shape, order="F")).T.astype(float)
    log(f"    aperture {len(Sxyz)} pts  Nt={kgrid.Nt}  point-source {int(pm.sum())} vox at {V.tolist()}")
    t0 = time.time()
    sd = kspaceFirstOrder(kgrid, medium, src, sen, device="gpu", dtype="float32")
    pap = np.nan_to_num(np.asarray(sd["p"], dtype=np.float32))
    if pap.shape[0] != len(Sxyz) and pap.shape[-1] == len(Sxyz):
        pap = pap.T                                                  # normalise to (n_ap, Nt)
    # arrival time per aperture point via MATCHED FILTER (cross-correlate with the emitted burst):
    # the direct arrival looks most like the burst, so this rejects skull reverberation/multipath
    # better than an envelope-peak argmax.
    Lb = base.shape[0]
    corr = fftconvolve(pap, base[::-1][None, :], mode="full", axes=1)
    tau = np.clip((corr.argmax(axis=1).astype(np.float64) - (Lb - 1)) * kgrid.dt, 0, None)
    delays = tau.max() - tau                                        # latest arrival emits first
    delays = delays - delays.min()
    S["delays_sec"] = delays; S["ap_coords"] = Sxyz
    np.savez_compressed(os.path.join(out, "step09b_delays.npz"),
                        delays_sec=delays, ap_coords=Sxyz, dt=np.float64(kgrid.dt), target=T)
    log(f"    done {time.time()-t0:.0f}s  aperture delay spread {delays.max()*1e6:.2f} us "
        f"({int(round(delays.max()/kgrid.dt))} samples)")
    return S


# ── STEP 10c: freeform variable-length lens from the backward-sim delays ──────
def build_lens_from_delays(data, out, S):
    log("[10c] building freeform variable-length lens from backward-sim delays ...")
    from scipy.interpolate import griddata
    Sxyz = np.asarray(S["ap_coords"], float); delays = np.asarray(S["delays_sec"], float)
    dx_mm = S["dx"] * 1e3
    txc = Sxyz.mean(0); T = S["target_crop"].astype(float)
    aimv = T - txc; aimv = aimv / (np.linalg.norm(aimv) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(aimv[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(aimv, ref); e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(aimv, e1);  e2 /= np.linalg.norm(e2) + 1e-9
    q = (Sxyz - txc) * dx_mm
    u = q @ e1; v = q @ e2                                           # aperture-plane coords (mm)
    # per-point resin LENGTH that produces each point's emission delay (single transducer + lens)
    inv = 1.0 / C_WATER - 1.0 / C_LENS                              # >0 if resin faster than water
    if inv > 0:                                                     # faster resin: longest where beam leads
        L = (delays.max() - delays) / inv
    else:                                                          # slower resin: longest where beam lags
        L = (delays - delays.min()) / (-inv)
    L = (L - L.min()) * 1e3                                         # mm, >= 0
    R = TX_DIAM_MM / 2.0
    step = max(0.3, dx_mm)
    g = np.arange(-R, R + step, step)
    Xg, Yg = np.meshgrid(g, g, indexing="ij")
    Lin = griddata(np.column_stack([u, v]), L, (Xg, Yg), method="linear")
    Lnn = griddata(np.column_stack([u, v]), L, (Xg, Yg), method="nearest")
    Lg = np.where(np.isnan(Lin), Lnn, Lin)
    Lg = ndi.gaussian_filter(Lg, 0.7)                              # printability: smooth spikes
    # FRESNEL WRAP: a delay of tau and tau+one-period are acoustically equivalent (near-CW), so wrap
    # the length modulo one acoustic period. Collapses a many-mm refractive lens to <1 wavelength of
    # material -> thin, printable, and it clears the brain instead of a long spike into it.
    L_T = (1.0 / FREQ_HZ) / abs(inv) * 1e3                         # lens length for one period (mm)
    Lg = np.mod(np.clip(Lg, 0, None), L_T)
    valid = (Xg ** 2 + Yg ** 2) <= R ** 2
    Lg = np.where(valid, Lg, 0.0)
    Ztop = LENS_BASE_MM + Lg
    lmm_lo, lmm_hi = float(Lg[valid].min()), float(Lg[valid].max())
    lens = _heightfield_solid(g, g, Ztop, valid)
    lens.export(os.path.join(out, "lens.stl"))

    # rasterise the lens into the sim-crop grid so the 3-D viewer can show it: from each aperture
    # point, a resin column of length LENS_BASE_MM+L runs along +aim (transducer face -> skull side)
    shp = np.asarray(S["c_crop"].shape)
    lens_vol = np.zeros(tuple(int(x) for x in shp), bool)
    ii, jj = np.where(valid)
    for a, b in zip(ii, jj):
        Lmm = LENS_BASE_MM + float(Lg[a, b])
        P0 = txc + (Xg[a, b] / dx_mm) * e1 + (Yg[a, b] / dx_mm) * e2
        for s in np.linspace(0.0, Lmm / dx_mm, max(2, int(Lmm / dx_mm / 0.5) + 1)):
            qv = np.round(P0 + s * aimv).astype(int)
            if (qv >= 0).all() and (qv < shp).all():
                lens_vol[qv[0], qv[1], qv[2]] = True
    S["lens_vol"] = lens_vol
    log(f"    lens occupies {int(lens_vol.sum())} sim voxels (shown in the 3-D viewer)")

    if all(k in S for k in ("aff", "tx_center", "tx_axis", "target")):   # skipped on --replot (no pose)
        pose, F, az, el = _pose_block(S)
        spec = pose + (
            "-- printable lens (lens.stl) : FREEFORM variable-length, FRESNEL-WRAPPED, ABERRATION-CORRECTED --\n"
            f"material sound speed      {C_LENS:.0f} m/s (coupling medium {C_WATER:.0f} m/s)\n"
            f"clear aperture            {2*R:.1f} mm\n"
            f"base plate thickness      {LENS_BASE_MM:.1f} mm\n"
            f"Fresnel wrap length       {L_T:.2f} mm (= one acoustic period)\n"
            f"variable length range     {lmm_lo:.2f}..{lmm_hi:.2f} mm  "
            f"(total height {float(Ztop[valid].min()):.2f}..{float(Ztop.max()):.2f} mm)\n"
            f"aperture delay spread     {delays.max()*1e6:.2f} us (unwrapped {L.max():.1f} mm; wrapped above)\n"
            f"lens volume               {lens.volume/1e3:.2f} cm^3   watertight={lens.is_watertight}\n\n"
            "Designed from a BACKWARD time-reversal sim through the real CT skull; each aperture point\n"
            "is a resin column whose length imposes its delay, wrapped modulo one acoustic period (a\n"
            "Fresnel/hologram lens) so it stays thin and clears the brain. A single transducer emits ONE\n"
            "waveform; this lens shapes it. Drive near-CW. The FORWARD (confirm) sim reports the true focus.\n"
        )
        with open(os.path.join(out, "transducer_spec.txt"), "w", encoding="utf-8") as f:
            f.write(spec)
    log(f"    lens.stl  wrapped length {lmm_lo:.2f}-{lmm_hi:.2f} mm (period {L_T:.2f} mm), "
        f"watertight={lens.is_watertight}, vol {lens.volume/1e3:.2f} cm^3")
    return S


# ── STEP 10: k-Wave ───────────────────────────────────────────────────────────
def step10_sim(data, out, S):
    log("[10] running k-Wave (GPU) ...")
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.utils.signals import tone_burst

    c, rho, tx = S["c_crop"], S["rho_crop"], S["tx_crop"]
    dx = S["dx"]; T = S["target_crop"].astype(float)
    Nx, Ny, Nz = c.shape
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    t_end = 1.8 * (max(c.shape) * dx) / max(float(c.min()), C_WATER)   # bulk-min speed, not the thin lens
    kgrid.makeTime(float(c.max()), cfl=0.3, t_end=t_end)     # dt from fastest medium (CFL)

    # absorbing backing behind the transducer (forward-only); same medium as the backward sim
    medium, back = _focus_medium(c, rho, tx, T, dx)
    log(f"    backing cylinder {int(back.sum())} vox behind transducer (forward-only)")

    base = tone_burst(1 / kgrid.dt, FREQ_HZ, N_CYCLES).flatten() * P0_PA
    src = kSource(); src.p_mask = tx.astype(np.uint8)
    ff = np.flatnonzero(tx.ravel(order="F"))                 # this build: source.p is F-order
    Sxyz = np.array(np.unravel_index(ff, tx.shape, order="F")).T.astype(float)
    if S.get("delays_sec") is not None and len(S["delays_sec"]) == len(Sxyz):
        nsh = np.round(np.asarray(S["delays_sec"]) / kgrid.dt).astype(int)   # from backward sim
        log("    focusing delays: BACKWARD time-reversal (skull-aberration corrected)")
    else:
        dd = np.linalg.norm((Sxyz - T) * dx, axis=1)                         # fallback: ideal water path
        nsh = np.round((dd.max() - dd) / C_WATER / kgrid.dt).astype(int)
        log("    focusing delays: geometric water-path (no backward sim available)")
    period = max(1, int(round((1.0 / FREQ_HZ) / kgrid.dt)))                  # samples per acoustic period
    nsh = np.mod(nsh, period)              # Fresnel-wrapped thin lens: delays are mod one period (near-CW)
    nsh = np.clip(nsh - nsh.min(), 0, kgrid.Nt - 1)
    psrc = np.zeros((len(Sxyz), kgrid.Nt), np.float32)
    for i, sh in enumerate(nsh):
        seg = base[: max(0, kgrid.Nt - int(sh))]; psrc[i, int(sh):int(sh) + len(seg)] = seg
    src.p = psrc

    V = np.round(T).astype(int)
    # This k-Wave build stores the FULL pressure time-history for EVERY sensor point
    # (n_points x Nt float32), so recording all voxels OOMs (~0.5 TB). To still get a
    # filled 3-D cloud we record the densest lattice of the whole volume that fits VRAM.
    # Measured on the A4500: solver fields+temps ~= BASE (grid-size only, ~4.8 GB for the
    # 252^3 grid); the sensor costs n_points*Nt*4 bytes. Size the lattice to the free VRAM.
    Nt = kgrid.Nt
    import cupy as _cp
    free_b = int(_cp.cuda.Device().mem_info[0])               # bytes physically free right now
    avail = min(free_b, float(S.get("lease_gb", 16.0)) * 1e9)
    BASE = 5.5e9 * (c.size / (252 * 240 * 245))              # solver footprint scales with grid size
    SAFETY = 1.5e9                                            # headroom (measured on A4500)
    manual = int(S.get("sensor_stride", 0) or 0)
    if manual >= 1:
        ix = [np.arange(0, c.shape[a], manual) for a in range(3)]
    else:
        n_max = max(1e3, (avail - BASE - SAFETY) / (Nt * 4))  # sensor points that fit VRAM
        k = min(1.0, (n_max / c.size) ** (1.0 / 3))           # per-axis fraction to keep
        dims = np.maximum(2, np.round(np.array(c.shape) * k)).astype(int)
        ix = [np.unique(np.linspace(0, c.shape[a] - 1, dims[a]).round().astype(int)) for a in range(3)]
    sen_mask = np.zeros(c.shape, bool)
    sen_mask[np.ix_(ix[0], ix[1], ix[2])] = True
    npts = int(sen_mask.sum())
    eff_mm = (c.size / npts) ** (1.0 / 3) * (dx * 1e3)        # effective pressure resolution
    sen = kSensor(); sen.mask = sen_mask.astype(np.uint8); sen.record = ["p_max"]
    log(f"    grid {Nx}x{Ny}x{Nz}  Nt={Nt}  free={free_b/1e9:.1f}GB  sensor={npts:,} "
        f"(~{npts*Nt*4/1e9:.1f}GB, {eff_mm:.2f}mm eff, predicted peak ~{(BASE+npts*Nt*4)/1e9:.1f}GB)")
    t0 = time.time()
    sd = kspaceFirstOrder(kgrid, medium, src, sen, device="gpu", dtype="float32")
    pv = np.zeros(c.shape, np.float32); pv[sen_mask] = np.squeeze(np.asarray(sd["p_max"]))
    pv = np.nan_to_num(pv)
    coarse = pv[np.ix_(ix[0], ix[1], ix[2])].copy()           # dense recorded lattice
    p = ndi.zoom(coarse, np.array(c.shape) / np.array(coarse.shape), order=1).astype(np.float32)
    log(f"    done {time.time()-t0:.0f}s  p_max={p.max():.3e} Pa  (recorded grid {coarse.shape})")
    S["p"] = p; S["view"] = V
    np.savez_compressed(os.path.join(out, "step10_pressure.npz"), p_coarse=coarse,
                        shape=np.array(c.shape), view=V, target=S["target_crop"], dx=np.float64(dx),
                        lens=S.get("lens_vol", np.zeros(c.shape, bool)).astype(np.uint8))
    return S


# ── self-contained WebGL2 3-D volume viewer ───────────────────────────────────
VIEWER_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>LIFU pressure — 3D viewer</title>
<style>
 html,body{margin:0;height:100%;background:#05060a;color:#cdd6f4;font:13px system-ui,sans-serif;overflow:hidden}
 #c{display:block;width:100vw;height:100vh;cursor:grab} #c:active{cursor:grabbing}
 #ui{position:fixed;top:12px;left:12px;background:rgba(15,18,28,.82);padding:12px 14px;border-radius:10px;width:230px}
 #ui h1{font-size:14px;margin:0 0 8px} #ui label{display:block;margin:9px 0 2px;font-size:11px;opacity:.8}
 #ui input[type=range]{width:100%} .leg{display:flex;align-items:center;gap:6px;margin-top:6px;font-size:11px}
 .sw{width:12px;height:12px;border-radius:3px}
 #hint{position:fixed;bottom:10px;left:12px;font-size:11px;opacity:.55}
 #err{position:fixed;inset:0;display:none;align-items:center;justify-content:center;padding:40px;text-align:center}
</style></head><body>
<canvas id="c"></canvas>
<div id="ui">
 <h1>__TITLE__</h1>
 <label>Pressure opacity <span id="vpo"></span></label><input id="po" type="range" min="0" max="1" step="0.01" value="0.5">
 <label>Pressure threshold <span id="vpt"></span></label><input id="pt" type="range" min="0" max="1" step="0.01" value="0.15">
 <label>Skull/brain ghost <span id="vao"></span></label><input id="ao" type="range" min="0" max="0.15" step="0.005" value="0.04">
 <label>STN target opacity <span id="vso"></span></label><input id="so" type="range" min="0" max="1" step="0.01" value="0.55">
 <label>Transducer opacity <span id="vto"></span></label><input id="to" type="range" min="0" max="1" step="0.01" value="0.55">
 <label>Lens opacity <span id="vlo"></span></label><input id="lo" type="range" min="0" max="1" step="0.01" value="0.75">
 <label>Cutaway (slice in) <span id="vcl"></span></label><input id="cl" type="range" min="0.15" max="1" step="0.01" value="1">
 <label>Peak marker <span id="vmo"></span></label><input id="mo" type="range" min="0" max="1" step="0.01" value="0.9">
 <div class="leg"><div class="sw" style="background:linear-gradient(90deg,#500,#f80,#ff8)"></div>pressure</div>
 <div class="leg"><div class="sw" style="background:#ff26f2"></div>brain-pressure peak</div>
 <div class="leg"><div class="sw" style="background:#28ff4b"></div>STN target</div>
 <div class="leg"><div class="sw" style="background:#33ccff"></div>transducer</div>
 <div class="leg"><div class="sw" style="background:#ffc748"></div>lens</div>
 <div class="leg"><div class="sw" style="background:#8a93a8"></div>skull / brain</div>
</div>
<div id="hint">drag = rotate · scroll = zoom</div>
<div id="err"><div><h2>WebGL2 not available</h2><p>Open this file in Chrome, Edge, or Firefox.</p></div></div>
<script>
window.onerror=function(m,s,l,c){var e=document.getElementById('err');
 if(e){e.style.display='flex';e.innerHTML='<div style="max-width:600px"><h2>viewer error</h2><pre style="white-space:pre-wrap;text-align:left">'+m+'\n@ line '+l+':'+c+'</pre></div>';}return false;};
const NX=__NX__,NY=__NY__,NZ=__NZ__,PEAK=[__PEAK__];
const bin=atob("__DATA__"),bytes=new Uint8Array(bin.length);
for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
const cv=document.getElementById('c'),gl=cv.getContext('webgl2');
if(!gl){document.getElementById('err').style.display='flex';}
else{
const VS=`#version 300 es
in vec2 pos;out vec2 uv;void main(){uv=pos;gl_Position=vec4(pos,0.,1.);}`;
const FS=`#version 300 es
precision highp float;precision highp sampler3D;
in vec2 uv;out vec4 frag;
uniform sampler3D vol;uniform mat3 rot;uniform vec3 bh;
uniform float aspect,dist,pOp,pTh,aOp,stnOp,txOp,lensOp,clip,mOp;
uniform vec3 peak;
vec3 hot(float t){t=clamp(t,0.,1.);return clamp(vec3(1.6*t,1.7*t-0.6,3.2*t-2.2),0.,1.);}
bool box(vec3 ro,vec3 rd,vec3 b,out float t0,out float t1){
 vec3 iv=1.0/rd,a=(-b-ro)*iv,c=(b-ro)*iv,mn=min(a,c),mx=max(a,c);
 t0=max(max(mn.x,mn.y),mn.z);t1=min(min(mx.x,mx.y),mx.z);return t1>max(t0,0.0);}
void main(){
 vec2 p=vec2(uv.x*aspect,uv.y);
 vec3 ro=vec3(0.,0.,dist),rd=normalize(vec3(p,-1.6));
 vec3 roL=rot*ro,rdL=rot*rd;float t0,t1;
 if(!box(roL,rdL,bh,t0,t1)){frag=vec4(0.02,0.024,0.04,1.);return;}
 t0=max(t0,0.0);float dt=(t1-t0)/192.0;vec3 acc=vec3(0.);float aa=0.;
 for(int i=0;i<192;i++){
  vec3 pl=roL+rdL*(t0+(float(i)+0.5)*dt);
  vec3 tc=(pl/bh)*0.5+0.5;
  if(tc.z>clip)continue;
  vec4 s=texture(vol,tc);
  float aP=smoothstep(pTh,1.0,s.r)*pOp;
  float aA=s.g*s.g*aOp;
  float aS=s.b*stnOp;
  float aT=smoothstep(0.75,0.9,s.a)*txOp;                     // device chan: ~1.0 = transducer
  float aLn=(smoothstep(0.25,0.42,s.a)-smoothstep(0.58,0.75,s.a))*lensOp;  // ~0.5 = lens
  vec3 col=hot(s.r)*aP+vec3(0.54,0.58,0.66)*aA+vec3(0.16,1.0,0.30)*aS
           +vec3(0.20,0.80,1.0)*aT+vec3(1.0,0.78,0.28)*aLn;
  float a=clamp(aP+aA+aS+aT+aLn,0.,1.);
  float mk=(1.0-smoothstep(0.0,0.028,length(tc-peak)))*mOp;   // peak-pressure marker
  col+=vec3(1.0,0.15,0.95)*mk; a=clamp(a+mk,0.,1.);
  acc+=(1.0-aa)*col;aa+=(1.0-aa)*a;if(aa>0.985)break;
 }
 frag=vec4(acc+(1.0-aa)*vec3(0.02,0.024,0.04),1.0);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
 if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(o));return o;}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(pr);gl.useProgram(pr);
const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
const lp=gl.getAttribLocation(pr,'pos');gl.enableVertexAttribArray(lp);gl.vertexAttribPointer(lp,2,gl.FLOAT,false,0,0);
const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,tex);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
for(const w of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,w,gl.CLAMP_TO_EDGE);
gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,NX,NY,NZ,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
const md=Math.max(NX,NY,NZ),bh=[NX/md*0.5,NY/md*0.5,NZ/md*0.5];
const U=n=>gl.getUniformLocation(pr,n);
let yaw=0.6,pitch=0.3,dist=2.6;
function mul(A,B){const M=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)M[i][j]+=A[i][k]*B[k][j];return M;}
function rotArr(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch);
 const Ry=[[cy,0,sy],[0,1,0],[-sy,0,cy]],Rx=[[1,0,0],[0,cx,-sx],[0,sx,cx]],M=mul(Ry,Rx);
 return [M[0][0],M[0][1],M[0][2],M[1][0],M[1][1],M[1][2],M[2][0],M[2][1],M[2][2]];}
function draw(){const w=cv.clientWidth,h=cv.clientHeight;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
 gl.viewport(0,0,w,h);gl.useProgram(pr);gl.bindTexture(gl.TEXTURE_3D,tex);
 gl.uniform1i(U('vol'),0);gl.uniformMatrix3fv(U('rot'),false,rotArr());
 gl.uniform3f(U('bh'),bh[0],bh[1],bh[2]);gl.uniform1f(U('aspect'),w/h);gl.uniform1f(U('dist'),dist);
 gl.uniform1f(U('pOp'),+po.value);gl.uniform1f(U('pTh'),+pt.value);gl.uniform1f(U('aOp'),+ao.value);
 gl.uniform1f(U('stnOp'),+so.value);gl.uniform1f(U('txOp'),+to.value);gl.uniform1f(U('lensOp'),+lo.value);
 gl.uniform1f(U('clip'),+cl.value);
 gl.uniform1f(U('mOp'),+mo.value);gl.uniform3f(U('peak'),PEAK[0],PEAK[1],PEAK[2]);
 gl.drawArrays(gl.TRIANGLES,0,3);}
const po=document.getElementById('po'),pt=document.getElementById('pt'),ao=document.getElementById('ao'),
 so=document.getElementById('so'),to=document.getElementById('to'),lo=document.getElementById('lo'),cl=document.getElementById('cl'),mo=document.getElementById('mo');
function lbl(){vpo.textContent=(+po.value).toFixed(2);vpt.textContent=(+pt.value).toFixed(2);vao.textContent=(+ao.value).toFixed(3);
 vso.textContent=(+so.value).toFixed(2);vto.textContent=(+to.value).toFixed(2);vlo.textContent=(+lo.value).toFixed(2);vcl.textContent=(+cl.value).toFixed(2);vmo.textContent=(+mo.value).toFixed(2);}
let raf=0;function schedule(){if(!raf)raf=requestAnimationFrame(()=>{raf=0;draw();});}  // defer heavy draw
[po,pt,ao,so,to,lo,cl,mo].forEach(e=>e.addEventListener('input',()=>{lbl();schedule();}));lbl();
let drag=false,lx=0,ly=0;
cv.onmousedown=e=>{drag=true;lx=e.clientX;ly=e.clientY;};
window.onmouseup=()=>drag=false;
window.onmousemove=e=>{if(!drag)return;yaw+=(e.clientX-lx)*0.01;pitch+=(e.clientY-ly)*0.01;
 pitch=Math.max(-1.55,Math.min(1.55,pitch));lx=e.clientX;ly=e.clientY;schedule();};
cv.onwheel=e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.001);dist=Math.max(1.3,Math.min(6,dist));schedule();};
window.onresize=schedule;draw();
}
</script></body></html>"""


def write_viewer(out, S):
    import base64
    praw = S["p"]; c = S["c_crop"]; st = S["stn_crop"]; tx = S["tx_crop"]; br = S["brain_crop"]
    tgt = np.clip(np.round(S["target_crop"]).astype(int), 0, np.array(praw.shape) - 1)
    dx_mm = S["dx"] * 1e3
    pk = float(praw.max()) or 1.0
    fk = np.unravel_index(int(np.argmax(praw)), praw.shape)          # true global peak
    bone = c > 2000.0
    tissue = "skull" if bone[fk] else ("brain" if br.any() and br[fk] else "coupling")
    # mark the BRAIN peak (skull hotspots are expected + separate); report how far it missed the target
    p_brain = np.where(br, praw, 0.0) if br.any() else praw
    fb = np.unravel_index(int(np.argmax(p_brain)), praw.shape)
    d_err = float(np.linalg.norm((np.array(fb) - tgt) * dx_mm))
    p_tgt = float((praw * st).max()) if st.any() else float(praw[tuple(tgt)])
    peak = f"{(fb[0]+0.5)/praw.shape[0]:.4f},{(fb[1]+0.5)/praw.shape[1]:.4f},{(fb[2]+0.5)/praw.shape[2]:.4f}"
    p = praw
    if (p > 0).mean() < 0.05:                                # sparse legacy 3-plane data -> thicken to sheets
        p = ndi.maximum_filter(p, size=5)
    zf = min(1.0, 144.0 / max(p.shape))                      # texture resolution (higher = crisper anatomy)
    pd = ndi.zoom(p, zf, order=1)
    cd = ndi.zoom(c, zf, order=1)
    sd = ndi.zoom(st.astype(np.float32), zf, order=1) > 0.3
    td = ndi.zoom(tx.astype(np.float32), zf, order=1) > 0.3
    lv = S.get("lens_vol")
    ld = (ndi.zoom(lv.astype(np.float32), zf, order=1) > 0.3) if lv is not None else np.zeros_like(td)
    Nx, Ny, Nz = pd.shape
    R = np.clip(pd / (pd.max() or 1.0), 0, 1)                # pressure 0..1
    G = np.clip((cd - C_WATER) / (C_BONE - C_WATER), 0, 1)   # anatomy ghost (water..bone)
    A = np.where(td, 1.0, np.where(ld, 0.5, 0.0))            # device chan: 1=transducer, 0.5=lens
    rgba = (np.stack([R, G, sd.astype(np.float32), A.astype(np.float32)], -1) * 255).astype(np.uint8)
    flat = np.transpose(rgba, (2, 1, 0, 3)).ravel(order="C")  # x-fastest for glTexImage3D
    b64 = base64.b64encode(flat.tobytes()).decode()
    title = (f"On target {p_tgt/1e3:.0f} kPa · brain peak {d_err:.1f} mm off · "
             f"global {pk/1e3:.0f} kPa ({tissue})")
    html = (VIEWER_HTML.replace("__NX__", str(Nx)).replace("__NY__", str(Ny)).replace("__NZ__", str(Nz))
            .replace("__TITLE__", title).replace("__PEAK__", peak).replace("__DATA__", b64))
    path = os.path.join(out, "pressure_viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"    wrote pressure_viewer.html  ({Nx}x{Ny}x{Nz} volume, {len(b64)//1024} KB) -> open in a browser")


def write_preview_viewer(out, S):
    """Geometry-only 3-D viewer (no pressure) for CONFIRMING placement before the sim:
    same viewer as the results one, but the pressure channel is empty so you inspect the
    anatomy ghost + STN target (green) + transducer (cyan) in 3-D. Uses the full-grid maps."""
    import base64
    c = S["c"]; st = S["stn"]; tx = S["tx_mask"]
    zf = min(1.0, 144.0 / max(c.shape))
    cd = ndi.zoom(c, zf, order=1)
    sd = ndi.zoom(st.astype(np.float32), zf, order=1) > 0.3
    td = ndi.zoom(tx.astype(np.float32), zf, order=1) > 0.3
    Nx, Ny, Nz = cd.shape
    R = np.zeros_like(cd, np.float32)                        # no pressure at preview time
    G = np.clip((cd - C_WATER) / (C_BONE - C_WATER), 0, 1)   # anatomy ghost (water..bone)
    rgba = (np.stack([R, G, sd.astype(np.float32), td.astype(np.float32)], -1) * 255).astype(np.uint8)
    flat = np.transpose(rgba, (2, 1, 0, 3)).ravel(order="C")
    b64 = base64.b64encode(flat.tobytes()).decode()
    title = "Placement preview — anatomy ghost · STN target (green) · transducer (cyan)"
    html = (VIEWER_HTML.replace("__NX__", str(Nx)).replace("__NY__", str(Ny)).replace("__NZ__", str(Nz))
            .replace("__TITLE__", title).replace("__PEAK__", "-1,-1,-1").replace("__DATA__", b64))
    with open(os.path.join(out, "placement_viewer.html"), "w", encoding="utf-8") as f:
        f.write(html)
    log(f"    wrote placement_viewer.html  ({Nx}x{Ny}x{Nz} volume) -> confirm placement in 3-D")


# ── STEP 11: visualise pressure ───────────────────────────────────────────────
def step11_view(data, out, S):
    log("[11] visualising pressure field ...")
    p = S["p"]; V = np.round(S["view"]).astype(int)
    c = S["c_crop"]; br = S["brain_crop"]; st = S["stn_crop"]; tx = S["tx_crop"]
    dx_mm = S["dx"] * 1e3
    pk = float(p.max()) or 1.0
    bone = c > 2000.0                                               # skull/bone (c: brain~1540, bone~3100)
    p_stn = float((p * st).max()) if st.any() else 0.0
    # key landmarks (crop voxel coords)
    tgt = np.round(S["target_crop"]).astype(int)
    txc = np.round(np.argwhere(tx).mean(0)).astype(int) if tx.any() else tgt
    foc = np.array(np.unravel_index(int(np.argmax(p)), p.shape))    # true GLOBAL peak (honest)
    d_mm = float(np.linalg.norm((foc - tgt) * dx_mm))               # targeting error
    # brain-only peak = the physically meaningful focus (skull hotspots are expected + separate)
    p_brain = np.where(br, p, 0.0) if br.any() else p
    focb = np.array(np.unravel_index(int(np.argmax(p_brain)), p.shape))
    pk_brain = float(p_brain.max())
    d_brain = float(np.linalg.norm((focb - tgt) * dx_mm))
    fc = np.clip(foc, 0, np.array(p.shape) - 1)
    tissue = "skull/bone" if bone[tuple(fc)] else ("brain" if br.any() and br[tuple(fc)] else "coupling")

    planes = [(2, "XY", 0, 1), (1, "XZ", 0, 2), (0, "YZ", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    im = None
    for ax, (fx, nm, a0, a1) in zip(axes, planes):
        sl = [slice(None)] * 3; sl[fx] = int(V[fx])
        # grayscale anatomy background (sound speed: water dark -> skull bright)
        ax.imshow(c[tuple(sl)].T, origin="lower", cmap="gray", aspect="equal",
                  vmin=C_WATER, vmax=C_BONE)
        # pressure as a translucent hot overlay, in kPa (hide the near-zero background)
        pl = p[tuple(sl)].T / 1e3
        pm = np.ma.masked_where(pl < 0.05 * pk / 1e3, pl)
        im = ax.imshow(pm, origin="lower", cmap="inferno", alpha=0.8, aspect="equal",
                       vmin=0, vmax=pk / 1e3)
        for m, col in [(br, "deepskyblue"), (st, "red")]:                 # brain + STN outlines
            mm = m[tuple(sl)].T
            if mm.any(): ax.contour(mm, levels=[0.5], colors=[col], linewidths=0.8)
        ax.plot([txc[a0], tgt[a0]], [txc[a1], tgt[a1]], "--", color="cyan", lw=1.0, alpha=0.7)
        ax.plot(txc[a0], txc[a1], "s", color="cyan", ms=7, label="transducer")
        ax.plot(tgt[a0], tgt[a1], "*", color="red", ms=14, label="STN target")
        ax.plot(focb[a0], focb[a1], "D", color="magenta", ms=8, mew=1.6, mfc="none", label="brain peak")
        ax.plot(foc[a0], foc[a1], "x", color="lime", ms=10, mew=2.2, label="global peak")
        ax.set_title(f"{nm} @ {'XYZ'[fx]}={int(V[fx])}"); ax.set_xlabel(f"voxels ({dx_mm:.1f} mm)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.65)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.7, pad=0.02); cb.set_label("pressure (kPa)")
    hit = "at STN" if br.any() and d_brain <= 3.0 else f"{d_brain:.1f} mm from STN"
    fig.suptitle(f"CONFIRM (forward) sim  |  on-target {p_stn/1e3:.0f} kPa  |  "
                 f"brain peak {pk_brain/1e3:.0f} kPa {hit}  |  global {pk/1e3:.0f} kPa in {tissue}",
                 fontsize=12.5)
    fig.savefig(os.path.join(out, "step11_pressure.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"    on-target {p_stn/1e3:.1f} kPa; brain peak {pk_brain/1e3:.1f} kPa {d_brain:.1f} mm from STN; "
        f"global {pk/1e3:.1f} kPa in {tissue} ({d_mm:.1f} mm from STN)")
    write_viewer(out, S)
    return S


def main():
    global FREQ_HZ, TX_DIAM_MM, P0_PA, N_CYCLES, C_LENS      # user adjustment options override these
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="folder with the .nii and .stl files")
    ap.add_argument("--out",  default="./pipeline_out")
    ap.add_argument("--dx-sim", type=float, default=0.3, help="sim voxel size (mm)")
    ap.add_argument("--run-sim", action="store_true", help="run k-Wave (needs GPU + kwave)")
    ap.add_argument("--replot", action="store_true",
                    help="just rebuild the step-11 figure + viewer from saved npz (no GPU, seconds)")
    ap.add_argument("--sensor-stride", type=int, default=0,
                    help="manual voxel stride for the recorded pressure lattice (0=auto-fit VRAM)")
    ap.add_argument("--lease-gb", type=float, default=16.0,
                    help="GPU memory you leased (GB); the recorded lattice is auto-sized to fit it")
    ap.add_argument("--target-label", type=int, default=STN_LABEL, help="atlas label number to target")
    ap.add_argument("--atlas", default="subcortical", choices=["subcortical", "cortical"])
    ap.add_argument("--target-name", default="", help="human name of the target (for figures/logs)")
    ap.add_argument("--target-offset", default="0,0,0",
                    help="nudge the focus off the atlas centroid, 'x,y,z' mm in scan/world frame")
    ap.add_argument("--freq-mhz",      type=float, default=FREQ_HZ / 1e6, help="transducer frequency (MHz)")
    ap.add_argument("--diam-mm",       type=float, default=TX_DIAM_MM,    help="transducer aperture diameter (mm)")
    ap.add_argument("--pressure-kpa",  type=float, default=P0_PA / 1e3,   help="source surface pressure (kPa)")
    ap.add_argument("--cycles",        type=int,   default=N_CYCLES,      help="tone-burst cycles")
    ap.add_argument("--lens-c",        type=float, default=C_LENS,        help="lens material sound speed (m/s)")
    ap.add_argument("--preview", action="store_true",
                    help="run geometry only (load + target + transducer confirmation), no GPU")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    FREQ_HZ = args.freq_mhz * 1e6; TX_DIAM_MM = float(args.diam_mm); P0_PA = args.pressure_kpa * 1e3
    N_CYCLES = int(args.cycles); C_LENS = float(args.lens_c)      # apply the user's adjustment options

    if args.replot:                                     # regenerate the figure + viewer only
        g = np.load(os.path.join(args.out, "step09_sim_input.npz"))
        d = np.load(os.path.join(args.out, "step10_pressure.npz"))
        if "p_coarse" in d:                             # strided volume -> upsample to full res
            shp = tuple(int(x) for x in d["shape"])
            p = ndi.zoom(d["p_coarse"], np.array(shp) / np.array(d["p_coarse"].shape), order=1).astype(np.float32)
        else:
            p = d["p_max"]                              # legacy 3-plane data
        S = dict(c_crop=g["c_crop"], brain_crop=g["brain_crop"].astype(bool),
                 stn_crop=g["stn_crop"].astype(bool), tx_crop=g["tx_crop"].astype(bool),
                 target_crop=g["target"], p=p, view=d["view"], dx=float(d["dx"]))
        if "lens" in d.files and d["lens"].any():
            S["lens_vol"] = d["lens"].astype(bool)             # show the lens in the viewer on --replot
        else:                                                  # older result: rebuild lens from saved delays (no GPU)
            b_npz = os.path.join(args.out, "step09b_delays.npz")
            if os.path.exists(b_npz):
                bd = np.load(b_npz)
                S["ap_coords"] = bd["ap_coords"]; S["delays_sec"] = bd["delays_sec"]
                build_lens_from_delays(args.data, args.out, S)
        step11_view(args.data, args.out, S)
        log(f"all outputs in {args.out}"); return

    S = {"sensor_stride": args.sensor_stride, "lease_gb": args.lease_gb,
         "target_label": args.target_label, "target_name": args.target_name,
         "target_offset": [float(x) for x in args.target_offset.split(",")],
         "atlas": "cort" if args.atlas == "cortical" else "subcort"}
    step1_load(args.data, args.out, S)
    step2_target(args.data, args.out, S)
    step3_entry(args.data, args.out, S)
    step4_register(args.data, args.out, S)
    step5_cut(args.data, args.out, S)
    step6_acoustics(args.data, args.out, S)
    step7_transducer(args.data, args.out, S)
    step7b_lens(args.data, args.out, S)
    if args.preview:                                    # geometry confirmation only (no GPU)
        log(f"preview done — confirm target + transducer placement; outputs in {args.out}")
        return
    step8_delays(args.data, args.out, S)
    step9_crop(args.data, args.out, S, args.dx_sim)
    if args.run_sim:
        step9b_backward(args.data, args.out, S)      # 1) backward: design delays through the real skull
        build_lens_from_delays(args.data, args.out, S)  # freeform lens from those delays
        step10_sim(args.data, args.out, S)           # 2) forward: confirm the true focus
        step11_view(args.data, args.out, S)
    else:
        log("geometry done. Re-run with --run-sim on a GPU node to simulate.")
    log(f"all outputs in {args.out}")


if __name__ == "__main__":
    main()
