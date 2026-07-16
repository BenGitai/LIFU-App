# LIFU Transcranial Focusing Planner

Plan **low-intensity focused ultrasound (LIFU)** through the skull to a brain target, from a
subject's own CT/MRI + a subcortical/cortical atlas and a transducer-adapter STL. The tool
places and aims a single-element transducer, designs a **3D-printable focusing lens** that
corrects skull aberration, runs a GPU **k-Wave** acoustic simulation, and reports **honestly**
where the pressure actually focuses — as 2-D charts and an interactive **3-D viewer**.

It runs as a small local **web app** (point it at a folder → pick a target → confirm → run →
results) or from the **command line**.

> **This is a planning framework, not a dataset.** No patient scans are included. You bring your
> own co-registered CT + MRI, brain/skull masks, an atlas + label CSV, and the device STLs (see
> [Input files](#input-files-bring-your-own)). It is research/educational software — **not a
> medical device**; do not use it for clinical decisions.

---

## How it focuses (the method)

A single-element transducer can only emit **one waveform**, so all focusing is done by the lens.
The planner does this honestly with **two simulations**:

1. **Backward (design) sim** — a virtual point source at the target radiates *through the real CT
   skull*; the pressure arriving on the transducer aperture is recorded. By acoustic reciprocity,
   the phase-conjugate of what arrives is what the aperture must emit to refocus there. Per-aperture
   arrival times are found by a matched filter (robust to skull reverberation).
2. **Lens design** — those delays become a **freeform, Fresnel-wrapped** lens (`lens.stl`): each
   aperture point is a resin column whose length sets its delay, wrapped modulo one acoustic period
   so the lens stays thin (a printable acoustic hologram) instead of a many-cm spike.
3. **Forward (confirm) sim** — the transducer + that lens is driven (near-CW) through the real
   skull, and the **true global pressure peak** is measured. Reporting is decomposed into
   *on-target* pressure, the *brain* peak (and how far it missed), and the *global* peak with its
   tissue (skull hotspots are expected and flagged separately) — no masking to the target.

---

## Requirements

- **Python 3.10** (via conda/miniconda recommended; scientific wheels are spotty on 3.13/3.14).
- **Geometry, figures, lens design, and the whole app UI run on CPU** — no GPU needed.
- The **simulation step** needs an **NVIDIA GPU** + drivers and a **CuPy** build matching your CUDA.
  ~8 GB VRAM is enough at `--dx-sim 0.4`; finer grids (`0.3`) want ~16–20 GB.

## Install

```bash
conda env create -f environment.yml
conda activate lifu
```

Then, **for the GPU simulation only**, install the CuPy build for your CUDA (see `nvidia-smi`):

```bash
pip install "cupy-cuda12x[ctk]"      # CUDA 12.x / 13.x drivers
# or
pip install "cupy-cuda11x[ctk]"      # CUDA 11.x drivers
```

The `[ctk]` extra pulls the NVRTC compiler + headers CuPy needs to build kernels.

(Prefer pip? `pip install -r requirements.txt`, then the CuPy line above.)

## Run — web app

```bash
python app.py            # opens http://localhost:5000
```
(or double-click `run_app.bat` on Windows).

1. **Data folder** — paste the path to a folder holding your scans + device STLs. Files are
   auto-detected. Outputs go to `<folder>/pipeline_out`.
2. **Choose target** — atlas + target by **name or label number** (the label CSV is loaded so you
   can type e.g. `subthalamic`). Optionally **nudge** the focus by X/Y/Z mm, or open **Advanced**
   for frequency, aperture, pressure, cycles, and lens material.
3. **Confirm** — a fast, no-GPU geometry pass shows the target and transducer placement, including
   an interactive **3-D placement viewer**, before you commit to a run.
4. **Run** — the GPU sim streams its log.
5. **Results** — pressure chart, embedded **3-D pressure viewer** (with the lens shown), and
   downloads for `lens.stl` and `transducer_spec.txt`.

## Run — command line

`app.py` just drives `lifu_pipeline.py`, which you can also run directly:

```bash
python lifu_pipeline.py --data <folder> --out <folder>/pipeline_out \
    --atlas subcortical --target-label 91 --target-name STN \
    --target-offset 0,0,0 --freq-mhz 1.0 --diam-mm 19 --pressure-kpa 100 \
    --cycles 30 --lens-c 2500 --dx-sim 0.4 --run-sim --lease-gb 8
```

| Flag | Meaning |
|------|---------|
| `--preview` | geometry only (target + transducer + placement viewer), **no GPU** |
| `--replot` | rebuild the figure + 3-D viewer from saved results, no GPU (seconds) |
| `--run-sim` | run the backward + forward k-Wave sims (needs GPU + CuPy) |
| `--dx-sim` | simulation voxel size in mm (0.4 fits ~8 GB; 0.3 is sharper, ~16–20 GB) |
| `--lease-gb` | GPU memory to size the pressure recording to |
| `--target-label` / `--atlas` / `--target-name` | target by atlas label / `subcortical`\|`cortical` / display name |
| `--target-offset x,y,z` | nudge the focus off the atlas centroid (mm, scan frame) |
| `--freq-mhz` `--diam-mm` `--pressure-kpa` `--cycles` `--lens-c` | transducer + lens parameters |

## Input files (bring your own)

Drop these in one folder; they're auto-detected by name/content. `.nii` and `.nii.gz` both work.

| Role | How it's detected |
|------|-------------------|
| CT (HU), co-registered to the MRI | the `.nii` with the most-negative values |
| MRI | remaining `.nii` (name containing `mri` preferred) |
| Skull mask | name contains `skull` |
| Brain mask | name contains `brain` |
| Subcortical atlas | name contains `subcort` |
| Cortical atlas | name contains `cort` (not `subcort`) |
| Adapter STL | name contains `adapter` |
| Base STL | name contains `base` |
| Label CSVs | `*subcort*label*.csv`, `*cort*label*.csv` (for target-by-name) |

The device STLs must be in the **same physical (scan-world) frame** as the NIfTIs — the pipeline
voxelizes them straight through the NIfTI affine (no registration). Export them once from your CAD
with `convert_step_to_stl.py` (needs `cadquery`/`OCP`, run locally where those are available).

## Outputs (`pipeline_out/`)

| File | What |
|------|------|
| `step01`–`step09` PNGs | per-step geometry/anatomy figures (loading, target, craniotomy, sim grid) |
| `placement_viewer.html` | interactive 3-D placement check (anatomy + target + transducer) |
| `lens.stl` | the printable Fresnel-wrapped focusing lens |
| `transducer_spec.txt` | transducer pose (world mm, aim, az/el), focal distance, lens spec |
| `step11_pressure.png` | pressure field on anatomy with on-target / brain-peak / global-peak |
| `pressure_viewer.html` | interactive 3-D pressure viewer (pressure, STN, transducer, lens) |
| `step09b_delays.npz`, `step10_pressure.npz` | saved delays + pressure for `--replot` |

## Running the sim on a remote GPU box

No GPU locally? Copy the code + your data to a machine with an NVIDIA GPU and run there.
`run_brains.sh` is an example runner for an unscheduled SSH GPU box (edit the env activation for
your setup); for a long run, launch it under `tmux`/`nohup` so it survives an SSH drop.

## Troubleshooting

- **`No module named 'networkx'`** during lens export — `pip install networkx` (or `conda install
  networkx`). The lens still exports without it; slicers repair the normals.
- **CuPy: "Failed to find CUDA headers"** — install the `[ctk]` extra: `pip install "cupy-cuda12x[ctk]"`.
- **CuPy: "No matching distribution"** — your Python is too new; use **Python 3.10**.
- **Out of VRAM in the sim** — use a larger `--dx-sim` (e.g. `0.4`), lower `--lease-gb`, or a
  bigger-memory GPU. The recording lattice auto-sizes to the memory you give it.
- **App says "no label CSV found"** — put a `*subcort*label*.csv` / `*cort*label*.csv` in the folder;
  targeting by label number still works without it.

## License

MIT — see [LICENSE](LICENSE). Provided for research and educational use. **Not a medical device;
do not use for clinical decisions.**
