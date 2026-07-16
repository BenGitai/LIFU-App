"""
convert_step_to_stl.py  --  run LOCALLY (Windows) once, where cadquery/OCP work.

Converts the device STEP parts to STL so the Brains cluster pipeline can load them
with trimesh alone (no cadquery/OCP needed on the cluster).

    conda activate <env with cadquery>
    python convert_step_to_stl.py

Writes inner_adapter.stl and outer_base_craniotomy.stl next to the .STEP files.
"""
import os
import numpy as np
import trimesh
from cadquery import Shape as CQShape
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect  import IFSelect_RetDone

HERE  = os.path.dirname(os.path.abspath(__file__))
PARTS = ["inner_adapter", "outer_base_craniotomy", "skull_craniotomy", "brain"]


def occ_read_step(path):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"OCC ReadFile failed: {path}")
    n_roots = reader.NbRootsForTransfer()
    for i in range(n_roots):
        reader.TransferRoot(i + 1)
    n = reader.NbShapes()
    if n == 0:
        raise IOError(f"No shapes in {path}")
    if n == 1:
        return reader.Shape(1)
    builder = BRep_Builder(); comp = TopoDS_Compound(); builder.MakeCompound(comp)
    for i in range(n):
        builder.Add(comp, reader.Shape(i + 1))
    return comp


def occ_to_mesh(shape, tol_mm=0.05, ang=0.3):
    verts, faces = CQShape(shape).tessellate(tol_mm, ang)
    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=True)


if __name__ == "__main__":
    for name in PARTS:
        step = os.path.join(HERE, name + ".STEP")
        out  = os.path.join(HERE, name + ".stl")
        if not os.path.exists(step):
            print(f"{name}: SKIP (no {name}.STEP)"); continue
        try:
            m = occ_to_mesh(occ_read_step(step))
            m.export(out)
            bb = m.bounds
            print(f"{name}: V={m.volume/1e3:.2f} cm^3  bounds {np.round(bb[0],1)}..{np.round(bb[1],1)} mm "
                  f"-> {os.path.basename(out)}")
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
    print("done. Ship the .stl files to Brains with the pipeline.")
