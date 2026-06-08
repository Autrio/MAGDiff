# add near the top of the file
import multiprocessing as mp
import traceback
import trimesh
import coacd as acd
import os
from pathlib import Path
import numpy as np

def _acd_worker(object_file: str, scale: float, user_scale: float,
                out_dir: str, max_convex_hull: int, q):
    """
    Runs CoACD in a separate process so all native memory is released
    when the worker exits.
    """
    try:
        mesh = trimesh.load(object_file, force="mesh")
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale * user_scale)

        acd_mesh = acd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        parts = acd.run_coacd(acd_mesh, max_convex_hull=max_convex_hull)

        os.makedirs(out_dir, exist_ok=True)
        for i, (vs, fs) in enumerate(parts):
            part_mesh = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
            part_mesh.export(str(Path(out_dir) / f"scaled_part{i}.obj"))

        q.put(("ok", len(parts)))

    except Exception:
        q.put(("err", traceback.format_exc()))