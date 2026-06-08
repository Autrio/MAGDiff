import gc
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import glob
import os

import h5py
import mujoco
import mujoco_warp as MW
import numpy as np
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from magdiff.controller.render import WarpGridVideoRenderer, GridLayout
from magdiff.paths import GRASP_DIR, MESH_DIR, GRIPPER_MESH_PATH, SCALES_JSON, WORLD_BASE_FILE

import multiprocessing as mp
import traceback
from magdiff.grasp.acd import _acd_worker



@dataclass
class WorldSample:
    grasp_file: str
    object_file: str
    scale: float
    grasp: np.ndarray


class ObjectLoader:
    def __init__(
        self,
        world_base_file: str = WORLD_BASE_FILE,
        grasp_dir: str = GRASP_DIR,
        mesh_dir: str = MESH_DIR,
        gripper_mesh_path: str = GRIPPER_MESH_PATH,
        scales_json: str = SCALES_JSON,
        user_scale: float = 0.5,
        nworld: int = 4,
        movable_object: bool = True,
    ):
        self.grasp_dir = Path(grasp_dir)
        self.mesh_dir = Path(mesh_dir)
        self.gripper_mesh_path = Path(gripper_mesh_path)
        self.scales_json = Path(scales_json)
        self.world_base_file = str(Path(world_base_file))
        self.user_scale = float(user_scale)
        self.nworld = int(nworld)
        self.movable_object = bool(movable_object)

        with open(self.scales_json, "r") as f:
            self.scales = json.load(f)

        self.grasp_files = [str(p) for p in self.grasp_dir.glob("*")]
        self.obj_files = [str(p) for p in self.mesh_dir.glob("*.obj")]

        if not self.grasp_files:
            raise FileNotFoundError(f"No grasp files found in {self.grasp_dir}")
        if not self.obj_files:
            raise FileNotFoundError(f"No object meshes found in {self.mesh_dir}")

        self._tmpdir = tempfile.TemporaryDirectory()
        self._gripper_temp = str(Path(self._tmpdir.name) / "gripper_temp.obj")
        self._prepare_gripper_mesh()

        self.R_conv = np.array(
            [[1, 0, 0],
             [0, 0, 1],
             [0, -1, 0]],
            dtype=np.float64,
        )
        self.rz90 = R.from_euler("z", 90, degrees=True).as_matrix()
        self.mj_frame_correction = np.eye(4, dtype=np.float64)
        self.mj_frame_correction[:3, :3] = self.R_conv @ self.rz90

        self.base_mj_model = None
        self.warp_model = None
        self.warp_data = None

        self._variant_cache = {}
        self._obj_mesh_temp = {}
        self.base_mesh_name = None

        self.pos_mesh_dict = {}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _prepare_gripper_mesh(self):
        mesh = trimesh.load(self.gripper_mesh_path, force="mesh")
        mesh.apply_translation(-mesh.centroid)
        mesh_rot = R.from_euler("xz", [90, 90], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = mesh_rot
        mesh.apply_transform(T)
        mesh.export(self._gripper_temp)

    def _load_sample(self, idx: int, gr_idx:int) -> WorldSample:
        grasp_file = self.grasp_files[idx]
        base_name = Path(grasp_file).stem
        object_file = str(self.mesh_dir / f"{base_name}.obj")
        if object_file not in self.obj_files:
            raise FileNotFoundError(f"Missing object mesh for {base_name}: {object_file}")

        scale = float(self.scales[base_name])

        print(f"Loading grasp file: {grasp_file} at index {idx}")
        print(f"Object file: {object_file} at index {self.obj_files.index(object_file)}")

        with h5py.File(grasp_file, "r") as f:
            grasps = f["grasps/grasps"][:]
            grasp = grasps[-1] if gr_idx is None else grasps[gr_idx]

        return WorldSample(
            grasp_file=grasp_file,
            object_file=object_file,
            scale=scale,
            grasp=grasp,
        )

    def _export_scaled_mesh(self, object_file: str, scale: float, max_convex_hull=5):
        key = (object_file, float(scale))
        if key in self._obj_mesh_temp:
            return self._obj_mesh_temp[key]

        # Load once in the main process only to get metadata.
        mesh = trimesh.load(object_file, force="mesh")
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale * self.user_scale)

        mesh_name = Path(object_file).stem
        pos = mesh.centroid.copy()
        bottom = float(mesh.bounds[0][2])

        out_dir = Path(object_file).parent / f"ACD/{mesh_name}"
        num_parts = len(glob.glob(str(out_dir / "scaled_part*.obj")))

        # Free the main-process mesh before launching CoACD.
        del mesh
        gc.collect()

        if num_parts == 0:
            ctx = mp.get_context("fork")
            q = ctx.Queue(maxsize=1)
            p = ctx.Process(
                target=_acd_worker,
                args=(
                    object_file,
                    scale,
                    self.user_scale,
                    str(out_dir),
                    max_convex_hull,
                    q,
                ),
            )
            p.start()
            status, payload = q.get()
            p.join()

            if status != "ok":
                raise RuntimeError(f"CoACD failed for {object_file}:\n{payload}")

            num_parts = int(payload)

        self._obj_mesh_temp[key] = (mesh_name, pos, bottom, num_parts)
        return mesh_name, pos, bottom, num_parts

    def _grasp_to_mocap_poses(self, grasp: np.ndarray, bottom: float):
        grasp1 = grasp[0] @ self.mj_frame_correction
        grasp2 = grasp[1] @ self.mj_frame_correction

        g1_pos = grasp1[:3, 3].copy() * self.user_scale
        g2_pos = grasp2[:3, 3].copy() * self.user_scale
        g1_pos[2] -= bottom
        g2_pos[2] -= bottom

        g1_quat = R.from_matrix(grasp1[:3, :3]).as_quat(scalar_first=True)
        g2_quat = R.from_matrix(grasp2[:3, :3]).as_quat(scalar_first=True)

        return (
            g1_pos.astype(np.float32),
            g1_quat.astype(np.float32),
            g2_pos.astype(np.float32),
            g2_quat.astype(np.float32),
        )

    def _sample_worlds(self, indices=None, gr_indices=None, randomize=True, seed=None):
        rng = np.random.default_rng(seed)

        if indices is None:
            if randomize:
                indices = rng.integers(0, len(self.grasp_files), size=self.nworld).tolist()
            else:
                indices = list(range(self.nworld))
        if gr_indices is None:
            if randomize:
                gr_indices = rng.integers(0, 1000, size=self.nworld).tolist()
            else:
                gr_indices = [None] * self.nworld

        if len(indices) != self.nworld:
            raise ValueError(f"Expected {self.nworld} indices, got {len(indices)}")

        samples = [self._load_sample(idx, gr_idx) for idx, gr_idx in zip(indices, gr_indices)]
        selected_object_files = list(dict.fromkeys(sample.object_file for sample in samples))
        return samples, selected_object_files

    def _build_super_spec(self, selected_object_files,max_convex_hull=50):
        spec = mujoco.MjSpec().from_file(self.world_base_file)
        max_parts = 0

        for obj_file in tqdm(selected_object_files, desc="Processing sampled meshes"):
            base = Path(obj_file).stem
            scale = float(self.scales.get(base, 0.2))
            mesh_name, pos, bottom, num_parts = self._export_scaled_mesh(obj_file, scale,max_convex_hull=max_convex_hull)
            assert num_parts > 0, f"No parts found for {obj_file} after ACD processing"
            assert mesh_name == base, f"Expected mesh name {base}, got {mesh_name}"
            self.pos_mesh_dict[base] = (pos, bottom)
            for i in range(num_parts):
                part_path = str(Path(obj_file).parent / f"ACD/{base}/scaled_part{i}.obj")
                mesh = spec.add_mesh()
                mesh.name = base + f"_scaled_part{i}"
                mesh.file = part_path
                
            if num_parts > max_parts:
                max_parts = num_parts
                self.base_mesh_name = base
                
        print(f"Max parts across all meshes: {max_parts}")

        gripper_mesh = spec.add_mesh()
        gripper_mesh.name = "gripper_mesh"
        gripper_mesh.file = self._gripper_temp

        obj_body = spec.worldbody.add_body()
        obj_body.name = "object"
        obj_body.pos = [0, 0, 0]
        obj_body.add_freejoint() if self.movable_object else None
        
        for i in range(max_parts):
            obj_geom = obj_body.add_geom()
            obj_geom.type = mujoco.mjtGeom.mjGEOM_MESH
            obj_geom.name = f"object_geom_part{i}"
            obj_geom.meshname = self.base_mesh_name + f"_scaled_part{i}"
            
        g1 = spec.worldbody.add_body()
        g1.name = "gripper1"
        g1.mocap = True
        g1_geom = g1.add_geom()
        g1_geom.name = "gripper1_geom"
        g1_geom.type = mujoco.mjtGeom.mjGEOM_MESH
        g1_geom.meshname = "gripper_mesh"
        g1_geom.rgba = [1, 0, 0, 1]
        g1_geom.contype = 0
        g1_geom.conaffinity = 0

        g2 = spec.worldbody.add_body()
        g2.name = "gripper2"
        g2.mocap = True
        g2_geom = g2.add_geom()
        g2_geom.name = "gripper2_geom"
        g2_geom.type = mujoco.mjtGeom.mjGEOM_MESH
        g2_geom.meshname = "gripper_mesh"
        g2_geom.rgba = [0, 1, 0, 1]
        g2_geom.contype = 0
        g2_geom.conaffinity = 0

        self.base_mj_model = spec.compile()
        return spec, self.base_mj_model

    def _compile_variant(self, spec, mesh_name):
        """Compile a variant where the object geom points to a specific mesh."""
        obj_geoms_old = []
        geom_ids = []
        for body in spec.worldbody.bodies:
            if body.name == "object":
                body.pos = [
                    self.pos_mesh_dict[mesh_name][0][0],
                    self.pos_mesh_dict[mesh_name][0][1],
                    -self.pos_mesh_dict[mesh_name][1] + 0.03,
                ]
                #! HOLD UP
                for g in body.geoms:
                    if "object_geom" in g.name:
                        obj_geoms_old.append(g)
        if not obj_geoms_old:
            raise RuntimeError("Could not find object_geom in spec")

        for body in spec.worldbody.bodies:
            for g in body.geoms:
                if "object_geom" in g.name:
                    part_id = int(g.name.split("part")[-1])
                    if mesh_name + f"_scaled_part{part_id}" in [m.name for m in spec.meshes]:
                        g.meshname = mesh_name + f"_scaled_part{part_id}"
                        geom_ids.append(g.id)
                    else:
                        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        g.size = [0.000001,0,0]
                        g.contype = 0
                        g.conaffinity = 0
                        g.mass=0
                                  
                
        model = spec.compile()
        
        #restore
        for body in spec.worldbody.bodies:
            if body.name == "object":
                for i,g in enumerate(body.geoms):
                    g.meshname = self.base_mesh_name + f"_scaled_part{i}"
                    g.type = mujoco.mjtGeom.mjGEOM_MESH
                    g.contype = 1
                    g.conaffinity = 1
                    
            
        return model, geom_ids

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, indices=None, gr_indices=None, randomize=True, seed=None, verbose=False):
        """
        Sample worlds, compile per-mesh variants one at a time, and upload
        everything to the GPU.

        Memory strategy
        ---------------
        Peak CPU memory = base_model + ONE variant model at a time.

        Previously all N_unique variant models lived simultaneously, giving
        peak = (1 + N_unique) × model_size.  Now it is always 2 × model_size
        regardless of how many unique objects are in the batch.  Numpy
        staging buffers are released immediately after the warp upload.
        """
        samples, selected_object_files = self._sample_worlds(
            indices=indices,
            gr_indices=gr_indices,
            randomize=randomize,
            seed=seed,
        )

        spec, base_model = self._build_super_spec(selected_object_files)

        m = MW.put_model(base_model)
        d = MW.make_data(base_model, nworld=self.nworld, nconmax=1000, njmax=1000)

        nw = self.nworld
        ng = base_model.ngeom
        nb = base_model.nbody

        # Helper: tile a 1-D or N-D array along a new leading world axis.
        def _tile(arr: np.ndarray) -> np.ndarray:
            return np.tile(arr[np.newaxis], (nw,) + (1,) * arr.ndim)

        # Pre-fill staging buffers from the base model.  Every world starts
        # with valid base data; the per-world loop below only overwrites what
        # actually differs (i.e. the object geom / body).
        geom_dataid     = _tile(base_model.geom_dataid)                    # (nw, ng)
        geom_size       = _tile(base_model.geom_size)                      # (nw, ng, 3)
        geom_rbound     = _tile(base_model.geom_rbound)                    # (nw, ng)
        geom_aabb       = _tile(base_model.geom_aabb.reshape(ng, 2, 3))    # (nw, ng, 2, 3)
        geom_pos        = _tile(base_model.geom_pos)                       # (nw, ng, 3)
        geom_quat       = _tile(base_model.geom_quat)                      # (nw, ng, 4)
        body_mass       = _tile(base_model.body_mass)                      # (nw, nb)
        body_subtreemass= _tile(base_model.body_subtreemass)               # (nw, nb)
        body_inertia    = _tile(base_model.body_inertia)                   # (nw, nb, 3)
        body_invweight0 = _tile(base_model.body_invweight0)                # (nw, nb, 2)
        body_ipos       = _tile(base_model.body_ipos)                      # (nw, nb, 3)
        body_iquat      = _tile(base_model.body_iquat)                     # (nw, nb, 4)
        body_pos        = _tile(base_model.body_pos)                       # (nw, nb, 3)

        mocap_pos  = d.mocap_pos.numpy().copy()
        mocap_quat = d.mocap_quat.numpy().copy()
        qpos       = d.qpos.numpy().copy()

        # Resolve body / mocap IDs once up front.
        object_body_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "object")
        geom_start = base_model.body_geomadr[object_body_id]
        geom_count = base_model.body_geomnum[object_body_id]
        geom_slice = slice(
            geom_start,
            geom_start + geom_count
        )
        g1_body_id     = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "gripper1")
        g2_body_id     = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "gripper2")
        mocap1 = int(base_model.body(g1_body_id).mocapid[0])
        mocap2 = int(base_model.body(g2_body_id).mocapid[0])
        nq     = base_model.nq  # capture before any del

        # ---- KEY CHANGE: group worlds by mesh, compile one variant at a time ----
        #
        # Old code kept variant_models = {mesh: compiled_model} for ALL unique
        # meshes simultaneously.  Here we compile, copy, and DELETE each variant
        # before moving to the next one.
        #
        # Peak memory:  base_model  +  ONE variant  (not 1 + N_unique variants).

        mesh_to_worlds: dict[str, list[int]] = defaultdict(list)
        for w, sample in enumerate(samples):
            mesh_to_worlds[Path(sample.object_file).stem].append(w)

        for mesh_name, world_indices in tqdm(mesh_to_worlds.items(), desc="Building variants"):
            variant, geom_ids = self._compile_variant(spec, mesh_name)

            for w in world_indices:
                sample = samples[w]
                bottom = self.pos_mesh_dict[str(Path(sample.object_file).stem)][-1]
                g1_pos, g1_quat, g2_pos, g2_quat = self._grasp_to_mocap_poses(
                    sample.grasp, bottom
                )

                # Geometry / inertia that differ per mesh.
                geom_dataid[w, geom_slice] = -1
                geom_dataid[w, geom_ids] = variant.geom_dataid[geom_ids]
                geom_size[w]        = variant.geom_size
                geom_rbound[w]      = variant.geom_rbound
                geom_aabb[w]        = variant.geom_aabb.reshape(ng, 2, 3)
                geom_pos[w]         = variant.geom_pos
                geom_quat[w]        = variant.geom_quat
                body_mass[w]        = variant.body_mass
                body_subtreemass[w] = variant.body_subtreemass
                body_inertia[w]     = variant.body_inertia
                body_invweight0[w]  = variant.body_invweight0
                body_ipos[w]        = variant.body_ipos
                body_iquat[w]       = variant.body_iquat
                body_pos[w]         = variant.body_pos

                # Per-world gripper poses (live in data, not model).
                mocap_pos[w, mocap1]  = g1_pos
                mocap_pos[w, mocap2]  = g2_pos
                mocap_quat[w, mocap1] = g1_quat
                mocap_quat[w, mocap2] = g2_quat

                qpos[w, :nq] = variant.qpos0
                qpos[w, :21] = np.array(
                    [-0.08, -0.8, 0.06, 0.92, -7.8e-05, 1.7, -0.036, 0.003, 0.003,
                     0.0032, 0.003, 0.003, 0.0032, -0.2, 0.18, 1.7, 0.015, 1.6,
                     -0.19, -0.0002, 0.0002],
                    dtype=np.float64,
                )

                tqdm.write(
                    f"  World {w}: mesh={mesh_name}, bottom={bottom:.4f}, "
                    f"obj_pos={self.pos_mesh_dict[mesh_name][0]}"
                ) if verbose else None

            # ---- Free the compiled variant immediately ----
            del variant
            gc.collect()  # return pages to the OS now, not at end of build()

        # ---- Upload all staging buffers to GPU in one pass ----
        m.geom_dataid      = wp.array(geom_dataid,      dtype=wp.int32)
        m.geom_size        = wp.array(geom_size,        dtype=wp.vec3)
        m.geom_rbound      = wp.array(geom_rbound,      dtype=wp.float32)
        m.geom_aabb        = wp.array(geom_aabb,        dtype=wp.vec3)
        m.geom_pos         = wp.array(geom_pos,         dtype=wp.vec3)
        m.geom_quat        = wp.array(geom_quat,        dtype=wp.quat)
        m.body_mass        = wp.array(body_mass,        dtype=wp.float32)
        m.body_subtreemass = wp.array(body_subtreemass, dtype=wp.float32)
        m.body_inertia     = wp.array(body_inertia,     dtype=wp.vec3)
        m.body_invweight0  = wp.array(body_invweight0,  dtype=wp.vec2)
        m.body_ipos        = wp.array(body_ipos,        dtype=wp.vec3)
        m.body_iquat       = wp.array(body_iquat,       dtype=wp.quat)
        m.body_pos         = wp.array(body_pos,         dtype=wp.vec3)

        d.mocap_pos  = wp.array(mocap_pos,  dtype=wp.vec3)
        d.mocap_quat = wp.array(mocap_quat, dtype=wp.quat)
        d.qpos       = wp.array(qpos,       dtype=wp.float32)

        # ---- Free numpy staging buffers; data now lives on the GPU ----
        del (
            geom_dataid, geom_size, geom_rbound, geom_aabb,
            geom_pos, geom_quat,
            body_mass, body_subtreemass, body_inertia,
            body_invweight0, body_ipos, body_iquat, body_pos,
            mocap_pos, mocap_quat, qpos,
        )
        gc.collect()

        self.warp_model = m
        self.warp_data = d
        return base_model, m, d


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = ObjectLoader(nworld=1)
    base_model, m, d = loader.build(indices=[42], randomize=False)

    rec = WarpGridVideoRenderer(
        base_model,
        m,
        d,
        nworld=1,
        layout=GridLayout(1, 1),
    )
    rec.open("test_output.mp4")

    for _ in range(100):
        MW.step(m, d)
        rec.capture()

    rec.close()