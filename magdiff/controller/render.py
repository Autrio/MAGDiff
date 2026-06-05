"""MJWarp grid video renderer.

Renders a subset of worlds (≤ total simulated) as a grid and streams
frames directly to disk with low CPU memory usage.

Expected usage::

    # Render all 6 worlds (auto layout → 2×3)
    recorder = WarpGridVideoRenderer(mj_model, osc.model, osc.data, nworld=6)

    # Render only the first 4 of 6 worlds (auto layout → 2×2)
    recorder = WarpGridVideoRenderer(mj_model, osc.model, osc.data,
                                     nworld=6, render_worlds=4)

    # Render specific worlds by index (auto layout → 1×3)
    recorder = WarpGridVideoRenderer(mj_model, osc.model, osc.data,
                                     nworld=6, render_worlds=[0, 2, 4])

    for _ in range(steps):
        osc.step()
        MW.step(osc.model, osc.data)
        recorder.capture()
    recorder.close()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import imageio.v2 as imageio
import mujoco_warp as MW
import warp as wp


# Helpers

def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to uint8 safely."""
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0 + 0.5).astype(np.uint8)


@dataclass
class GridLayout:
    rows: int = 2
    cols: int = 2

    @property
    def capacity(self) -> int:
        return self.rows * self.cols


def _auto_layout(n: int) -> GridLayout:
    """Return a near-square GridLayout for *n* panels.

    The grid may have up to ``cols - 1`` empty (black) trailing cells when
    *n* is not a perfect rectangle.  Examples::

        n=1 → 1×1   n=2 → 1×2   n=3 → 2×2 (+1 empty)
        n=4 → 2×2   n=5 → 2×3 (+1 empty)   n=6 → 2×3
        n=7 → 3×3 (+2 empty)   n=8 → 2×4   n=9 → 3×3
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return GridLayout(rows=rows, cols=cols)


# Renderer

class WarpGridVideoRenderer:
    """Render a subset of worlds (≤ total simulated) into a single grid video.

    Parameters
    ----------
    mj_model
        MuJoCo model (``mujoco.MjModel``).
    warp_model, warp_data
        Warp-side model/data from your simulation.
    nworld
        **Total** number of worlds in the warp simulation.  The render
        context and GPU buffer are always allocated for all *nworld* worlds.
    render_worlds
        Which worlds to include in the output video.  Three forms:

        ``None`` (default)
            Render all *nworld* worlds.
        ``int``
            Render the first *N* worlds (indices ``0 … N-1``).
        ``Sequence[int]``
            Render exactly those world indices (must be in ``[0, nworld)``).

        The number of selected worlds drives the auto layout and output
        resolution, not *nworld*.
    camera_index
        Camera to render from (index into ``rc.cam_res``).
    width, height
        Per-tile resolution (pixels).
    layout
        Explicit ``GridLayout``.  When ``None`` (default), a near-square
        layout is computed automatically.  If provided, it must have at
        least as many cells as there are rendered worlds; extra cells are
        left black.
    """

    def __init__(
        self,
        mj_model,
        warp_model,
        warp_data,
        nworld: int = 4,
        render_worlds: int | Sequence[int] | None = None,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        layout: GridLayout | None = None,
        render_depth: bool = False,
        render_seg: bool = False,
        use_textures: bool = True,
        use_shadows: bool = True,
        render_skybox: bool = False,
    ):
        # ---- resolve world_ids ------------------------------------------------
        
        self.NO_RENDER_FLAG = False  # If True, capture() becomes a no-op and we skip GPU buffer and RC setup.
        
        if render_worlds is None:
            world_ids: tuple[int, ...] = tuple(range(nworld))
            
        elif isinstance(render_worlds, int):
            if render_worlds == 0:
                self.NO_RENDER_FLAG = True
                return  # No worlds to render; skip setup and create an empty renderer.
            
            if not (1 <= render_worlds <= nworld):
                raise ValueError(
                    f"render_worlds={render_worlds} must be between 1 and "
                    f"nworld={nworld} (inclusive)"
                )
            world_ids = tuple(range(render_worlds))

        else:  # Sequence[int]
            world_ids = tuple(render_worlds)
            if len(world_ids) == 0:
                raise ValueError("render_worlds sequence must not be empty")
            out_of_range = [w for w in world_ids if not (0 <= w < nworld)]
            if out_of_range:
                raise ValueError(
                    f"world indices {out_of_range} are out of range for nworld={nworld}"
                )
            duplicates = [w for i, w in enumerate(world_ids) if w in world_ids[:i]]
            if duplicates:
                raise ValueError(f"render_worlds contains duplicate indices: {duplicates}")

        n_render = len(world_ids)
        

        # ---- resolve layout --------------------------------------------------
        if layout is None:
            layout = _auto_layout(n_render)
        elif layout.capacity < n_render:
            raise ValueError(
                f"layout {layout.rows}×{layout.cols} ({layout.capacity} cells) is "
                f"too small for {n_render} rendered worlds"
            )

        self.mj_model = mj_model
        self.model = warp_model
        self.data = warp_data
        self.nworld = nworld
        self.world_ids = world_ids
        self.camera_index = camera_index
        self.layout = layout

        # Render context must cover ALL nworld worlds; MW.get_rgb writes the
        # whole batch.  We then index into it by world_id in capture().
        self.rc = MW.create_render_context(
            mj_model,
            nworld=nworld,
            cam_res=(width, height),
            render_rgb=True,
            render_depth=render_depth,
            render_seg=render_seg,
            use_textures=use_textures,
            use_shadows=use_shadows,
            render_skybox=render_skybox,
        )

        cam_res = self.rc.cam_res.numpy()[camera_index]
        self.cam_w = int(cam_res[0])
        self.cam_h = int(cam_res[1])

        # GPU buffer: full nworld size (required by MW.get_rgb)
        self.rgb_buf = wp.zeros((nworld, self.cam_h, self.cam_w), dtype=wp.vec3)

        # CPU grid buffer: sized for the *rendered* subset only.
        # Zeros → empty trailing cells appear black.
        self.grid_buf = np.zeros(
            (layout.rows * self.cam_h, layout.cols * self.cam_w, 3),
            dtype=np.uint8,
        )

        self.writer = None

    # Convenience properties

    @property
    def n_render(self) -> int:
        """Number of worlds that will appear in the output video."""
        return len(self.world_ids)

    @property
    def grid_size(self) -> tuple[int, int]:
        """Output video resolution as (width, height) in pixels."""
        return (
            self.layout.cols * self.cam_w,
            self.layout.rows * self.cam_h,
        )

    # Public API

    def open(self, filename: str, fps: int = 30, codec: str = "libx264", quality: int = 8):
        """Open *filename* for streaming frame writes."""
        self.writer = imageio.get_writer(
            filename,
            fps=fps,
            codec=codec,
            quality=quality,
            pixelformat="yuv420p",
        )

    def _render_rgb(self) -> np.ndarray:
        """Step the physics/renderer and return an (nworld, H, W, 3) array."""
        MW.forward(self.model, self.data)
        MW.refit_bvh(self.model, self.data, self.rc)
        MW.render(self.model, self.data, self.rc)
        MW.get_rgb(self.rc, rgb_out=self.rgb_buf, camera_index=self.camera_index)
        wp.synchronize()
        return self.rgb_buf.numpy()


    def delete(self):
        if hasattr(self, 'grid_buf'):
                del self.grid_buf
        if hasattr(self, 'rgb_buf'):
            del self.rgb_buf
        if hasattr(self, 'rc'):
            del self.rc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'data'):
            del self.data
        if hasattr(self, 'mj_model'):
            del self.mj_model
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            del self.writer

    def capture(self):
        """Render one frame and append the grid tile to the video."""
        if self.writer is None:
            raise RuntimeError("Call open(filename) before capture().")
        
        if self.NO_RENDER_FLAG:
            return

        rgb = self._render_rgb()

        cols = self.layout.cols
        for idx, world_id in enumerate(self.world_ids):
            r, c = divmod(idx, cols)
            tile = _to_uint8_rgb(rgb[world_id])
            y0 = r * self.cam_h
            x0 = c * self.cam_w
            self.grid_buf[y0 : y0 + self.cam_h, x0 : x0 + self.cam_w] = tile

        self.writer.append_data(self.grid_buf)

    def close(self):
        """Flush and close the video writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"WarpGridVideoRenderer("
            f"nworld={self.nworld}, "
            f"render_worlds={list(self.world_ids)}, "
            f"layout={self.layout.rows}×{self.layout.cols}, "
            f"grid_size={self.grid_size})"
        )


# Example

if __name__ == "__main__":
    import mujoco as MJ

    xml_path = "./models/world_6_general.xml"
    mj_model = MJ.MjModel.from_xml_path(xml_path)

    # --- Scenario A: render all 6 worlds, auto layout → 2×3
    # renderer = WarpGridVideoRenderer(mj_model, warp_model, warp_data, nworld=6)

    # --- Scenario B: render only first 4 of 6, auto layout → 2×2
    # renderer = WarpGridVideoRenderer(mj_model, warp_model, warp_data,
    #                                  nworld=6, render_worlds=4)

    # --- Scenario C: render specific worlds by index, auto layout → 2×2 (+1 empty)
    # renderer = WarpGridVideoRenderer(mj_model, warp_model, warp_data,
    #                                  nworld=6, render_worlds=[0, 2, 4])

    # --- Scenario D: explicit layout override
    # renderer = WarpGridVideoRenderer(mj_model, warp_model, warp_data,
    #                                  nworld=6, render_worlds=[1, 3],
    #                                  layout=GridLayout(rows=1, cols=2))

    # --- Idiomatic context-manager usage
    # with WarpGridVideoRenderer(mj_model, warp_model, warp_data, nworld=6,
    #                            render_worlds=4) as r:
    #     r.open("out.mp4")
    #     for _ in range(500):
    #         osc.step(); MW.step(osc.model, osc.data)
    #         r.capture()

    print("Import this file and create WarpGridVideoRenderer(...) from your simulation loop.")