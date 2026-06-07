import warp as wp
import torch

# pyright: reportInvalidTypeForm=false

wp.init()


# ============================================================
# Helpers
# ============================================================

@wp.func
def cubic_bezier(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    t: float,
):
    omt = 1.0 - t

    return (
        omt * omt * omt * p0
        + 3.0 * omt * omt * t * p1
        + 3.0 * omt * t * t * p2
        + t * t * t * p3
    )


@wp.func
def perpendicular_basis(u: wp.vec3):

    tmp = wp.vec3(1.0, 0.0, 0.0)

    if wp.abs(wp.dot(tmp, u)) > 0.9:
        tmp = wp.vec3(0.0, 1.0, 0.0)

    v = wp.normalize(wp.cross(u, tmp))
    w = wp.normalize(wp.cross(u, v))

    return v, w


# ============================================================
# Kernel
# ============================================================

@wp.kernel
def generate_bezier_se3(
    start_pos: wp.array2d(dtype=wp.vec3),
    goal_pos: wp.array2d(dtype=wp.vec3),

    start_rot: wp.array2d(dtype=wp.quat),
    goal_rot: wp.array2d(dtype=wp.quat),

    noise1: wp.array3d(dtype=float),
    noise2: wp.array3d(dtype=float),

    traj_pos: wp.array3d(dtype=wp.vec3),
    traj_rot: wp.array3d(dtype=wp.quat),

    sigma: float,
):

    world, pair, step = wp.tid()

    nsteps = traj_pos.shape[2]

    t = float(step) / float(nsteps - 1)

    p0 = start_pos[world, pair]
    p3 = goal_pos[world, pair]

    q0 = start_rot[world, pair]
    q3 = goal_rot[world, pair]

    d = p3 - p0
    L = wp.length(d)

    if L < 1.0e-8:

        traj_pos[world, pair, step] = p0
        traj_rot[world, pair, step] = wp.quat_slerp(q0, q3, t)
        return

    u = d / L

    v, w = perpendicular_basis(u)

    n1v = noise1[world, pair, 0]
    n1w = noise1[world, pair, 1]

    n2v = noise2[world, pair, 0]
    n2w = noise2[world, pair, 1]

    p1 = (
        p0
        + 0.30 * d
        + sigma * L * n1v * v
        + sigma * L * n1w * w
    )

    p2 = (
        p0
        + 0.70 * d
        + sigma * L * n2v * v
        + sigma * L * n2w * w
    )

    pos = cubic_bezier(
        p0,
        p1,
        p2,
        p3,
        t,
    )

    rot = wp.quat_slerp(
        q0,
        q3,
        t,
    )

    traj_pos[world, pair, step] = pos
    traj_rot[world, pair, step] = rot


# ============================================================
# Wrapper
# ============================================================

class BezierTrajectory:

    def __init__(
        self,
        nworld: int,
        nsteps: int,
        device: str = "cuda",
    ):
        self.nworld = nworld
        self.nsteps = nsteps
        self.device = device

    def generate(
        self,
        start_pos_wp,
        goal_pos_wp,
        start_rot_wp,
        goal_rot_wp,
        sigma: float = 0.2,
    ):

        #
        # one trajectory for each pose pair
        #
        # shape:
        # (nworld, 2, 2)
        #
        noise1_torch = torch.randn(
            self.nworld,
            2,
            2,
            device=self.device,
        )

        noise2_torch = torch.randn(
            self.nworld,
            2,
            2,
            device=self.device,
        )

        noise1_wp = wp.from_torch(
            noise1_torch,
            dtype=wp.float32,
        )

        noise2_wp = wp.from_torch(
            noise2_torch,
            dtype=wp.float32,
        )

        traj_pos_wp = wp.empty(
            shape=(self.nworld, 2, self.nsteps),
            dtype=wp.vec3,
            device=self.device,
        )

        traj_rot_wp = wp.empty(
            shape=(self.nworld, 2, self.nsteps),
            dtype=wp.quat,
            device=self.device,
        )

        wp.launch(
            generate_bezier_se3,
            dim=(self.nworld, 2, self.nsteps),
            inputs=[
                start_pos_wp,
                goal_pos_wp,
                start_rot_wp,
                goal_rot_wp,
                noise1_wp,
                noise2_wp,
                traj_pos_wp,
                traj_rot_wp,
                sigma,
            ],
            device=self.device,
        )

        #
        # Warp layout:
        #
        # (nworld,2,nsteps,...)
        #
        # Controller layout:
        #
        # (nsteps,nworld,2,...)
        #

        traj_pos = (
            wp.to_torch(traj_pos_wp)
            .permute(2, 1, 0, 3)
            .contiguous()
        )

        traj_rot = (
            wp.to_torch(traj_rot_wp)
            .permute(2, 1, 0, 3)
            .contiguous()
        )

        return traj_pos, traj_rot