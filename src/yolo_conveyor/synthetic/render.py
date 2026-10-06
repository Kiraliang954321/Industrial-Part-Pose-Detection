"""Procedural shaded plastic caps and independently seeded conveyor scenes."""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Scene:
    width: int
    height: int
    seed: int
    background_bgr: tuple[int, int, int]
    grain: float
    stripe_spacing: float
    stripe_angle: float
    light_angle: float
    light_strength: float
    camera_compression: float

    @classmethod
    def create(cls, width: int, height: int, seed: int) -> "Scene":
        rng = np.random.default_rng(seed)
        bases = [(62, 64, 60), (84, 73, 58), (56, 72, 55), (94, 91, 87),
                 (42, 47, 55), (104, 85, 68)]
        base = bases[int(rng.integers(len(bases)))]
        color = tuple(int(np.clip(v + rng.uniform(-15, 15), 20, 130)) for v in base)
        return cls(width, height, seed, color, float(rng.uniform(2, 7)),
                   float(rng.uniform(16, 44)), float(rng.uniform(-0.20, 0.20)),
                   float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0.65, 1.1)),
                   float(rng.uniform(0.68, 0.98)))

    def description(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CapPose:
    center_x: float
    center_y: float
    diameter: float
    compression: float
    angle: float
    material_seed: int
    brightness: float = 1.0
    blur_sigma: float = 0.0
    occlusion: bool = False


def conveyor_background(scene: Scene) -> np.ndarray:
    """Fixed session texture; sample sensor noise is added independently later."""
    rng = np.random.default_rng(scene.seed)
    y, x = np.mgrid[:scene.height, :scene.width].astype(np.float32)
    across = y + x * math.tan(scene.stripe_angle)
    stripe = np.mod(across, scene.stripe_spacing)
    channels = np.asarray(scene.background_bgr, dtype=np.float32)
    texture = rng.normal(0, scene.grain, (scene.height, scene.width)).astype(np.float32)
    texture += 1.8 * np.sin(x * 0.18 + y * 0.04)
    texture += 3.5 * np.cos(across / scene.stripe_spacing * 2 * math.pi)
    texture -= (stripe < 1.6) * 9
    texture += (stripe > scene.stripe_spacing - 1.2) * 4
    lamp = 12 * (x / scene.width - 0.5) * math.cos(scene.light_angle)
    lamp += 10 * (y / scene.height - 0.5) * math.sin(scene.light_angle)
    image = np.clip(channels + (texture + lamp)[..., None], 0, 255).astype(np.uint8)
    # Conveyor rail edges remain outside the usual object region.
    cv2.line(image, (0, 15), (scene.width, 15), tuple(int(v * 1.6) for v in channels), 8)
    cv2.line(image, (0, scene.height - 16), (scene.width, scene.height - 16),
             tuple(int(v * 0.5) for v in channels), 9)
    return image


def _cap_surface(face: str, scene: Scene, pose: CapPose) -> tuple[np.ndarray, np.ndarray]:
    """Canonical 256 x 272 cap: round face plus an exposed 16-pixel sidewall."""
    rng = np.random.default_rng(pose.material_seed)
    y, x = np.mgrid[:272, :256].astype(np.float32)
    nx, ny = (x - 127.5) / 127.5, (y - 127.5) / 127.5
    radius = np.sqrt(nx * nx + ny * ny)
    angle = np.arctan2(ny, nx)
    disk = radius <= 1
    lower_radius = np.sqrt(nx * nx + ((y - 143.5) / 127.5) ** 2)
    side = (lower_radius <= 1) & ~disk
    alpha = (disk | side).astype(np.float32)
    yellow = np.array([18, 188, 240], dtype=np.float32)
    yellow += rng.uniform(-6, 6, 3).astype(np.float32)
    side_ridges = 0.10 * np.cos(np.arccos(np.clip(nx, -1, 1)) * 72)
    side_light = 0.48 + 0.18 * (ny - 0.3) + side_ridges
    color = yellow[None, None, :] * side_light[..., None]
    lx, ly = math.cos(scene.light_angle), math.sin(scene.light_angle)
    # A shallow crown, rounded shoulder and 72 moulded grip ridges.
    normal_x = nx * 0.28
    normal_y = ny * 0.28
    shoulder = np.clip((radius - 0.83) / 0.17, 0, 1)
    normal_x += nx * shoulder * 0.72
    normal_y += ny * shoulder * 0.72
    lambert = np.clip(0.90 - 0.26 * (normal_x * lx + normal_y * ly), 0.44, 1.18)
    ridges = np.cos(angle * 72) * 0.095 * np.clip((radius - 0.86) / 0.09, 0, 1)
    specular = 0.20 * np.exp(-((nx + lx * 0.38) ** 2 + (ny + ly * 0.38) ** 2) / 0.065)
    material_noise = rng.normal(0, 0.009, radius.shape).astype(np.float32)
    surface = yellow[None, None, :] * (lambert + ridges + specular + material_noise)[..., None]
    if face == "back":
        # Recessed bowl and two thread rings, rather than a flat color switch.
        inner = radius < 0.77
        bowl = 0.74 + radius * 0.24 + (nx * lx + ny * ly) * 0.12
        surface[inner] = (yellow[None, None, :] * bowl[..., None])[inner]
        for ring_r, ring_width in ((0.79, 0.018), (0.67, 0.013), (0.50, 0.008)):
            shadow = np.exp(-((radius - ring_r) / ring_width) ** 2)
            highlight = np.exp(-((radius - ring_r + 0.018) / ring_width) ** 2)
            surface *= (1 - shadow[..., None] * 0.30)
            surface += highlight[..., None] * 14
    color[disk] = surface[disk]
    color = np.clip(color, 0, 255).astype(np.uint8)
    # The face graphic rotates with the object, and is independent of the scene.
    graphic = np.zeros_like(color)
    ink = np.zeros(radius.shape, dtype=np.uint8)
    if face == "front":
        points = np.array([[91, 83], [136, 92], [155, 109], [143, 125],
                           [168, 153], [153, 171], [128, 143], [116, 157],
                           [95, 144], [78, 104]], dtype=np.int32)
        cv2.fillPoly(ink, [points], 218, lineType=cv2.LINE_AA)
        cv2.ellipse(ink, (124, 128), (68, 67), 8, 215, 342, 182, 6, cv2.LINE_AA)
        cv2.line(ink, (85, 168), (124, 174), 190, 5, cv2.LINE_AA)
        graphic[:] = (26, 40, 183)
    else:
        cv2.rectangle(ink, (100, 101), (153, 153), 222, -1, cv2.LINE_AA)
        cv2.rectangle(ink, (109, 110), (144, 144), 150, 3, cv2.LINE_AA)
        cv2.line(ink, (114, 138), (139, 115), 115, 3, cv2.LINE_AA)
        graphic[:] = (35, 40, 44)
        cv2.circle(color, (128, 128), 9, (35, 147, 193), 2, cv2.LINE_AA)
    amount = (ink.astype(np.float32) / 255)[..., None]
    color = color.astype(np.float32) * (1 - amount) + graphic * amount
    color *= scene.light_strength * pose.brightness
    return np.clip(color, 0, 255).astype(np.uint8), alpha


def cap_corners(scene: Scene, pose: CapPose) -> list[list[float]]:
    """Exact enclosing rectangle of the affine-projected canonical cap silhouette."""
    width = pose.diameter
    height = pose.diameter * (272 / 256) * pose.compression
    c, s = math.cos(pose.angle), math.sin(pose.angle)
    result = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        result.append([(pose.center_x + c * x - s * y) / scene.width,
                       (pose.center_y + s * x + c * y) / scene.height])
    return result


def render_cap(scene: Scene, background: np.ndarray, face: str | None,
               pose: CapPose | None, rng: np.random.Generator) -> np.ndarray:
    image = background.astype(np.float32).copy()
    if face is not None and pose is not None:
        color, alpha = _cap_surface(face, scene, pose)
        c, s = math.cos(pose.angle), math.sin(pose.angle)
        sx, sy = pose.diameter / 256, pose.diameter * pose.compression / 256
        matrix = np.array([[c * sx, -s * sy, pose.center_x - c * sx * 127.5 + s * sy * 135.5],
                           [s * sx, c * sy, pose.center_y - s * sx * 127.5 - c * sy * 135.5]],
                          dtype=np.float32)
        size = (scene.width, scene.height)
        # Warp premultiplied color: boundary interpolation does not introduce black halos.
        projected_alpha = cv2.warpAffine(alpha, matrix, size, flags=cv2.INTER_LINEAR)
        projected = cv2.warpAffine(color.astype(np.float32) * alpha[..., None], matrix, size,
                                   flags=cv2.INTER_LINEAR)
        shadow_matrix = matrix.copy()
        shadow_matrix[0, 2] += math.cos(scene.light_angle) * pose.diameter * 0.04
        shadow_matrix[1, 2] += math.sin(scene.light_angle) * pose.diameter * 0.04 + 5
        shadow = cv2.warpAffine(alpha, shadow_matrix, size, flags=cv2.INTER_LINEAR)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, pose.diameter * 0.027))
        image *= (1 - shadow[..., None] * 0.32)
        image = image * (1 - projected_alpha[..., None]) + projected
        if pose.occlusion:
            # A narrow translucent strand near the outer rim leaves the central face cue intact.
            corners = np.asarray(cap_corners(scene, pose)) * [scene.width, scene.height]
            start = corners[0] * 0.90 + corners[3] * 0.10
            end = corners[1] * 0.90 + corners[2] * 0.10
            strand = image.copy()
            cv2.line(strand, tuple(start.astype(int)), tuple(end.astype(int)), (96, 102, 104),
                     max(1, round(pose.diameter * 0.035)), cv2.LINE_AA)
            image = image * 0.25 + strand * 0.75
        if pose.blur_sigma > 0:
            image = cv2.GaussianBlur(image, (0, 0), pose.blur_sigma)
    image += rng.normal(0, 0.7, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)
