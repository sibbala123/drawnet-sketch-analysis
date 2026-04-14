"""
DrawNet Synthetic Deviation Augmentation
-----------------------------------------
Applies four geometrically motivated drawing deviations to clean PIL images.
Each function operates on a PIL image and returns a PIL image.

Deviation index mapping (matches DEVIATION_CLASSES in dataset.py):
    0 - rotation
    1 - closure_failure
    2 - spatial_disorganization
    3 - size_distortion
"""

import random
from typing import List, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Individual deviation functions
# ---------------------------------------------------------------------------

def apply_rotation(
    image: Image.Image,
    min_angle: float = 15.0,
    max_angle: float = 90.0,
) -> Image.Image:
    """
    Randomly rotate the drawing by an angle in [min_angle, max_angle]
    (or its negative counterpart, chosen with equal probability).
    Background is filled with white.
    """
    angle = random.uniform(min_angle, max_angle)
    if random.random() < 0.5:
        angle = -angle
    return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def apply_closure_failure(
    image: Image.Image,
    fraction: float = 0.1,
) -> Image.Image:
    """
    Simulate closure failure by erasing a fraction of dark (stroke) pixels,
    creating gaps in closed shapes.

    Locates dark pixels, randomly erases `fraction` of them by setting small
    neighbourhoods to white.
    """
    img_np = np.array(image.convert("L"))
    h, w = img_np.shape

    dark_coords = np.argwhere(img_np < 128)
    if len(dark_coords) == 0:
        return image

    n_erase = max(1, int(len(dark_coords) * fraction))
    chosen = dark_coords[np.random.choice(len(dark_coords), n_erase, replace=False)]

    result = img_np.copy()
    radius = max(2, int(min(h, w) * 0.01))
    for r, c in chosen:
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        result[r0:r1, c0:c1] = 255

    gray = Image.fromarray(result, mode="L")
    return Image.merge("RGB", [gray, gray, gray])


def apply_spatial_disorganization(
    image: Image.Image,
    num_shifts: int = 3,
) -> Image.Image:
    """
    Simulate spatial disorganization by dividing the image into a 3x3 grid,
    randomly selecting `num_shifts` cells, and displacing their content
    by a random offset.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    result = np.full_like(img_np, 255)

    grid = 3
    cell_h, cell_w = h // grid, w // grid

    cells = [(r, c) for r in range(grid) for c in range(grid)]
    to_shift = random.sample(cells, min(num_shifts, len(cells)))
    to_shift_set = set(to_shift)

    for r in range(grid):
        for c in range(grid):
            r0, r1 = r * cell_h, (r + 1) * cell_h
            c0, c1 = c * cell_w, (c + 1) * cell_w
            patch = img_np[r0:r1, c0:c1]

            if (r, c) in to_shift_set:
                dr = random.randint(-cell_h // 2, cell_h // 2)
                dc = random.randint(-cell_w // 2, cell_w // 2)
                tr0 = np.clip(r0 + dr, 0, h - (r1 - r0))
                tc0 = np.clip(c0 + dc, 0, w - (c1 - c0))
                tr1 = tr0 + (r1 - r0)
                tc1 = tc0 + (c1 - c0)
                result[tr0:tr1, tc0:tc1] = np.minimum(
                    result[tr0:tr1, tc0:tc1], patch
                )
            else:
                result[r0:r1, c0:c1] = np.minimum(result[r0:r1, c0:c1], patch)

    return Image.fromarray(result)


def apply_size_distortion(
    image: Image.Image,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> Image.Image:
    """
    Scale the drawing content within the canvas to simulate size distortion
    (too small or too large). Canvas size stays fixed at the original dimensions.
    """
    scale = random.uniform(*scale_range)
    orig_w, orig_h = image.size
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    scaled = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))
    paste_x = (orig_w - new_w) // 2
    paste_y = (orig_h - new_h) // 2

    crop_x = max(0, -paste_x)
    crop_y = max(0, -paste_y)
    crop_w = min(new_w, orig_w - max(0, paste_x))
    crop_h = min(new_h, orig_h - max(0, paste_y))
    region = scaled.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    dest_x = max(0, paste_x)
    dest_y = max(0, paste_y)
    canvas.paste(region, (dest_x, dest_y))
    return canvas


# ---------------------------------------------------------------------------
# Compound generator
# ---------------------------------------------------------------------------

DEVIATION_NAMES = [
    "rotation",
    "closure_failure",
    "spatial_disorganization",
    "size_distortion",
]

_DEVIATION_FUNCS = [
    apply_rotation,
    apply_closure_failure,
    apply_spatial_disorganization,
    apply_size_distortion,
]


def generate_deviation_sample(
    image: Image.Image,
    p_each: float = 0.6,
    min_deviations: int = 1,
) -> Tuple[Image.Image, List[int]]:
    """
    Apply a random subset of deviations to a drawing that has been selected
    for augmentation (i.e. is in the 20% deviated pool).

    Each of the 4 deviations fires independently at probability p_each.
    min_deviations=1 guarantees at least one deviation is applied.

    Parameters
    ----------
    image : PIL.Image
    p_each : float
        Independent probability each deviation is applied. Default 0.6 gives
        ~1.6 deviations per image on average.
    min_deviations : int
        Minimum number of deviations to apply (re-samples if none fired).

    Returns
    -------
    augmented_image : PIL.Image
    deviation_label_vector : List[int]
        Binary list of length 4; 1 if the corresponding deviation was applied.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    label_vector = [1 if random.random() < p_each else 0
                    for _ in _DEVIATION_FUNCS]

    # Enforce minimum — if nothing fired, force one random deviation
    if sum(label_vector) < min_deviations:
        idx = random.randrange(len(label_vector))
        label_vector[idx] = 1

    aug = image
    for i, apply_fn in enumerate(_DEVIATION_FUNCS):
        if label_vector[i]:
            aug = apply_fn(aug)
            if aug.mode != "RGB":
                aug = aug.convert("RGB")

    return aug, label_vector
