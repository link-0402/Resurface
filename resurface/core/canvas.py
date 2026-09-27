"""UVs for a texture whose canvas was cropped or extended.

When an image editor changes the canvas size, the pixels keep their content but move:
the new canvas sits somewhere on the old image (its anchor, plus an optional offset).
UVs have to follow so they keep sampling the same pixels.  The mapping is exact; the
only rounding is for centred anchors with an odd size difference, where the offset is
rounded down.

UV space: u to the right, v up, (0, 0) is the bottom left of the texture.  Offsets are
given the image editor way: x to the right, y down, from the top left corner.
"""

import math

# anchor -> (horizontal, vertical) position of the new canvas on the old one:
# 0 = left / top edges aligned, 0.5 = centred, 1 = right / bottom edges aligned
ANCHORS = {
    "TOP_LEFT": (0.0, 0.0), "TOP": (0.5, 0.0), "TOP_RIGHT": (1.0, 0.0),
    "LEFT": (0.0, 0.5), "CENTER": (0.5, 0.5), "RIGHT": (1.0, 0.5),
    "BOTTOM_LEFT": (0.0, 1.0), "BOTTOM": (0.5, 1.0), "BOTTOM_RIGHT": (1.0, 1.0),
}


def canvas_offset(old_size, new_size, anchor="TOP_LEFT", extra=(0, 0)):
    """Top left corner of the new canvas in pixels of the old image (x right, y down)."""
    fx, fy = ANCHORS[anchor]
    ox = math.floor(fx * (old_size[0] - new_size[0]))
    oy = math.floor(fy * (old_size[1] - new_size[1]))
    return ox + int(extra[0]), oy + int(extra[1])


def canvas_transform(old_size, new_size, offset):
    """(scale, shift) so that uv_new = uv_old * scale + shift."""
    W0, H0 = (float(v) for v in old_size)
    W1, H1 = (float(v) for v in new_size)
    ox, oy = offset
    # pixel x = u * W0 - ox on the new canvas; y counts down from the top
    su = W0 / W1
    sv = H0 / H1
    return (su, sv), (-ox / W1, 1.0 - sv + oy / H1)
