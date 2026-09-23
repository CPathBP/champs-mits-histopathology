"""Tissue images for figures: slide thumbnails and tile crops at the resolution of the features.

Tile positions come from the coordinates stored with the tile features, so a
crop shows exactly a tile that the encoder saw. Tiles are chosen by a fixed
rule, so that a figure shows the same tissue every time it is rendered.
"""

import h5py
import numpy as np
import openslide

CANDIDATE_TILES = 200


def thumbnail(slide_path, max_pixels):
    """The whole slide as an RGB image whose longer side is at most ``max_pixels``."""
    with openslide.OpenSlide(str(slide_path)) as slide:
        return slide.get_thumbnail((max_pixels, max_pixels)).convert("RGB")


def slide_dimensions(slide_path):
    """Width and height of the slide at full resolution, in pixels."""
    with openslide.OpenSlide(str(slide_path)) as slide:
        return slide.dimensions


def microns_per_pixel(slide_path):
    """The horizontal size of a full-resolution pixel of the slide, in micrometres."""
    with openslide.OpenSlide(str(slide_path)) as slide:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])


def feature_description(feature_path):
    """The number of tiles, the feature dimension, the tile size and the tile magnification."""
    with h5py.File(feature_path, "r") as features:
        n_tiles, dimension = features["features"].shape
        attributes = features["coords"].attrs
        return {
            "tiles": int(n_tiles),
            "feature_dimension": int(dimension),
            "tile_pixels": int(attributes["patch_size"]),
            "magnification": int(attributes["target_magnification"]),
        }


def tile_grid(feature_path):
    """The full-resolution tile coordinates, the tile size there, and the size the encoder saw."""
    with h5py.File(feature_path, "r") as features:
        coordinates = features["coords"][:]
        attributes = features["coords"].attrs
        return coordinates, int(attributes["patch_size_level0"]), int(attributes["patch_size"])


def stained_tiles(slide_path, feature_path, count):
    """The ``count`` most strongly stained tiles of a slide, with their full-resolution positions.

    Candidates are evenly spaced tiles of the slide in the stored order; among
    them the tiles with the highest mean colour saturation are returned, which
    favours tissue over background and faint areas. Each tile is returned as
    ``(x, y, image)`` with the image at the resolution the encoder saw.
    """
    coordinates, size, encoder_size = tile_grid(feature_path)
    step = max(1, len(coordinates) // CANDIDATE_TILES)
    scored = []
    with openslide.OpenSlide(str(slide_path)) as slide:
        for x, y in coordinates[::step]:
            region = slide.read_region((int(x), int(y)), 0, (size, size)).convert("RGB")
            tile = region.resize((encoder_size, encoder_size))
            saturation = np.asarray(tile.convert("HSV"), dtype=float)[..., 1].mean()
            scored.append((saturation, int(x), int(y), tile))
    scored.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    return [(x, y, tile) for _, x, y, tile in scored[:count]]
