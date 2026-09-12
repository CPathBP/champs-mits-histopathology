"""Import shim: the DINOv2 fork imports ``champs_tile_dataset`` from the path."""
from champs_pipeline.ssl.tile_dataset import ChampsTileDataset, build_shard_index  # noqa: F401
