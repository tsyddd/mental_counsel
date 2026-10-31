"""Relation VideoARM public API with lazy loading of video tooling."""
from .api import APIClient
from .graph import build_graph, load_caption_blocks


def __getattr__(name):
    # Keep image/video processing dependencies out of lightweight API/worker imports.
    if name == "RelationVideoARMAgent":
        from .agent import RelationVideoARMAgent
        return RelationVideoARMAgent
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["APIClient", "RelationVideoARMAgent", "build_graph", "load_caption_blocks"]
