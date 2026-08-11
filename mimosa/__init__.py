"""mimosa-fl: Clustered Federated Learning framework built on Flower.

Implements the CFL method from Sattler, Muller & Samek (arXiv:1910.01991):
clients are recursively grouped into clusters by the cosine similarity of
their weight-update directions, and each cluster maintains its own model.
"""

__version__ = "0.1.0"
