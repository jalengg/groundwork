import numpy as np
import networkx as nx
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def compute_connectivity_index(G: nx.Graph) -> float:
    """CI = average node degree (sum of degrees / number of nodes)."""
    if G.number_of_nodes() == 0:
        return 0.0
    degrees = [d for _, d in G.degree()]
    return sum(degrees) / G.number_of_nodes()


def compute_transport_convenience(G: nx.Graph, sample_pairs: int = 200) -> float:
    """
    TC = mean(euclidean_dist / shortest_path_dist) over random node pairs.

    Requires node attributes 'x', 'y' for spatial graphs (e.g. from OSMnx),
    or uses integer node IDs as 1-D coordinates for simple graphs (e.g. path_graph).
    Values closer to 1.0 indicate more direct routes (less detour).
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    scores = []
    n_pairs = min(sample_pairs, len(nodes) * (len(nodes) - 1) // 2)
    for _ in range(n_pairs):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            sp = nx.shortest_path_length(G, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if sp == 0:
            continue
        # Coordinates: tuple nodes (grid_2d_graph), OSMnx x/y attrs, or integer ID
        if isinstance(u, tuple):
            ux, uy = u
            vx, vy = v
        else:
            ux = G.nodes[u].get("x", float(u))
            uy = G.nodes[u].get("y", 0.0)
            vx = G.nodes[v].get("x", float(v))
            vy = G.nodes[v].get("y", 0.0)
        euclid = ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5
        if euclid > 0:
            scores.append(euclid / sp)
    return float(np.mean(scores)) if scores else 0.0


class ImageQualityTracker:
    """Accumulates real/fake samples and computes FID and KID."""

    def __init__(self, device="cpu"):
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
        self.device = device

    def update_real(self, images_rgb: torch.Tensor):
        """images_rgb: (B, 3, H, W) float32 in [0, 1]"""
        self.fid.update(images_rgb.to(self.device), real=True)
        self.kid.update(images_rgb.to(self.device), real=True)

    def update_fake(self, images_rgb: torch.Tensor):
        self.fid.update(images_rgb.to(self.device), real=False)
        self.kid.update(images_rgb.to(self.device), real=False)

    def compute(self):
        return {
            "fid": self.fid.compute().item(),
            "kid": self.kid.compute()[0].item(),
        }
