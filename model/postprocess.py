#!/usr/bin/env python3
"""
CaRoLS Section 3.3 vectorization post-processing.

Six-step pipeline (paper-aligned):
  1. Semantic discretization (argmax → S)
  2. Morphological buffering (binary dilation, r=1-2 px)
  3. Skeleton extraction + graph construction (medial axis → networkx graph;
     edges inherit majority road level from underlying pixels)
  4. Topology cleaning (drop CCs < 20 px path length; bridge remaining
     disjoint CCs to the main graph by shortest-Euclidean nearest-node link)
  5. Node merging (< 2 px) + Douglas-Peucker simplification (tol 1-2 px)
  6. Render the cleaned graph back to a raster for visual eval (paper exports
     ESRI Shapefile here; we re-render so we can keep using image-domain diag)

The paper's evaluation protocol (Sec 4.2) computes CI/TC ONLY on the vectorized
output — they never grade the raw raster. This module is what you run BEFORE
any topological/connectivity metric to be apples-to-apples with paper numbers.

Usage:
    python model/postprocess.py \\
        --vae checkpoints/vae_categorical/vae_epoch_050.pth \\
        --diffusion checkpoints/diff_eq9_planAfix_v2/diffusion_epoch_200.pth \\
        --n 4 --output samples/postprocessed_eq9_planAfix_v2
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sknw
import torch
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from skimage.morphology import binary_dilation, disk, medial_axis

from model.diffusion import DDPM
from model.unet import DiffusionUNet
from model.vae import RoadVAE


# Display colors (match train_diffusion.onehot_to_rgb)
ROAD_COLORS = {
    0: [0.15, 0.15, 0.15],  # background
    1: [0.9, 0.9, 0.9],     # residential
    2: [0.6, 0.8, 0.4],     # tertiary
    3: [0.9, 0.6, 0.2],     # primary
    4: [0.9, 0.2, 0.2],     # motorway
}

# Render widths per class (paper doesn't specify; chosen to match
# data_pipeline.road_layers per-class line widths, scaled for vis).
RENDER_WIDTH_PX = {1: 2, 2: 3, 3: 4, 4: 5}


def onehot_to_rgb(idx):
    rgb = np.zeros((*idx.shape, 3))
    for c, col in ROAD_COLORS.items():
        rgb[idx == c] = col
    return rgb


def _edge_majority_level(edge_pts, sem_map):
    """Sample sem_map at each (y,x) along an edge polyline → majority level
    among non-bg pixels (paper §3.3 step 3)."""
    ys = edge_pts[:, 0].astype(int).clip(0, sem_map.shape[0] - 1)
    xs = edge_pts[:, 1].astype(int).clip(0, sem_map.shape[1] - 1)
    labels = sem_map[ys, xs]
    road_labels = labels[labels > 0]
    if road_labels.size == 0:
        return 0
    counts = np.bincount(road_labels, minlength=5)
    return int(counts.argmax())


def _build_initial_graph(sem_map, dilate_radius=2):
    """Steps 2-3: dilate the binary road mask, medial-axis skeletonize, then
    sknw → networkx graph. Each edge gets a 'level' attribute via majority
    vote and an 'orig_pts' attribute (the raw skeleton polyline)."""
    road_mask = sem_map > 0
    if not road_mask.any():
        return nx.Graph(), road_mask
    buffered = binary_dilation(road_mask, footprint=disk(dilate_radius))
    skel = medial_axis(buffered)

    # sknw returns a multigraph with node 'o' (origin pixel) and edge 'pts'
    G_multi = sknw.build_sknw(skel.astype(np.uint8), multi=True)
    G = nx.Graph()
    for node, attrs in G_multi.nodes(data=True):
        G.add_node(node, pos=tuple(attrs["o"]))  # (y, x) in row-major coords
    for u, v, attrs in G_multi.edges(data=True):
        pts = attrs["pts"]                                   # (N, 2)
        level = _edge_majority_level(pts, sem_map)
        if level == 0:
            continue                                         # skip pure-bg edges
        if G.has_edge(u, v):
            # Keep the longer of two parallel edges (sknw multigraph collapse)
            if len(pts) > len(G[u][v]["pts"]):
                G[u][v].update(pts=pts, level=level)
        else:
            G.add_edge(u, v, pts=pts, level=level, weight=float(len(pts)))
    return G, skel


def _drop_short_components(G, min_path_px=20):
    """Step 4a: drop connected components whose total edge-path-length < N px."""
    components = list(nx.connected_components(G))
    keep_nodes = set()
    for comp in components:
        total = 0
        for u, v in G.subgraph(comp).edges():
            total += G[u][v]["weight"]
        if total >= min_path_px:
            keep_nodes |= comp
    return G.subgraph(keep_nodes).copy()


def _bridge_disjoint_components(G):
    """Step 4b: attach every non-largest CC to the largest by shortest-
    Euclidean node-to-node link. Adds a fake 'level=1' (residential) bridge
    edge; paper doesn't say what level the bridge gets so we use the lowest
    so it never wins a majority vote downstream."""
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return G
    main = components[0]
    main_nodes = list(main)
    main_xy = np.array([G.nodes[n]["pos"] for n in main_nodes])
    tree = cKDTree(main_xy)
    for comp in components[1:]:
        comp_nodes = list(comp)
        comp_xy = np.array([G.nodes[n]["pos"] for n in comp_nodes])
        d, idx = tree.query(comp_xy, k=1)
        best = int(d.argmin())
        u = comp_nodes[best]                          # closest node in this CC
        v = main_nodes[int(idx[best])]                # closest in main
        bridge_pts = np.array([G.nodes[u]["pos"], G.nodes[v]["pos"]])
        G.add_edge(
            u, v,
            pts=bridge_pts,
            level=1,
            weight=float(d[best]),
        )
        # bridged comp now part of main for subsequent queries
        main_nodes.extend(comp_nodes)
        main_xy = np.vstack([main_xy, comp_xy])
        tree = cKDTree(main_xy)
    return G


def _merge_close_nodes(G, merge_px=2):
    """Step 5a: collapse node pairs whose Euclidean distance < merge_px.
    Greedy: pick a node, find all neighbors-within-merge_px in coord space,
    coalesce them into one node at their centroid, redirect edges."""
    if len(G) == 0:
        return G
    pos = {n: np.asarray(G.nodes[n]["pos"], dtype=float) for n in G.nodes()}
    nodes_remaining = set(G.nodes())
    parent = {n: n for n in nodes_remaining}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    coords = np.array([pos[n] for n in G.nodes()])
    node_list = list(G.nodes())
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=merge_px)
    for i, j in pairs:
        a, b = find(node_list[i]), find(node_list[j])
        if a != b:
            parent[a] = b

    # Build merged graph
    merged = nx.Graph()
    cluster_pos = {}
    for n in G.nodes():
        root = find(n)
        cluster_pos.setdefault(root, []).append(pos[n])
    for root, ps in cluster_pos.items():
        merged.add_node(root, pos=tuple(np.mean(ps, axis=0)))
    for u, v, attrs in G.edges(data=True):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue                                  # self-loop after merge
        if merged.has_edge(ru, rv):
            # Keep the higher-priority level + longer path among parallels
            if attrs["level"] > merged[ru][rv]["level"]:
                merged[ru][rv].update(attrs)
        else:
            merged.add_edge(ru, rv, **attrs)
    return merged


def _simplify_edges(G, dp_tolerance=2):
    """Step 5b: Douglas-Peucker simplification per edge polyline."""
    for u, v, attrs in G.edges(data=True):
        pts = attrs["pts"]
        if len(pts) < 3:
            continue
        ls = LineString([(p[1], p[0]) for p in pts])  # shapely is (x, y)
        simp = ls.simplify(dp_tolerance, preserve_topology=False)
        coords = np.array(simp.coords)
        attrs["pts"] = np.column_stack([coords[:, 1], coords[:, 0]])
    return G


def _render_graph(G, shape, render_widths=None):
    """Step 6: rasterize the cleaned graph back to (H, W) class-index map.
    Higher-priority classes overwrite lower (motorway > primary > tertiary >
    residential)."""
    if render_widths is None:
        render_widths = RENDER_WIDTH_PX
    H, W = shape
    out = np.zeros((H, W), dtype=np.int32)
    edges_by_level = sorted(
        G.edges(data=True), key=lambda e: e[2]["level"]
    )                                                 # ascending so high overwrites
    for _, _, attrs in edges_by_level:
        level = attrs["level"]
        if level == 0:
            continue
        pts = attrs["pts"]
        # cv2 polylines wants (x, y) int32
        pts_xy = np.array(
            [[int(round(p[1])), int(round(p[0]))] for p in pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(canvas, [pts_xy], isClosed=False,
                      color=1, thickness=render_widths.get(level, 2))
        out[canvas > 0] = level
    return out


def vectorize_layout(
    logits,
    dilate_radius=2,
    min_path_px=20,
    merge_px=2,
    dp_tolerance=2,
    render_widths=None,
):
    """Full CaRoLS Sec 3.3 pipeline. Takes (5, H, W) logits → (H, W) class-
    index raster of the vectorized network, plus the networkx graph for
    downstream metrics.

    Returns: (rendered_raster: np.ndarray[H,W] int32, graph: nx.Graph)
    """
    if torch.is_tensor(logits):
        sem_map = logits.argmax(0).cpu().numpy().astype(np.int32)
    else:
        sem_map = logits.argmax(0).astype(np.int32)
    H, W = sem_map.shape

    G, _ = _build_initial_graph(sem_map, dilate_radius=dilate_radius)
    if len(G) == 0:
        return np.zeros((H, W), dtype=np.int32), G
    G = _drop_short_components(G, min_path_px=min_path_px)
    if len(G) == 0:
        return np.zeros((H, W), dtype=np.int32), G
    G = _bridge_disjoint_components(G)
    G = _merge_close_nodes(G, merge_px=merge_px)
    G = _simplify_edges(G, dp_tolerance=dp_tolerance)
    rendered = _render_graph(G, (H, W), render_widths=render_widths)
    return rendered, G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="checkpoints/vae_categorical/vae_epoch_050.pth")
    p.add_argument("--diffusion", required=True)
    p.add_argument("--data", default="data/irving_tx")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--output", default="samples/postprocessed")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--min-path-px", type=int, default=20)
    p.add_argument("--merge-px", type=int, default=2)
    p.add_argument("--dp-tol", type=int, default=2)
    p.add_argument("--local-module", choices=["lde", "load"], default="lde")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=7,
                        local_module=args.local_module).to(device)
    net.load_state_dict(torch.load(args.diffusion, map_location=device)["model"])
    net.eval()

    ddpm = DDPM(T=1000)
    cond_files = sorted(
        f for f in os.listdir(args.data)
        if f.startswith("cond_") and f.endswith(".npy")
    )[:args.n]

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = [axes]

    for i, cf in enumerate(cond_files):
        idx = cf.replace("cond_", "").replace(".npy", "")
        cond_full = np.load(os.path.join(args.data, cf)).astype(np.float32)
        road_np = np.load(os.path.join(args.data, f"road_{idx}.npy")).astype(np.float32)
        cond = torch.from_numpy(cond_full).unsqueeze(0).to(device)

        with torch.no_grad():
            z = ddpm.sample_ddim(net, cond, n_steps=args.steps, guidance_scale=args.guidance)
            road_pred = vae.decode(z)[0]                     # (5, H, W)

        raw_idx = road_pred.argmax(0).cpu().numpy()
        gt_idx = torch.from_numpy(road_np).argmax(0).numpy()
        vec_idx, G = vectorize_layout(
            road_pred,
            dilate_radius=args.dilate,
            min_path_px=args.min_path_px,
            merge_px=args.merge_px,
            dp_tolerance=args.dp_tol,
        )

        axes[i][0].imshow(cond_full[0], cmap="terrain")
        axes[i][0].set_title(f"Elevation ({idx})"); axes[i][0].axis("off")
        axes[i][1].imshow(onehot_to_rgb(gt_idx))
        axes[i][1].set_title("GT"); axes[i][1].axis("off")
        axes[i][2].imshow(onehot_to_rgb(raw_idx))
        axes[i][2].set_title("Raw pred"); axes[i][2].axis("off")
        axes[i][3].imshow(onehot_to_rgb(vec_idx))
        axes[i][3].set_title(f"Vectorized ({len(G)} nodes, {G.number_of_edges()} edges)")
        axes[i][3].axis("off")
        print(f"  [{i+1}/{args.n}] tile {idx}: "
              f"raw road px={int((raw_idx>0).sum())}, "
              f"vec road px={int((vec_idx>0).sum())}, "
              f"graph={len(G)} nodes, {G.number_of_edges()} edges")

    plt.tight_layout()
    out_path = os.path.join(args.output, "samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
