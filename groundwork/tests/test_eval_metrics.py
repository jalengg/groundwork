import networkx as nx
from model.eval_metrics import compute_connectivity_index, compute_transport_convenience


def test_connectivity_index_on_grid():
    G = nx.grid_2d_graph(3, 3)
    ci = compute_connectivity_index(G)
    # 3x3 grid: corners deg=2, edges deg=3, center deg=4
    # avg = (4*2 + 4*3 + 1*4) / 9 = 24/9 ≈ 2.67
    assert 2.0 < ci < 4.0


def test_transport_convenience_on_known_graph():
    G = nx.path_graph(5)  # straight line: 0-1-2-3-4
    # For nodes 0 and 4: euclidean dist = 4, shortest path = 4 → ratio = 1.0
    tc = compute_transport_convenience(G)
    assert 0.5 < tc <= 1.0
