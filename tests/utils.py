from typing import List, Tuple
import pygraphblas as pb


def graph_from_edges_list(
    n: int, edges: List[Tuple[int, int]], is_undirected: bool = True
) -> pb.Matrix:
    """

    Parameters
    ----------
    n: int
        Count of vertices
    edges: List[Tuple[int, int]]
        List of edges (from, to)
    is_undirected: bool
        Is edges undirected
    Returns
    -------
    adj: pb.Matrix
        Adjacency matrix for passed edges
    """

    adj = pb.Matrix.sparse(pb.types.BOOL, nrows=n, ncols=n)
    for edge in edges:
        adj[edge[0], edge[1]] = True
        if is_undirected:
            adj[edge[1], edge[0]] = True

    return adj
