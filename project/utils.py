from typing import List, Tuple
import pygraphblas as pb
import networkx as nx


def graph_from_edges_list(
    n: int, edges: List[Tuple[int, int]], is_undirected: bool = True
) -> pb.Matrix:
    """
    Build adjacency matrix from list of edges

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


def read_graph(path: str) -> nx.MultiDiGraph:
    """
    Read graph from file

    Parameters
    ----------
    path: str
        Path to graph (include *name.dot*)

    Returns
    -------
    graph: nx.MultiDiGraph
    """
    return nx.nx_agraph.read_dot(path)


def graph_to_ajd_matrix(graph: nx.MultiDiGraph) -> pb.Matrix:
    """
    Transform nx.MultiDiGraph to adjacency matrix

    Parameters
    ----------
    graph: nx.MultiDiGraph

    Returns
    -------
    adj: pb.Matrix
    """
    adj_matrix = pb.Matrix.sparse(
        pb.BOOL, graph.number_of_nodes(), graph.number_of_nodes()
    )
    for (source, target) in graph.edges():
        adj_matrix[int(source), int(target)] = True
    return adj_matrix
