import math
from typing import List

import pygraphblas as pb


def count_triangles_for_vertex(graph: pb.Matrix) -> List[int]:
    """
    Counts the number of triangles for each vertex

    Parameters
    ----------
    graph: pb.Matrix
        adjacency matrix for graph

    Returns
    -------
    vertices: List[int]
        array with the number of triangles for each vertex
    """

    squared = graph.mxm(graph, cast=pb.INT64, mask=graph)
    triangles = squared.reduce_vector()
    return [math.ceil(triangles.get(i, default=0) / 2) for i in range(triangles.size)]


def cohen_triangles(graph: pb.Matrix) -> int:
    """
    Counts the number of triangles in a graph using Cohen's algorithm

    Parameters
    ----------
    graph: pb.Matrix
        adjacency matrix for graph

    Returns
    -------
    cnt: int
        number of unique triangles in a graph
    """
    graph = graph.nonzero()

    low, upp = graph.tril(), graph.triu()
    cnt = low.mxm(upp, cast=pb.INT64, mask=graph)
    return math.ceil(cnt.reduce_int() / 2)


def sandia_triangles(graph: pb.Matrix) -> int:
    """
    Counts the number of triangles in a graph using Sandia's algorithm

    Parameters
    ----------
    graph: pb.Matrix
        adjacency matrix for graph

    Returns
    -------
    cnt: int
        number of unique triangles in a graph
    """

    low = graph.tril()
    cnt = low.mxm(low, cast=pb.INT64, mask=low)
    return cnt.reduce_int()
