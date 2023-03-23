from typing import List, Optional, Tuple

import pygraphblas as pb


def bfs(graph: pb.Matrix, start: int) -> List[int]:
    """
    Provide graph breadth first search algorithm using matrix operations

    Parameters
    ----------
    graph: pb.Matrix
        adjacency matrix for graph
    start: int
        number of start vertex

    Returns
    -------
    vertices: List[int]
        array with the step number at which each vertex was visited
    """

    # some checks
    if (
        start is None or graph.nrows == 0
    ):  # start vertex undefined, e.g. for empty graph
        return []
    if not graph.square:
        raise Exception("Adjacency matrix must be square")
    if not (0 <= start < graph.nrows):
        raise Exception("Incorrect number of start vertex. Number out of range.")

    # algorithm
    vertices = pb.Vector.sparse(pb.INT64, size=graph.nrows)
    vertices[start] = 0
    visited = pb.Vector.dense(pb.types.BOOL, size=graph.nrows)
    visited[start] = True

    step = 1
    while visited.reduce():
        visited.vxm(graph, out=visited, mask=vertices.S, desc=pb.descriptor.RC)
        vertices.assign_scalar(step, mask=visited)
        step += 1

    return [vertices.get(i, default=-1) for i in range(vertices.size)]


def msbfs(graph: pb.Matrix, starts: List[int]) -> List[Tuple[int, List[int]]]:
    """
    Provide graph multi source breadth first search algorithm using matrix operations

    Parameters
    ----------
    graph: pb.Matrix
        adjacency matrix for graph
    starts: List[int]
        List of start vertices

    Returns
    -------
    vertices: List[int]
        array with the step number at which each vertex was visited
    """

    # some checks
    if (
        starts is None or graph.nrows == 0
    ):  # start vertex undefined, e.g. for empty graph
        return []
    if not graph.square:
        raise Exception("Adjacency matrix must be square")

    graph = graph.nonzero()

    parents = pb.Matrix.sparse(pb.INT64, nrows=len(starts), ncols=graph.ncols)
    front = pb.Matrix.sparse(pb.INT64, nrows=len(starts), ncols=graph.ncols)
    for row, start in enumerate(starts):
        if not (0 <= start < graph.nrows):
            raise Exception("Incorrect number of start vertex. Number out of range.")
        parents[row, start] = -1
        front[row, start] = start

    while front.nvals > 0:
        front.mxm(
            graph,
            out=front,
            semiring=pb.INT64.MIN_FIRST,
            mask=parents.S,
            desc=pb.descriptor.RC,
        )
        parents.assign(front, mask=front.S)
        front.apply(pb.INT64.POSITIONJ, out=front, mask=front.S)

    parents.assign_scalar(-2, mask=parents, desc=pb.descriptor.S & pb.descriptor.C)
    return [(starts[i], list(parents[i, :].vals)) for i in range(len(starts))]
