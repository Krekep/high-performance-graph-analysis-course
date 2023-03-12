from typing import List, Optional

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
    vertices: List[int] = [-1] * graph.nrows
    vertices[0] = 0
    visited = pb.Vector.dense(pb.types.BOOL, size=graph.nrows)
    visited[0] = True

    prev_visited_size = 0
    curr_visited_size = len(visited.nonzero())
    step = 1
    while prev_visited_size != curr_visited_size:
        prev_visited_size = curr_visited_size
        visited = visited + visited @ graph

        for vertex, _ in visited.nonzero():
            if vertices[vertex] == -1:
                vertices[vertex] = step

        curr_visited_size = len(visited.nonzero())
        step += 1

    return vertices
