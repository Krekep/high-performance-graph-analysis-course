import pygraphblas as pb
import numpy as np


def sssp(matrix: pb.Matrix, start_node: int):
    """
    Single Source Shortest Path using Bellman-Ford algorithm (SSSP)

    Parameters
    ----------
    matrix: pb.Matrix
        Graph adjacency matrix
    start_node: int
        Search start node

    Returns
    -------
    D: list
        Distances list D,
        D[i] == -1 if i-th node is unreachable from start
    """

    int_max = np.iinfo(np.int32).max

    d = pb.Vector.dense(matrix.type, size=matrix.nrows, fill=int_max)
    d[start_node] = 0

    for k in range(1, d.size - 1 + 1):  # from 1 to n - 1
        d.min_plus(matrix, out=d, accum=pb.INT64.min)

    d_cycle = d.dup()
    d_cycle.min_plus(matrix, out=d_cycle, accum=pb.INT64.min)
    if d_cycle.isne(d):
        print("There is a negative cycle")

    # d.assign_scalar(-1, mask=(d == int_max))
    result = d.to_lists()[1]

    return result


def mssp(matrix: pb.Matrix, start_nodes: list):
    """
    All-Pairs Shortest Path using Floyd-Warshall algorithm (APSP)

    Parameters
    ----------
    matrix: pb.Matrix
        Graph adjacency matrix
    start_nodes: list
        Start source for path MSSP

    Returns
    -------
    D: list[tuple[int, list]]
        Distances list D,
        D[i] == -1 if i-th node is unreachable from start
    """
    int_max = np.iinfo(np.int32).max

    d = pb.Matrix.dense(
        matrix.type, nrows=matrix.nrows, ncols=matrix.ncols, fill=int_max
    )
    for i, j in enumerate(start_nodes):
        d[i, j] = 0

    for k in range(1, matrix.nrows - 1 + 1):  # from 1 to n - 1
        d.min_plus(matrix, out=d, accum=pb.INT64.min)

    result = []
    for i, node in enumerate(start_nodes):
        line = d[i]
        # line.assign_scalar(-1, mask=(d[i] == int_max))
        result.append((node, list(line.vals)))

    return result


def apsp(matrix: pb.Matrix):
    """
    All-Pairs Shortest Path using Floyd-Warshall algorithm (APSP)

    Parameters
    ----------
    matrix: pb.Matrix
        Graph adjacency matrix

    Returns
    -------
    D: list[list]
        Graph squared matrix, where D[i, j] = shortest_path_between_i_j
    """

    int_max = np.iinfo(np.int32).max

    num_iterations = int(np.ceil(np.log2(matrix.nrows)))
    nrows, ncols = matrix.nrows, matrix.ncols

    d = pb.Matrix.dense(matrix.type, nrows=nrows, ncols=ncols, fill=int_max)

    for i in range(nrows):
        d[i, i] = 0

    rows, cols, vals = matrix.to_lists()
    for i, j, w in zip(rows, cols, vals):
        d[i, j] = w

    for k in range(1, num_iterations + 1):
        d.min_plus(d, out=d, accum=pb.INT64.min)

    # d.assign_scalar(-1, mask=(d == int_max))
    result = _matrix_to_list_of_lists(d)

    return result


def _matrix_to_list_of_lists(matrix: pb.Matrix):
    """
    Transform pygraphblas matrix to python list of lists matrix

    Parameters
    ----------
    matrix: pb.Matrix
        Matrix

    Returns
    -------
    M: list[list]
        Transformed matrix
    """

    nrows, ncols = matrix.nrows, matrix.ncols
    rows, cols, vals = matrix.to_lists()
    result = []
    for _ in range(nrows):
        result.append([0] * ncols)

    for i, j, w in zip(rows, cols, vals):
        result[i][j] = w

    return result
