import pytest
import pygraphblas as pb
from project.path_finding import mssp
from project.utils import weight_graph_from_edges_list


@pytest.mark.parametrize(
    "n, edges, start_nodes, ans",
    [
        (
            7,  # graph from lecture. weights *= 10
            [
                (0, 3, 1),
                (0, 8, 3),
                (1, 1, 4),
                (1, 7, 6),
                (2, 5, 5),
                (3, 2, 0),
                (3, 4, 2),
                (4, 1, 5),
                (5, 5, 2),
                (6, 1, 2),
                (6, 5, 3),
                (6, 8, 4),
            ],
            [0],
            [(0, [0, 3, 10, 8, 4, 5, 10])],
        ),
        (
            3,
            [(0, 10, 1), (1, 10, 2), (0, 30, 2)],
            [0],
            [(0, [0, 10, 20])],
        ),
        (
            3,
            [(0, 10, 1), (1, 10, 2), (0, 30, 2)],
            [0, 2],
            [
                (0, [0, 10, 20]),
                (2, [2147483647, 2147483647, 0]),
            ],
        ),
    ],
)
def test_mssp(n, edges, start_nodes, ans):
    graph = weight_graph_from_edges_list(
        n, edges, val_type=pb.types.INT64, is_undirected=False
    )
    assert mssp(graph, start_nodes) == ans
