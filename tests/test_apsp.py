import pytest
import pygraphblas as pb
from project.path_finding import apsp
from project.utils import weight_graph_from_edges_list


@pytest.mark.parametrize(
    "n, edges, ans",
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
            [
                [0, 3, 10, 8, 4, 5, 10],
                [14, 0, 7, 12, 1, 2, 7],
                [2147483647, 2147483647, 0, 2147483647, 2147483647, 5, 2147483647],
                [2, 5, 4, 0, 6, 7, 12],
                [2147483647, 2147483647, 6, 2147483647, 0, 1, 2147483647],
                [2147483647, 2147483647, 5, 2147483647, 2147483647, 0, 2147483647],
                [7, 10, 1, 5, 8, 6, 0],
            ],
        ),
        (
            3,
            [(0, 10, 1), (1, 10, 2), (0, 30, 2)],
            [
                [0, 10, 20],
                [2147483647, 0, 10],
                [2147483647, 2147483647, 0],
            ],
        ),
    ],
)
def test_apsp(n, edges, ans):
    graph = weight_graph_from_edges_list(
        n, edges, val_type=pb.types.INT64, is_undirected=False
    )
    assert apsp(graph) == ans
