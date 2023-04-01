import pytest
import pygraphblas as pb
from project.path_finding import sssp
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
            [0, 3, 10, 8, 4, 5, 10],
        ),
        (
            3,
            [(0, 10, 1), (1, 10, 2), (0, 30, 2)],
            [0, 10, 20],
        ),
    ],
)
def test_sssp(n, edges, ans):
    graph = weight_graph_from_edges_list(
        n, edges, val_type=pb.types.INT64, is_undirected=False
    )
    assert sssp(graph, 0) == ans
