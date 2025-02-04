import pytest

from project import matrix_bfs
from project.utils import graph_from_edges_list


_metaparams = [
    (graph_from_edges_list(3, [(0, 1), (1, 2)]), 0, [0, 1, 2]),
    (
        graph_from_edges_list(4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
        0,
        [0, 1, 2, 1],
    ),
    (graph_from_edges_list(2, []), 0, [0, -1]),
    (
        graph_from_edges_list(
            3,
            [
                (0, 1),
                (1, 2),
                (2, 0),
            ],
            is_undirected=True,
        ),
        0,
        [0, 1, 1],
    ),
    (
        graph_from_edges_list(
            3,
            [
                (0, 1),
                (1, 2),
                (2, 0),
            ],
            is_undirected=False,
        ),
        0,
        [0, 1, 2],
    ),
    (graph_from_edges_list(0, []), None, []),
]


def test_bfs():
    for params in _metaparams:
        graph = params[0]
        start = params[1]
        expected = params[2]

        actual = matrix_bfs.bfs(graph, start)
        assert actual == expected
