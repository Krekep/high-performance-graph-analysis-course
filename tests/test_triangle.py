import pytest

from project.triangle import (
    count_triangles_for_vertex,
    cohen_triangles,
    sandia_triangles,
)
from project.utils import graph_from_edges_list


@pytest.mark.parametrize(
    "n, edges, ans",
    [
        (
            2,
            [(0, 1)],
            [0, 0],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 0)],
            [1, 1, 1, 0],
        ),
        (
            5,
            [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)],
            [1, 1, 2, 1, 1],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3), (3, 0), (2, 0), (1, 3)],
            [3, 3, 3, 3],
        ),
    ],
)
def test_vertex_to_triangles_count(n, edges, ans):
    graph = graph_from_edges_list(n, edges)
    assert count_triangles_for_vertex(graph) == ans


@pytest.mark.parametrize(
    "n, edges, ans",
    [
        (
            2,
            [(0, 1)],
            0,
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 0)],
            1,
        ),
        (
            5,
            [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)],
            2,
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3), (3, 0), (2, 0), (1, 3)],
            4,
        ),
    ],
)
def test_triangles(n, edges, ans):
    graph = graph_from_edges_list(n, edges)
    cohen_res = cohen_triangles(graph)
    sandia_res = sandia_triangles(graph)
    assert cohen_res == ans
    assert sandia_res == ans
