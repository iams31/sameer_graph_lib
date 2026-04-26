from sameer_graph_lib import HexGraph, SpatialIngestor
from sameer_graph_lib._h3 import get_resolution


def test_latlng_route_ingestion_and_graph_stats():
    ingestor = SpatialIngestor(resolution=9)
    route = ingestor.ingest_latlng_sequence(
        [
            (12.9716, 77.5946),
            (12.9740, 77.5970),
            (12.9760, 77.5990),
        ]
    )

    graph = HexGraph(hex_resolution=9)
    graph.add_route(route)
    graph.add_route(route)

    stats = graph.get_graph_stats()
    assert stats["num_nodes"] >= 1
    assert stats["total_count"] == len(route) * 2
    assert stats["route_count"] == 2
    assert all(data.get("kind") == "attachment" for _, _, data in graph.graph.edges(data=True))
    assert graph.get_route_affinity_score(route, route) == 1.0


def test_branch_decomposition_shape():
    graph = HexGraph(hex_resolution=9)
    route = graph.add_latlng_sequence(
        [
            (12.9716, 77.5946),
            (12.9740, 77.5970),
            (12.9760, 77.5990),
        ]
    )

    result = graph.decompose_branches(seed_hexes=route)
    assert "main_branch" in result
    assert "minor_branches" in result


def test_latlng_input_uses_requested_hex_resolution():
    graph = HexGraph(hex_resolution=8)
    route = graph.add_latlng_sequence(
        [
            (12.9716, 77.5946),
            (12.9760, 77.5990),
        ]
    )

    assert route
    assert all(get_resolution(cell) == 8 for cell in route)
    assert graph.get_graph_stats()["num_nodes"] == len(set(route))


def test_polyline_input_converts_to_hex_array_before_insert():
    graph = HexGraph(hex_resolution=4)
    route = graph.add_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@", resolution=4)

    assert route
    assert all(get_resolution(cell) == 4 for cell in route)
    assert graph.get_graph_stats()["total_count"] == len(route)
