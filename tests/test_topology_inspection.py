from fascia_robot.experiments.fascia_topology_inspection import build_current_topology, topology_document


def test_inspection_preserves_current_topology_and_classifies_every_edge():
    _, topology, _, materials = build_current_topology()
    document = topology_document(topology, materials)
    assert document["counts"]["nodes"] == len(topology.nodes) == 412
    assert document["counts"]["edges"] == len(topology.edges) == 1510
    assert len(document["edges"]) == len(topology.edges)
    assert all(edge["anatomical_category"] and edge["routing_category"] for edge in document["edges"])


def test_structural_fill_and_spiral_status_are_explicit():
    _, topology, _, materials = build_current_topology()
    document = topology_document(topology, materials)
    fill = [edge for edge in document["edges"] if edge["structural_fill"]]
    expected = sum(edge.edge_class == "superficial_interface" or
                   ("diagonal" in edge.edge_class and edge.layer != "reinforced")
                   for edge in topology.edges)
    assert len(fill) == expected
    assert not document["explicit_spiral_helical_pathway_present"]
    assert not any(edge["routing_category"] == "spiral_helical" for edge in document["edges"])


def test_region_catalog_covers_every_node_region_and_has_required_fields():
    _, topology, _, materials = build_current_topology()
    document = topology_document(topology, materials)
    assert {r["region_name"] for r in document["regions"]} == {n.region for n in topology.nodes}
    for region in document["regions"]:
        assert region["node_count"] > 0
        assert region["fiber_orientations"]
        assert region["material_stiffness_classes"]
        assert region["pathway_types"]
        assert region["anatomical_rationale"]
