import pytest
from src.dag import (
    CausalGraph,
    create_classic_confounding_dag,
    create_mediation_dag,
    create_m_bias_dag,
    create_instrumental_variable_dag
)


def test_causal_graph_identification():
    dag = CausalGraph(
        treatment="Drug",
        outcome="Recovery",
        confounders=["Age", "Severity"],
        mediators=["Enzyme"],
        colliders=["Hospitalization"]
    )
    
    # 1. Perfectly identified set
    check_ok = dag.validate_variable_roles(selected_covariates=["Age", "Severity"])
    assert check_ok["is_identified"] is True
    assert len(check_ok["warnings"]) == 0

    # 2. Missing confounder
    check_miss = dag.validate_variable_roles(selected_covariates=["Age"])
    assert check_miss["is_identified"] is False
    assert "Severity" in check_miss["missing_confounders"]

    # 3. Bad control: conditioning on mediator
    check_med = dag.validate_variable_roles(selected_covariates=["Age", "Severity", "Enzyme"])
    assert check_med["is_identified"] is False
    assert "Enzyme" in check_med["conditioned_mediators"]

    # 4. Bad control: conditioning on collider
    check_col = dag.validate_variable_roles(selected_covariates=["Age", "Severity", "Hospitalization"])
    assert check_col["is_identified"] is False
    assert "Hospitalization" in check_col["conditioned_colliders"]


def test_causal_graph_mermaid():
    dag = CausalGraph(
        treatment="T",
        outcome="Y",
        confounders=["C1"],
        mediators=["M1"],
        colliders=["K1"]
    )
    spec = dag.generate_mermaid_spec(selected_covariates=["C1"])
    assert "graph LR" in spec
    assert "C1 --> T" in spec
    assert "T --> M1" in spec
    assert "T ==>|Causal Effect tau| Y" in spec


def test_node_and_edge_management():
    dag = CausalGraph(treatment="A", outcome="B")
    assert "A" in dag._graph.nodes
    assert "B" in dag._graph.nodes
    assert ("A", "B") in dag._graph.edges

    # Add node
    assert dag.add_node("C", role="confounder") is True
    assert "C" in dag._graph.nodes
    assert dag.node_meta["C"]["role"] == "confounder"

    # Add valid edge
    success, msg = dag.add_edge("C", "A")
    assert success is True
    assert ("C", "A") in dag._graph.edges

    # Remove edge
    assert dag.remove_edge("C", "A") is True
    assert ("C", "A") not in dag._graph.edges

    # Remove node
    assert dag.remove_node("C") is True
    assert "C" not in dag._graph.nodes


def test_cycle_detection_and_prevention():
    dag = CausalGraph(treatment="X", outcome="Y")
    dag.add_node("Z")
    dag.add_edge("X", "Z")
    dag.add_edge("Z", "Y")

    # 1. Prohibit self-loops
    ok_self, msg_self, cycle_self = dag.validate_edge("X", "X")
    assert ok_self is False
    assert "Self-loops" in msg_self
    assert cycle_self == ["X", "X"]

    # 2. Attempt to add direct 2-node cycle (Y -> X when X -> Y exists)
    ok_2, msg_2, cycle_2 = dag.validate_edge("Y", "X")
    assert ok_2 is False
    assert "Acyclicity Violation" in msg_2
    assert cycle_2 == ["Y", "X", "Y"]

    # 3. Attempt to add 3-node cycle (Y -> X via Z: X -> Z -> Y -> X)
    ok_3, msg_3, cycle_3 = dag.validate_edge("Y", "Z")
    assert ok_3 is False
    assert "Acyclicity Violation" in msg_3

    # Graph remains acyclic
    assert dag.is_acyclic() is True
    assert len(dag.find_cycles()) == 0


def test_backdoor_paths_and_adjustment_discovery():
    # Confounded Triangle: X causes both T and Y
    dag = create_classic_confounding_dag()
    assert dag.is_acyclic() is True

    # Backdoor paths without conditioning
    bd_unadj = dag.analyze_backdoor_paths([])
    assert len(bd_unadj) == 1
    assert bd_unadj[0]["path"] == ["Treatment", "Confounder_X", "Outcome"]
    assert bd_unadj[0]["is_blocked"] is False
    assert bd_unadj[0]["status"] == "OPEN"

    # Backdoor paths with X conditioned
    bd_adj = dag.analyze_backdoor_paths(["Confounder_X"])
    assert len(bd_adj) == 1
    assert bd_adj[0]["is_blocked"] is True
    assert bd_adj[0]["status"] == "BLOCKED"

    # Automated adjustment set discovery
    adj_info = dag.find_valid_adjustment_sets()
    assert adj_info["is_identified"] is True
    assert ["Confounder_X"] in adj_info["minimal_sets"]
    assert adj_info["recommended_set"] == ["Confounder_X"]


def test_m_bias_collider_path():
    # In M-Bias: U1 -> T, U1 -> C <- U2, U2 -> Y
    dag = create_m_bias_dag()
    assert dag.is_acyclic() is True

    # Without conditioning on collider C, the path is naturally blocked at collider C!
    bd_unadj = dag.analyze_backdoor_paths([])
    # Backdoor path: Treatment <- U1 -> C <- U2 -> Outcome
    m_path = [p for p in bd_unadj if "Collider_C" in p["path"]]
    assert len(m_path) == 1
    assert m_path[0]["is_blocked"] is True  # Blocked by default because C is a collider!

    # Conditioning on Collider C OPENS the spurious path!
    bd_adj_collider = dag.analyze_backdoor_paths(["Collider_C"])
    m_path_opened = [p for p in bd_adj_collider if "Collider_C" in p["path"]]
    assert len(m_path_opened) == 1
    assert m_path_opened[0]["is_blocked"] is False  # Path is now OPEN!


def test_assumptions_register_and_guardrails():
    dag = create_mediation_dag()
    audit = dag.generate_assumptions_register()

    assert audit["total_nodes"] >= 4
    assert audit["total_edges"] >= 4
    assert len(audit["direct_causal_assumptions"]) >= 4
    assert "CRITICAL SCIENTIFIC EPISTEMOLOGY" in audit["critical_warning"]
    assert "CANNOT verify edge direction" in audit["critical_warning"]


def test_serialization_and_plotly():
    dag = create_classic_confounding_dag()
    data = dag.to_dict()
    assert data["treatment"] == "Treatment"
    assert data["outcome"] == "Outcome"
    assert data["is_acyclic"] is True

    # Reconstruct
    rebuilt = CausalGraph.from_dict(data)
    assert rebuilt.treatment == "Treatment"
    assert rebuilt.outcome == "Outcome"
    assert rebuilt.is_acyclic() is True

    # Interactive Plotly figure
    fig = dag.generate_plotly_figure(selected_covariates=["Confounder_X"])
    assert fig is not None
    assert len(fig.data) >= 1
