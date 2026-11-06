"""
Structural Causal Model & Interactive Directed Acyclic Graph (DAG) Engine.
Implements Judea Pearl's Causal Graphs (Pearl 2000, 2009), d-separation,
backdoor criterion path analysis, cycle detection, and adjustment set discovery.
"""

from itertools import combinations
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
import numpy as np
import plotly.graph_objects as go


class CausalGraph:
    """
    Structural Causal Model / Directed Acyclic Graph (DAG) Engine.
    
    Supports:
    - Node creation, removal, and role classification (Treatment, Outcome, Confounder, Mediator, Collider, Instrument).
    - Causal edge connection and strict cycle detection (preventing feedback loops).
    - D-separation and comprehensive backdoor path tracking (open vs. blocked paths).
    - Automated derivation of minimal and sufficient backdoor adjustment sets.
    - Explicit assumptions register distinguishing untestable structural causal assumptions
      from empirical statistical correlations.
    - Interactive Plotly network visualization and Mermaid flowchart export.
    """

    def __init__(
        self,
        treatment: Optional[str] = "Treatment",
        outcome: Optional[str] = "Outcome",
        confounders: Optional[List[str]] = None,
        mediators: Optional[List[str]] = None,
        colliders: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str]]] = None,
    ):
        self.treatment = treatment or "Treatment"
        self.outcome = outcome or "Outcome"
        self.confounders = list(confounders or [])
        self.mediators = list(mediators or [])
        self.colliders = list(colliders or [])

        # Internal NetworkX DiGraph
        self._graph = nx.DiGraph()

        # Node metadata: {name: {"role": str, "observed": bool, "description": str}}
        self.node_meta: Dict[str, Dict[str, Any]] = {}

        # Add initial focal nodes
        self.add_node(self.treatment, role="treatment", observed=True)
        self.add_node(self.outcome, role="outcome", observed=True)

        for c in self.confounders:
            self.add_node(c, role="confounder", observed=True)
        for m in self.mediators:
            self.add_node(m, role="mediator", observed=True)
        for col in self.colliders:
            self.add_node(col, role="collider", observed=True)

        # Populate edges
        if edges:
            for u, v in edges:
                if u not in self._graph:
                    self.add_node(u, role="covariate")
                if v not in self._graph:
                    self.add_node(v, role="covariate")
                self.add_edge(u, v)
        else:
            # Build default causal canonical structure if provided
            self.add_edge(self.treatment, self.outcome)
            for c in self.confounders:
                self.add_edge(c, self.treatment)
                self.add_edge(c, self.outcome)
            for m in self.mediators:
                self.add_edge(self.treatment, m)
                self.add_edge(m, self.outcome)
            for col in self.colliders:
                self.add_edge(self.treatment, col)
                self.add_edge(self.outcome, col)

    # -------------------------------------------------------------------------
    # Node & Edge Management
    # -------------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        role: str = "covariate",
        observed: bool = True,
        description: str = ""
    ) -> bool:
        """Add a variable node with role and metadata."""
        clean_name = str(name).strip()
        if not clean_name:
            return False

        self._graph.add_node(clean_name)
        self.node_meta[clean_name] = {
            "role": role,
            "observed": observed,
            "description": description or f"Variable {clean_name}"
        }
        return True

    def remove_node(self, name: str) -> bool:
        """Remove a variable node and all attached causal edges."""
        if name not in self._graph:
            return False

        self._graph.remove_node(name)
        self.node_meta.pop(name, None)

        if name in self.confounders:
            self.confounders.remove(name)
        if name in self.mediators:
            self.mediators.remove(name)
        if name in self.colliders:
            self.colliders.remove(name)

        return True

    def set_treatment(self, name: str) -> None:
        """Set treatment focal node."""
        if name not in self._graph:
            self.add_node(name, role="treatment")
        self.treatment = name
        self.node_meta[name]["role"] = "treatment"

    def set_outcome(self, name: str) -> None:
        """Set outcome focal node."""
        if name not in self._graph:
            self.add_node(name, role="outcome")
        self.outcome = name
        self.node_meta[name]["role"] = "outcome"

    def set_node_role(self, name: str, role: str) -> None:
        """Assign or update a node's causal role."""
        if name in self.node_meta:
            self.node_meta[name]["role"] = role
            if role == "confounder" and name not in self.confounders:
                self.confounders.append(name)
            elif role == "mediator" and name not in self.mediators:
                self.mediators.append(name)
            elif role == "collider" and name not in self.colliders:
                self.colliders.append(name)

    def get_node_role(self, name: str) -> str:
        """Retrieve the assigned causal role of a node."""
        if name == self.treatment:
            return "treatment"
        if name == self.outcome:
            return "outcome"
        if name in self.node_meta:
            return self.node_meta[name].get("role", "covariate")
        if name in self.confounders:
            return "confounder"
        if name in self.mediators:
            return "mediator"
        if name in self.colliders:
            return "collider"
        return "covariate"

    # -------------------------------------------------------------------------
    # Cycle Detection & Edge Validation
    # -------------------------------------------------------------------------

    def validate_edge(self, source: str, target: str) -> Tuple[bool, str, List[str]]:
        """
        Validate whether adding source -> target maintains directed acyclicity.

        Returns:
            Tuple of (is_valid: bool, message: str, cycle: List[str])
        """
        if source == target:
            return False, f"Self-loops ({source} -> {source}) are strictly prohibited in causal DAGs.", [source, source]

        if source not in self._graph:
            return False, f"Source node '{source}' does not exist in graph.", []
        if target not in self._graph:
            return False, f"Target node '{target}' does not exist in graph.", []

        if self._graph.has_edge(source, target):
            return False, f"Causal edge '{source} -> {target}' already exists.", []

        # Check if target already has a directed path to source
        if nx.has_path(self._graph, target, source):
            # Find the path that would form the cycle
            path = nx.shortest_path(self._graph, target, source)
            cycle = [source] + path
            cycle_str = " -> ".join(cycle)
            msg = (
                f"Acyclicity Violation: Adding edge '{source} -> {target}' creates a causal feedback cycle: "
                f"[{cycle_str}]. In causal inference (Pearl 2009), DAGs must be strictly acyclic."
            )
            return False, msg, cycle

        return True, "Valid causal edge.", []

    def add_edge(self, source: str, target: str) -> Tuple[bool, str]:
        """
        Add a directed causal edge from source to target with cycle validation.
        """
        if source not in self._graph:
            self.add_node(source, role="covariate")
        if target not in self._graph:
            self.add_node(target, role="covariate")

        is_valid, msg, _ = self.validate_edge(source, target)
        if not is_valid:
            return False, msg

        self._graph.add_edge(source, target)
        return True, f"Causal edge '{source} -> {target}' successfully added."

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove a directed causal edge."""
        if self._graph.has_edge(source, target):
            self._graph.remove_edge(source, target)
            return True
        return False

    def is_acyclic(self) -> bool:
        """Verify whether the graph is currently a valid DAG."""
        return nx.is_directed_acyclic_graph(self._graph)

    def find_cycles(self) -> List[List[str]]:
        """Find all directed cycles in the graph (empty if acyclic)."""
        try:
            return list(nx.simple_cycles(self._graph))
        except Exception:
            return []

    def topological_sort(self) -> List[str]:
        """Return a topological ordering of variables (causal time order)."""
        if not self.is_acyclic():
            return []
        return list(nx.topological_sort(self._graph))

    # -------------------------------------------------------------------------
    # Kinship & Causal Relations (Pearl 2009)
    # -------------------------------------------------------------------------

    def parents(self, node: str) -> List[str]:
        """Direct causes of node."""
        if node not in self._graph:
            return []
        return list(self._graph.predecessors(node))

    def children(self, node: str) -> List[str]:
        """Direct effects of node."""
        if node not in self._graph:
            return []
        return list(self._graph.successors(node))

    def ancestors(self, node: str) -> Set[str]:
        """All upstream causes (direct and indirect)."""
        if node not in self._graph:
            return set()
        return nx.ancestors(self._graph, node)

    def descendants(self, node: str) -> Set[str]:
        """All downstream effects (direct and indirect)."""
        if node not in self._graph:
            return set()
        return nx.descendants(self._graph, node)

    # -------------------------------------------------------------------------
    # Backdoor Path & D-Separation Analysis
    # -------------------------------------------------------------------------

    def get_all_undirected_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all simple undirected paths between source and target."""
        if source not in self._graph or target not in self._graph:
            return []
        undirected = self._graph.to_undirected()
        try:
            return list(nx.all_simple_paths(undirected, source, target))
        except Exception:
            return []

    def is_path_backdoor(self, path: List[str], treatment: str) -> bool:
        """
        Check if an undirected path from treatment to outcome is a backdoor path.
        A backdoor path starts with an arrow pointing INTO treatment (path[1] -> path[0]).
        """
        if len(path) < 2 or path[0] != treatment:
            return False
        first_step = path[1]
        # Does first_step point into treatment?
        return self._graph.has_edge(first_step, treatment)

    def is_path_blocked(self, path: List[str], conditioning_set: Set[str]) -> Tuple[bool, str]:
        """
        Evaluate d-separation for a single path given a conditioning set Z.
        
        Rules:
        1. A chain (i -> m -> j or i <- m <- j) or fork (i <- m -> j) is BLOCKED if m in Z.
        2. A collider (i -> c <- j) is BLOCKED if neither c nor any descendant of c is in Z.
           (Conditioning on a collider or its descendant OPENS the collider).
        
        A path is blocked if AT LEAST ONE node along the path blocks it.
        """
        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            curr_node = path[i]
            next_node = path[i + 1]

            # Check if curr_node is a collider along this path
            is_collider = self._graph.has_edge(prev_node, curr_node) and self._graph.has_edge(next_node, curr_node)

            if is_collider:
                # Collider is blocked if neither curr_node nor any of its descendants are in Z
                curr_descendants = self.descendants(curr_node) | {curr_node}
                if not (curr_descendants & conditioning_set):
                    return True, f"Collider '{curr_node}' is unconditioned (blocks path)"
            else:
                # Chain or Fork: blocked if curr_node is conditioned on
                if curr_node in conditioning_set:
                    return True, f"Non-collider '{curr_node}' is conditioned on (blocks path)"

        return False, "Path is open (active non-causal association flows)"

    def analyze_backdoor_paths(self, conditioning_set: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find and audit all backdoor paths from Treatment to Outcome.
        Classifies each path as OPEN or BLOCKED.
        """
        z_set = set(conditioning_set or [])
        paths = self.get_all_undirected_paths(self.treatment, self.outcome)
        backdoor_analysis = []

        for p in paths:
            if not self.is_path_backdoor(p, self.treatment):
                continue

            is_blocked, reason = self.is_path_blocked(p, z_set)

            # Build formatted string e.g. T <- X -> Y
            segments = []
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                if self._graph.has_edge(u, v):
                    segments.append(f"{u} ->")
                elif self._graph.has_edge(v, u):
                    segments.append(f"{u} <-")
                else:
                    segments.append(f"{u} --")
            segments.append(p[-1])
            path_str = " ".join(segments)

            backdoor_analysis.append({
                "path": p,
                "path_str": path_str,
                "is_blocked": is_blocked,
                "status": "BLOCKED" if is_blocked else "OPEN",
                "reason": reason
            })

        return backdoor_analysis

    # -------------------------------------------------------------------------
    # Backdoor Adjustment Set Discovery (Pearl 2009)
    # -------------------------------------------------------------------------

    def find_valid_adjustment_sets(self) -> Dict[str, Any]:
        """
        Derive valid backdoor adjustment sets for estimating the causal effect of T on Y.

        Criteria:
        1. No node in Z is a descendant of T (Z ∩ De(T) = ∅).
        2. Z blocks all backdoor paths between T and Y.

        Returns:
            Dict containing minimal adjustment sets, all valid sets, and recommendation.
        """
        if not self.is_acyclic():
            return {
                "is_identified": False,
                "minimal_sets": [],
                "all_valid_sets": [],
                "recommended_set": [],
                "message": "Graph contains directed cycles. Acyclicity must be restored before causal identification."
            }

        t_descendants = self.descendants(self.treatment)
        candidate_pool = [
            n for n in self._graph.nodes
            if n not in t_descendants and n != self.treatment and n != self.outcome
        ]

        valid_sets = []

        # Test empty set first
        bd_paths_empty = self.analyze_backdoor_paths([])
        if all(bp["is_blocked"] for bp in bd_paths_empty):
            valid_sets.append([])

        # Test combinations up to candidate_pool length
        for r in range(1, len(candidate_pool) + 1):
            for candidate in combinations(candidate_pool, r):
                candidate_set = set(candidate)
                bd_paths = self.analyze_backdoor_paths(list(candidate_set))
                if all(bp["is_blocked"] for bp in bd_paths):
                    valid_sets.append(sorted(list(candidate_set)))

        # Deduplicate
        unique_valid = []
        for s in valid_sets:
            if s not in unique_valid:
                unique_valid.append(s)

        # Find minimal adjustment sets (no proper subset is also valid)
        minimal_sets = []
        for s in unique_valid:
            s_set = set(s)
            has_subset = False
            for other in unique_valid:
                other_set = set(other)
                if other_set < s_set:
                    has_subset = True
                    break
            if not has_subset and s not in minimal_sets:
                minimal_sets.append(s)

        # Recommended set: parents of T if valid, otherwise smallest minimal set
        t_parents = sorted([p for p in self.parents(self.treatment) if p in candidate_pool])
        if t_parents in unique_valid:
            recommended = t_parents
        elif minimal_sets:
            recommended = sorted(minimal_sets, key=lambda x: len(x))[0]
        elif unique_valid:
            recommended = unique_valid[0]
        else:
            recommended = []

        is_identified = len(unique_valid) > 0

        return {
            "is_identified": is_identified,
            "minimal_sets": minimal_sets,
            "all_valid_sets": unique_valid,
            "recommended_set": recommended,
            "candidate_pool": candidate_pool,
            "message": (
                f"Causal effect identified via Backdoor Criterion. Recommended adjustment set: {recommended}"
                if is_identified
                else "Unidentified: No subset of observed variables blocks all backdoor paths without opening colliders."
            )
        }

    # -------------------------------------------------------------------------
    # Legacy & High-Level Validation
    # -------------------------------------------------------------------------

    def validate_variable_roles(self, selected_covariates: List[str]) -> Dict[str, Any]:
        """
        Validate whether the user's selected covariate set violates backdoor adjustment principles:
        1. Valid Confounders: X -> T and X -> Y. Adjusting for them blocks backdoor paths.
        2. Mediators: T -> M -> Y. Conditioning on mediators blocks the indirect path and biases total effect estimation.
        3. Colliders: T -> C <- Y. Conditioning on a collider opens a spurious non-causal path (Berkson's bias / collider stratification).
        """
        selected_set = set(selected_covariates or [])

        # Check for unblocked backdoor paths
        bd_paths = self.analyze_backdoor_paths(selected_covariates)
        unblocked_paths = [bp for bp in bd_paths if not bp["is_blocked"]]

        # Bad controls: conditioned mediators or colliders
        t_descendants = self.descendants(self.treatment)
        conditioned_mediators = [m for m in selected_set if m in t_descendants and m != self.outcome]
        conditioned_colliders = [c for c in self.colliders if c in selected_set]

        # Missing confounders (parents of both T and Y that are not conditioned)
        t_parents = set(self.parents(self.treatment))
        y_parents = set(self.parents(self.outcome))
        common_causes = (t_parents & y_parents) | set(self.confounders)
        missing_confounders = [c for c in common_causes if c not in selected_set and c != self.treatment and c != self.outcome]

        is_identified = (len(unblocked_paths) == 0) and (len(conditioned_mediators) == 0) and (len(conditioned_colliders) == 0)

        warnings = []
        if unblocked_paths:
            path_examples = [bp["path_str"] for bp in unblocked_paths[:2]]
            warnings.append(
                f"Open Backdoor Path Warning: Spurious non-causal association flows through: {', '.join(path_examples)}. "
                "Omitted confounding induces bias."
            )
        if missing_confounders:
            warnings.append(
                f"Omitted Confounder Warning: Common causes {missing_confounders} are unadjusted."
            )
        if conditioned_mediators:
            warnings.append(
                f"Over-Controlling / Mediator Warning: You conditioned on post-treatment variable(s) {conditioned_mediators}. "
                "Conditioning on a mediator blocks the indirect causal mechanism (T -> M -> Y) and biases total ATE."
            )
        if conditioned_colliders:
            warnings.append(
                f"Collider Stratification Warning: You conditioned on collider(s) {conditioned_colliders}. "
                "Conditioning on a collider opens a spurious association (Berkson's paradox / M-bias)."
            )

        adj_info = self.find_valid_adjustment_sets()

        return {
            "is_identified": is_identified,
            "sufficient_adjustment_set": adj_info.get("recommended_set", list(self.confounders)),
            "minimal_adjustment_sets": adj_info.get("minimal_sets", []),
            "selected_covariates": selected_covariates,
            "missing_confounders": missing_confounders,
            "conditioned_mediators": conditioned_mediators,
            "conditioned_colliders": conditioned_colliders,
            "unblocked_backdoor_paths": unblocked_paths,
            "warnings": warnings,
            "identification_status": (
                "Identified: Backdoor criterion satisfied." if is_identified
                else "Unidentified or Biased: Backdoor criterion violated."
            )
        }

    # -------------------------------------------------------------------------
    # Assumptions Register & Epistemology Audit
    # -------------------------------------------------------------------------

    def generate_assumptions_register(self) -> Dict[str, Any]:
        """
        Generate a comprehensive, scientifically rigorous audit of all structural assumptions
        embedded in this DAG.
        """
        edges_list = list(self._graph.edges())
        nodes_list = list(self._graph.nodes())

        # Direct causal claims
        direct_claims = [f"'{u}' has a direct causal effect on '{v}'." for u, v in edges_list]

        # Exclusion restrictions (absent edges)
        all_possible_pairs = [(u, v) for u in nodes_list for v in nodes_list if u != v]
        absent_pairs = [pair for pair in all_possible_pairs if pair not in self._graph.edges()]
        
        # Relevant exclusions between treatment, outcome, and mediators
        relevant_exclusions = []
        for u, v in absent_pairs:
            if u == self.treatment and v != self.outcome and v in self.mediators:
                relevant_exclusions.append(f"No direct arrow from '{u}' to '{v}'.")
            elif u != self.treatment and v == self.outcome and u in self.confounders:
                pass

        return {
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list),
            "direct_causal_assumptions": direct_claims,
            "exclusion_restrictions_count": len(absent_pairs),
            "critical_warning": (
                "CRITICAL SCIENTIFIC EPISTEMOLOGY (Pearl 2009):\n"
                "1. Directed edges in this graph represent substantive theoretical causal assumptions, "
                "NOT verified empirical statistical facts.\n"
                "2. Observational data alone CANNOT verify edge direction (e.g. distinguishing A -> B from B -> A) "
                "or rule out unobserved confounding (A <- U -> B).\n"
                "3. High correlation does NOT imply causation. If an arrow is misspecified or a common cause is omitted, "
                "all downstream causal effect estimates (OLS, Matching, IPW, AIPW, DML) will be biased."
            )
        }

    # -------------------------------------------------------------------------
    # Visualizations: Mermaid & Interactive Plotly
    # -------------------------------------------------------------------------

    def generate_mermaid_spec(self, selected_covariates: Optional[List[str]] = None) -> str:
        """Generate Mermaid flowchart syntax for rendering the DAG."""
        selected_set = set(selected_covariates or [])
        lines = [
            "graph LR",
            "    %% Nodes",
            f'    {self.treatment}["Treatment: {self.treatment}"]:::treatment',
            f'    {self.outcome}["Outcome: {self.outcome}"]:::outcome',
        ]

        for node in self._graph.nodes():
            if node in [self.treatment, self.outcome]:
                continue
            role = self.node_meta.get(node, {}).get("role", "covariate")
            if role == "confounder":
                style = "adjusted" if node in selected_set else "unadjusted"
                lines.append(f'    {node}["{node} (Confounder)"]:::{style}')
            elif role == "mediator":
                style = "danger" if node in selected_set else "mediator"
                lines.append(f'    {node}["{node} (Mediator)"]:::{style}')
            elif role == "collider":
                style = "danger" if node in selected_set else "collider"
                lines.append(f'    {node}["{node} (Collider)"]:::{style}')
            else:
                style = "adjusted" if node in selected_set else "neutral"
                lines.append(f'    {node}["{node}"]:::{style}')

        lines.append("    %% Causal Edges")
        for u, v in self._graph.edges():
            if u == self.treatment and v == self.outcome:
                lines.append(f"    {u} ==>|Causal Effect tau| {v}")
            else:
                lines.append(f"    {u} --> {v}")

        # Styles
        lines += [
            "    classDef treatment fill:#059669,stroke:#10B981,color:#fff,stroke-width:2px;",
            "    classDef outcome fill:#2563EB,stroke:#3B82F6,color:#fff,stroke-width:2px;",
            "    classDef adjusted fill:#0D9488,stroke:#2DD4BF,color:#fff,stroke-width:2px;",
            "    classDef unadjusted fill:#D97706,stroke:#F59E0B,color:#fff,stroke-width:2px,stroke-dasharray: 4 4;",
            "    classDef mediator fill:#D97706,stroke:#F97316,color:#fff,stroke-width:2px;",
            "    classDef collider fill:#DC2626,stroke:#EF4444,color:#fff,stroke-width:2px;",
            "    classDef danger fill:#B91C1C,stroke:#EF4444,color:#fff,stroke-width:3px;",
            "    classDef neutral fill:#1E293B,stroke:#64748B,color:#E2E8F0;"
        ]

        return "\n".join(lines)

    def generate_plotly_figure(self, selected_covariates: Optional[List[str]] = None) -> go.Figure:
        """
        Generate an interactive Plotly network diagram for exploring the DAG.
        Nodes are color-coded by causal role and adjustment status.
        """
        selected_set = set(selected_covariates or [])

        # Compute positions
        if len(self._graph.nodes) == 0:
            fig = go.Figure()
            fig.update_layout(title="Empty Causal Graph")
            return fig

        # Layered coordinate layout for intuitive causal flow (left-to-right)
        pos = {}
        t_descendants = self.descendants(self.treatment)
        confounder_nodes = [n for n in self._graph.nodes if n not in [self.treatment, self.outcome] and n not in t_descendants]
        mediator_nodes = [n for n in self._graph.nodes if n in t_descendants and n != self.outcome]

        # Fixed anchor positions
        pos[self.treatment] = np.array([0.0, 0.0])
        pos[self.outcome] = np.array([2.0, 0.0])

        # Position confounders above
        for i, c in enumerate(confounder_nodes):
            x = 0.5 + (i % 3) * 0.5
            y = 0.8 + (i // 3) * 0.6
            pos[c] = np.array([x, y])

        # Position mediators in between
        for i, m in enumerate(mediator_nodes):
            x = 1.0
            y = -0.6 - i * 0.5
            pos[m] = np.array([x, y])

        # Fallback for any unpositioned nodes
        for n in self._graph.nodes:
            if n not in pos:
                pos[n] = np.array([np.random.uniform(0.2, 1.8), np.random.uniform(-1.0, 1.0)])

        # Node coordinates & styling
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_labels = []

        role_color_map = {
            "treatment": "#10B981",       # Emerald
            "outcome": "#3B82F6",         # Sapphire Blue
            "confounder": "#14B8A6",      # Teal
            "mediator": "#F97316",        # Orange
            "collider": "#EF4444",        # Red
            "covariate": "#64748B",       # Slate
        }

        for node in self._graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)

            role = self.node_meta.get(node, {}).get("role", "covariate")
            if node == self.treatment:
                color = role_color_map["treatment"]
                role_label = "Focal Treatment (T)"
            elif node == self.outcome:
                color = role_color_map["outcome"]
                role_label = "Focal Outcome (Y)"
            elif node in selected_set:
                color = "#2DD4BF"  # Adjusted
                role_label = f"{role.capitalize()} (Adjusted in Model)"
            elif role in role_color_map:
                color = role_color_map[role]
                role_label = role.capitalize()
            else:
                color = "#64748B"
                role_label = "Covariate"

            node_colors.append(color)

            parents_list = list(self._graph.predecessors(node))
            children_list = list(self._graph.successors(node))
            desc = self.node_meta.get(node, {}).get("description", "")

            hover_html = (
                f"<b>{node}</b><br>"
                f"Role: <b>{role_label}</b><br>"
                f"Parents (Causes): {', '.join(parents_list) or 'None (Exogenous)'}<br>"
                f"Children (Effects): {', '.join(children_list) or 'None (Terminal)'}<br>"
                f"Description: {desc}"
            )
            node_text.append(hover_html)

        fig = go.Figure()

        # Add directed edge annotations (arrows)
        for u, v in self._graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            is_main_causal = (u == self.treatment and v == self.outcome)
            edge_color = "#2DD4BF" if is_main_causal else "rgba(148, 163, 184, 0.6)"
            arrow_width = 3 if is_main_causal else 1.5

            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=arrow_width,
                arrowcolor=edge_color,
                standoff=14,
                startstandoff=14,
                opacity=0.85
            )

        # Add node scatter trace
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=36,
                color=node_colors,
                line=dict(color="#FFFFFF", width=2)
            ),
            text=node_labels,
            textposition="bottom center",
            textfont=dict(color="#F8FAFC", size=12, family="Inter, sans-serif"),
            hoverinfo="text",
            hovertext=node_text,
            name="Variables"
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#070410",
            plot_bgcolor="#070410",
            showlegend=False,
            margin=dict(l=30, r=30, t=30, b=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hoverlabel=dict(bgcolor="#150E28", bordercolor="#A855F7", font_size=12, font_family="Plus Jakarta Sans, sans-serif")
        )

        return fig

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG state for persistence or reports."""
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "confounders": list(self.confounders),
            "mediators": list(self.mediators),
            "colliders": list(self.colliders),
            "nodes": [
                {"name": n, **self.node_meta.get(n, {})}
                for n in self._graph.nodes()
            ],
            "edges": list(self._graph.edges()),
            "is_acyclic": self.is_acyclic()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        """Reconstruct CausalGraph instance from serialized dictionary."""
        graph = cls(
            treatment=data.get("treatment", "Treatment"),
            outcome=data.get("outcome", "Outcome"),
            confounders=data.get("confounders", []),
            mediators=data.get("mediators", []),
            colliders=data.get("colliders", []),
            edges=data.get("edges", [])
        )
        for n_info in data.get("nodes", []):
            name = n_info.get("name")
            if name:
                graph.add_node(
                    name=name,
                    role=n_info.get("role", "covariate"),
                    observed=n_info.get("observed", True),
                    description=n_info.get("description", "")
                )
        return graph


# -----------------------------------------------------------------------------
# Classic Preset Templates (Pearl 2009)
# -----------------------------------------------------------------------------

def create_classic_confounding_dag() -> CausalGraph:
    """Classic Confounder Triangle: X causes both Treatment and Outcome."""
    return CausalGraph(
        treatment="Treatment",
        outcome="Outcome",
        confounders=["Confounder_X"],
        edges=[
            ("Confounder_X", "Treatment"),
            ("Confounder_X", "Outcome"),
            ("Treatment", "Outcome")
        ]
    )


def create_mediation_dag() -> CausalGraph:
    """Mediation Model: Treatment affects Outcome both directly and indirectly via Mediator M."""
    return CausalGraph(
        treatment="Treatment",
        outcome="Outcome",
        confounders=["Baseline_Confounder"],
        mediators=["Mediator_M"],
        edges=[
            ("Baseline_Confounder", "Treatment"),
            ("Baseline_Confounder", "Outcome"),
            ("Treatment", "Mediator_M"),
            ("Mediator_M", "Outcome"),
            ("Treatment", "Outcome")
        ]
    )


def create_m_bias_dag() -> CausalGraph:
    """
    Classic M-Bias / Butterfly Collider Paradox:
    U1 -> Treatment, U1 -> Collider_C <- U2, U2 -> Outcome, Treatment -> Outcome.
    Conditioning on Collider_C induces spurious association between Treatment and Outcome!
    """
    g = CausalGraph(
        treatment="Treatment",
        outcome="Outcome",
        colliders=["Collider_C"],
        edges=[
            ("U1_PriorStatus", "Treatment"),
            ("U1_PriorStatus", "Collider_C"),
            ("U2_GeneticRisk", "Collider_C"),
            ("U2_GeneticRisk", "Outcome"),
            ("Treatment", "Outcome")
        ]
    )
    g.node_meta["U1_PriorStatus"] = {"role": "unobserved", "observed": False, "description": "Unmeasured Factor 1"}
    g.node_meta["U2_GeneticRisk"] = {"role": "unobserved", "observed": False, "description": "Unmeasured Factor 2"}
    return g


def create_instrumental_variable_dag() -> CausalGraph:
    """
    Instrumental Variable (IV) DAG:
    Z -> Treatment -> Outcome, with unmeasured confounding U -> Treatment, U -> Outcome.
    Z has no direct effect on Outcome (Exclusion Restriction).
    """
    g = CausalGraph(
        treatment="Treatment",
        outcome="Outcome",
        edges=[
            ("Instrument_Z", "Treatment"),
            ("Unmeasured_U", "Treatment"),
            ("Unmeasured_U", "Outcome"),
            ("Treatment", "Outcome")
        ]
    )
    g.node_meta["Instrument_Z"] = {"role": "instrument", "observed": True, "description": "Exogenous Instrument"}
    g.node_meta["Unmeasured_U"] = {"role": "unobserved", "observed": False, "description": "Latent Confounder"}
    return g
