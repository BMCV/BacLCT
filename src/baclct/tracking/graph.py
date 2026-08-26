"""Convert edge predictions to graph and label trajectories."""

from __future__ import annotations

import networkx as nx
import polars as pl

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def edge_df_to_graph(edge_data: pl.DataFrame, node_features: pl.DataFrame) -> nx.DiGraph:
    """Convert Edge Data to Directed Graph.

    Edge data is converted to directed graph and isolated nodes are added based on the
    provided node feature dataframe.
    """
    G = nx.DiGraph()
    G.add_edges_from(edge_data.sort("src", "dst").select("src", "dst").iter_rows())
    G.add_nodes_from(node_features.get_column("index"))

    return G


def label_trajectories(
    graph: nx.DiGraph,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Label Trajectories as Connected Components.

    Each node is assigned a root label (i.e. based on connected component representing all
    trajectories including daughters originating from a single cell), a trajectory label
    (i.e. unique label for trajectory started at root or daughter cell), and if the
    trajectory was initiated by division a parent label.
    """
    trajectory_labels = {}
    parent_labels = {}
    root_labels = {}

    current_label = 0
    components = sorted(nx.weakly_connected_components(graph), key=lambda c: sorted(c)[0])
    for component in components:
        current_label += 1
        component_label = current_label
        # the component's own trajectory has no ancestor inside it, daughters get their
        # root when they are pushed
        root_labels[component_label] = 0

        stack = [(sorted(component)[0], component_label)]
        while stack:
            v, lbl = stack.pop()
            if v in trajectory_labels:
                continue

            trajectory_labels[v] = lbl

            children = list(graph.successors(v))
            if len(children) == 1:
                stack.append((children[0], lbl))
            elif len(children) > 1:
                for child in children:
                    current_label += 1
                    child_label = current_label
                    parent_labels[child_label] = lbl
                    root_labels[child_label] = component_label
                    stack.append((child, child_label))

    return trajectory_labels, parent_labels, root_labels
