"""Message Passing Neural Network for simultaneous tracking and state classification.

The message passing formulation of this module, `MPModel` together with its edge, node,
and time-aware node updates, is adapted from Brasó et al., Int. J. Comput. Vis. 2022
(https://github.com/dvl-tum/mot_neural_solver), which in turn builds on
https://github.com/deepmind/graph_nets. Joint node classification and the deep and
handcrafted feature fusion are additions of this work.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter

from baclct.models.node_encoder import NodeEncoder
from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


class EdgeModel(nn.Module):
    """Edge update for one message passing step.

    For each edge, concatenates the current source node, destination node, and edge
    embeddings and passes them through `edge_mlp` to produce the updated edge embedding.
    """

    def __init__(self, edge_mlp: nn.Module):
        """Initialize module.

        Args:
            edge_mlp: Module for edge update with input dim `2 * node_dim + edge_dim` and
                output `edge_dim`.
        """
        super().__init__()
        self.edge_mlp = edge_mlp

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_edge: torch.Tensor,
        *ignore_args,
    ) -> torch.Tensor:
        """Run edge update.

        Each edge is updated from its adjacent nodes using current node and edge feats.
        Feats are concatenated and encoded by MLP.
        """
        edge_input = torch.cat([x_src, x_dst, x_edge], 1)
        return self.edge_mlp(edge_input)


class NodeModel(nn.Module):
    """Message passing and node update for one step.

    For each edge, a message is built from the source node embedding and the (already
    updated) edge embedding via `flow_mlp`. Incoming messages are aggregated at each
    destination node using an order-invariant reducer. The aggregated message is
    concatenated with the previous node embedding and passed through `node_mlp` to produce
    the updated node embedding.
    """

    def __init__(
        self,
        node_mlp: nn.Module,
        flow_mlp: nn.Module,
        aggregation: Literal["mean", "max", "min"] = "mean",
    ):
        """Initialize module.

        Args:
            node_mlp: Module for final node update with input `flow_dim + node_dim`.
            flow_mlp: Module for message passing with input `node_dim + edge_dim`.
            aggregation: Order-invariant aggregation function.
        """
        super().__init__()
        self.node_mlp = node_mlp
        self.flow_mlp = flow_mlp
        self.aggregation = aggregation

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *ignore_args,
    ) -> torch.Tensor:
        """Run message passing and node update."""
        src, dst = edge_index
        flow_input = torch.cat([x[src], edge_attr], dim=1)
        flow = self.flow_mlp(flow_input)
        flow = scatter(flow, dst, dim=0, dim_size=x.size(0), reduce=self.aggregation)
        input_node_update = torch.cat([x, flow], dim=1)
        return self.node_mlp(input_node_update)


class TimeAwareNodeModel(nn.Module):
    """Message passing and node update with temporally directed messages.

    Like `NodeModel`, but messages are routed by edge direction: Edges with `src < dst`
    carry past-to-future messages (encoded by `flow_out_mlp`) and edges with `src > dst`
    carry future-to-past messages (encoded by `flow_in_mlp`). At each destination node,
    both aggregated streams are concatenated with the previous node embedding and passed
    through `node_mlp` to produce the updated embedding.
    """

    def __init__(
        self,
        node_mlp: nn.Module,
        flow_in_mlp: nn.Module,
        flow_out_mlp: nn.Module,
        aggregation: Literal["mean", "max", "min"] = "mean",
    ):
        """Initialize module.

        Args:
            node_mlp: Module for final node update with input `flow_dim + node_dim`.
            flow_in_mlp: Module for message passing with input `node_dim + edge_dim`
                (future -> past).
            flow_out_mlp: Module for message passing with input `node_dim + edge_dim`
                (past -> future).
            aggregation: Order-invariant aggregation function.
        """
        super().__init__()
        self.node_mlp = node_mlp
        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.aggregation = aggregation

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *ignore_args,
    ) -> torch.Tensor:
        """Run message passing and node update."""
        src, dst = edge_index
        flow_out_mask = src < dst
        flow_out_src, flow_out_dst = src[flow_out_mask], dst[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_src], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = scatter(
            flow_out, flow_out_dst, dim=0, dim_size=x.size(0), reduce=self.aggregation
        )

        flow_in_mask = src > dst
        flow_in_src, flow_in_dst = src[flow_in_mask], dst[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_src], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)

        flow_in = scatter(
            flow_in, flow_in_dst, dim=0, dim_size=x.size(0), reduce=self.aggregation
        )
        flow = torch.cat(
            [flow_in, flow_out, x], dim=1
        )  # x is not included in method by Brasó et al.

        return self.node_mlp(flow)


class MPModel(torch.nn.Module):
    """Message passing GNN for joint edge and node classification.

    Operates on a graph whose nodes carry offline-extracted object features and
    whose edges connect candidate associations. The network has three phases:

    1. Initial embedding of node features (handcrafted and/or deep) via `node_encoder` and
       of edge features via `edge_encoder`. If `similarity_func` is provided, the
       similarity between the node embeddings is added to the edge features before edge
       encoding.
    2. Message passing for `num_layers` iterations. Each step first updates the edge
       embeddings from the neighboring nodes and initial edge features (`edge_model`),
       then sends, aggregates, and applies messages to update the node embeddings
       (`node_model`).
    3. Classification of edges via `edge_classifier`, and optionally of nodes via
       `node_classifier`.

    With `reattach_initial_nodes` / `reattach_initial_edges`, the initial embeddings are
    re-concatenated onto the running embeddings before each message passing step. With
    `aggregate_outputs=True`, classifier outputs from every step are returned (e.g., for
    layer-wise supervision); otherwise only the final step's outputs are returned.

    Adapted from Brasó et al., see module docstring.
    """

    def __init__(
        self,
        node_model: NodeModel,
        edge_model: EdgeModel,
        num_layers: int,
        edge_classifier: nn.Module,
        node_encoder: NodeEncoder,
        edge_encoder: nn.Module | None = None,
        node_classifier: nn.Module | None = None,
        reattach_initial_nodes: bool = False,
        reattach_initial_edges: bool = False,
        aggregate_outputs: bool = False,
        similarity_func: nn.Module | None = None,
        similarity_input: Literal["deep", "all"] | None = "deep",
        **ignore_kwargs,
    ):
        """Initialize Graph Model.

        Args:
            node_model: Models used for node and edge updates.
            edge_model: Models used for node and edge updates.
            num_layers: Number of layers of the GNN, i.e. the number of message passing
                steps.
            node_encoder: Encoder for handcrafted and deep features. Combines and/or
                embeds them.
            edge_encoder: Optional encoder for initial edge features.
            edge_classifier: Optional classifiers for binary or multi-class edge
                classification.
            node_classifier: Optional classifiers for binary or multi-class node
                classification.
            reattach_initial_nodes: If `True`, initial node embeddings are concatenated
                with the updated embeddings after each message passing step.
            reattach_initial_edges: If `True`, initial embeddings are concatenated with
                the updated embeddings after each message passing step.
            aggregate_outputs: If `True`, all intermediate outputs are returned as list.
            similarity_func: Optional similarity function applied to node features. Its
                output is appended to `edge_attr` before the edge encoder.
            similarity_input: Which node tensor `similarity_func` is computed for. 'deep'
                uses the raw deep image embeddings (out of the optimization loop, matches
                the offline default), 'all' uses the post-fusion encoded node features,
                and `None` omits the similarity feature. Ignored when `similarity_func`
                is `None`.
            ignore_kwargs: Hidden hydra config args, e.g., number of node features.
        """
        super().__init__()

        self.mp_net = MetaLayer(edge_model, node_model)
        self.num_layers = num_layers

        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.edge_classifier = edge_classifier
        self.node_classifier = node_classifier
        if similarity_input not in (None, "deep", "all"):
            raise ValueError(
                f"similarity_input must be one of None, 'deep', 'all', "
                f"got {similarity_input!r}."
            )
        self.similarity_func = similarity_func
        self.similarity_input = similarity_input
        self._warned_missing_deep = False

        self.reattach_initial_nodes = reattach_initial_nodes
        self.reattach_initial_edges = reattach_initial_edges
        self.aggregate_outputs = aggregate_outputs

    def forward(
        self,
        x_handcrafted: torch.Tensor | None,
        x_deep: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **ignore_kwargs,
    ) -> dict[Literal["edge_predictions", "node_predictions"], list[torch.Tensor]]:
        """Run encoding, message passing, and classification.

        Args:
            x_handcrafted: Normalized unembedded node features of shape
                `(num_nodes, num_node_feats)`.
            x_deep: Normalized unembedded node features of shape
                `(num_nodes, num_node_feats)`.
            edge_index: Continuous zeroed edge indices (`src -> dst`) of shape
                `(2, num_edges)`.
            edge_attr: Normalized unembedded edge features of shape
                `(num_edges, num_edge_feats)`
            **ignore_kwargs: Arguments expected by `MetaLayer`, e.g., global graph attrs.

        Returns:
            Edge and optional node predictions. Returned for last or all message passing
            steps.
        """
        x = self.node_encoder(x_handcrafted, x_deep)

        if self.similarity_func is not None and self.similarity_input is not None:
            sim_x = x_deep if self.similarity_input == "deep" else x
            if sim_x is None:
                if not self._warned_missing_deep:
                    logger.warning(
                        "similarity_input='deep' but no deep node features are "
                        "available. Skipping the similarity edge feature."
                    )
                    self._warned_missing_deep = True
            else:
                src, dst = edge_index
                sim = self.similarity_func(sim_x[src], sim_x[dst])
                if sim.dim() == 1:
                    sim = sim.unsqueeze(1)
                edge_attr = torch.cat([edge_attr, sim], dim=1)

        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        init_node_feats, init_edge_feats = x, edge_attr
        if self.num_layers == 0:
            edge_predictions = self.edge_classifier(edge_attr)
            node_predictions = self.node_classifier(x) if self.node_classifier else None
            return {
                "edge_predictions": [edge_predictions],
                "node_predictions": [node_predictions]
                if node_predictions is not None
                else [],
            }

        edge_outputs = []
        node_outputs = []
        for _l in range(self.num_layers):
            if self.reattach_initial_nodes:
                x = torch.cat([init_node_feats, x], 1)
            if self.reattach_initial_edges:
                edge_attr = torch.cat([init_edge_feats, edge_attr], 1)

            x, edge_attr, _ = self.mp_net(x, edge_index, edge_attr)

            edge_outputs.append(self.edge_classifier(edge_attr))
            if self.node_classifier:
                node_outputs.append(self.node_classifier(x))

        if not self.aggregate_outputs:
            edge_outputs = [edge_outputs[-1]]
            if node_outputs:
                node_outputs = [node_outputs[-1]]

        return {
            "edge_predictions": edge_outputs,
            "node_predictions": node_outputs,
        }
