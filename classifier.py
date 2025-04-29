from typing import Optional

import networkx as nx
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils.convert import from_networkx
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead

from models.encoders import HGCN, HGNN

GNN = {"gcn": gnn.GCNConv, "hgcn": HGCN, "hie": HGNN}


class HieRoberta(XLMRobertaPreTrainedModel):
    """Largely based on the official RobertaForSequenceClassification implementation."""

    def __init__(self, args, config, hierarchy: Optional[nx.DiGraph] = None) -> None:
        super().__init__(config)
        self.config = config
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.hierarchy = hierarchy
        self.hyperbolic = bool(hierarchy) and args.gnn in ["hgcn", "hie"]
        self.node_classification = args.node_classification
        self.node_dim = args.node_dim
        self.output_dim = 1 if args.node_classification else args.node_dim
        self.pooling = args.pooling

        self.roberta = XLMRobertaModel(config, add_pooling_layer=args.pooling)

        # Freeze the first 50% of layers
        if args.freeze:
            modules = [self.roberta.embeddings, self.roberta.encoder.layer[config.num_hidden_layers // 2]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        # Insert GNN between LM and classification head
        if self.hyperbolic:
            self.edges_fwd = nx.adjacency_matrix(hierarchy)
            self.edges_bwd = self.edges_fwd.T
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.projection = nn.Linear(config.hidden_size, config.num_labels * self.node_dim)
            self.gconv_fwd = GNN[args.gnn](...)
            self.gconv_bwd = GNN[args.gnn](...)
            self.out_proj = nn.Linear(config.num_labels * self.node_dim, config.num_labels)
        elif self.hierarchy:
            self.edges_fwd = from_networkx(hierarchy).edge_index
            self.edges_bwd = from_networkx(hierarchy.reverse()).edge_index
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.projection = nn.Linear(config.hidden_size, config.num_labels * self.node_dim)
            self.gconv_fwd = GNN[args.gnn](self.node_dim, self.output_dim)
            self.gconv_bwd = GNN[args.gnn](self.node_dim, self.output_dim)
            self.out_proj = nn.Linear(config.num_labels * self.node_dim, config.num_labels)
        else:
            self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # Pass outputs through GNN
        if self.hierarchy:
            self.edges_fwd = self.edges_fwd.to(sequence_output.device)
            self.edges_bwd = self.edges_bwd.to(sequence_output.device)
            projected = self.dropout(self.projection(sequence_output[:, 0, :]))
            projected = projected.view(projected.shape[0], self.config.num_labels, -1)
            fwd = self.dropout(self.gconv_fwd(projected, self.edges_fwd))
            bwd = self.dropout(self.gconv_bwd(projected, self.edges_bwd))
            convolved = fwd + bwd
            if self.node_classification:
                logits = convolved  # no activation in output layer
            else:
                convolved = torch.relu(convolved)
                convolved = convolved.view(convolved.shape[0], self.config.num_labels * self.node_dim)
                logits = self.out_proj(convolved)
        else:
            logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
