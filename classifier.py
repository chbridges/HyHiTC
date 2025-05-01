from typing import Optional

import networkx as nx
import torch
from torch import nn
from torch_geometric.nn import GCN
from torch_geometric.utils.convert import from_networkx
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead

from models.base_models import NCModel


class HieRoberta(XLMRobertaPreTrainedModel):
    """Largely based on the official RobertaForSequenceClassification implementation."""

    def __init__(self, args, config, hierarchy: Optional[nx.DiGraph] = None) -> None:
        super().__init__(config)
        self.config = config
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.hierarchy = hierarchy
        self.hyperbolic = bool(hierarchy) and args.gnn in ["HGCN", "HIE"]
        self.node_classification = args.node_classification
        self.node_dim = args.node_dim
        self.pooling = args.pooling

        self.roberta = XLMRobertaModel(config, add_pooling_layer=args.pooling)

        # Freeze the first 50% of layers
        if args.freeze:
            modules = [self.roberta.embeddings, self.roberta.encoder.layer[: config.num_hidden_layers // 2]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        # Insert GNN between LM and classification head
        self.projection = nn.Linear(config.hidden_size, config.num_labels * self.node_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.hyperbolic:
            sparse_adj = nx.adjacency_matrix(hierarchy.to_undirected(as_view=True))
            self.adj = torch.Tensor(sparse_adj.todense())
            self.hgcn = NCModel(args)
        elif self.hierarchy:
            self.adj = from_networkx(hierarchy.to_undirected(as_view=True)).edge_index
            self.gcn = GCN(
                in_channels=self.node_dim,
                hidden_channels=self.node_dim,
                num_layers=args.layers,
                out_channels=1 if self.node_classification else None,
                dropout=config.hidden_dropout_prob,
            )
        else:
            self.classifier = XLMRobertaClassificationHead(config)

        if self.hierarchy and not self.node_classification:
            self.out_proj = nn.Linear(config.num_labels * self.node_dim, config.num_labels)

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
            self.adj = self.adj.to(sequence_output.device)
            projected = self.dropout(self.projection(sequence_output[:, 0, :]))
            projected = projected.view(projected.shape[0], self.config.num_labels, -1)

            if self.hyperbolic:
                convolved_hyp = self.hgcn.encode(projected, self.adj)
                convolved = self.hgcn.decoder.decode(convolved_hyp, self.adj)
            else:
                convolved = self.gcn(projected, self.adj)

            if self.node_classification:
                logits = convolved
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
