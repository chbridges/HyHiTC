from typing import Optional

import networkx as nx
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils.convert import from_networkx
from transformers import RobertaPreTrainedModel, XLMRobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

GNN = {"gcn": gnn.GCNConv}


class MultilabelModel(RobertaPreTrainedModel):
    """Largely based on the official RobertaForSequenceClassification implementation."""

    def __init__(self, args, config, hierarchy: Optional[nx.DiGraph] = None) -> None:
        super().__init__(config)
        self.config = config
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.hierarchy = hierarchy

        self.roberta = XLMRobertaModel(config)

        # Insert GNN between LM and classification head
        if hierarchy:
            self.edge_index = from_networkx(hierarchy)
            self.projection = nn.Linear(config.hidden_size, config.num_labels * args.node_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.gconv = GNN[args.gnn](config.num_labels * args.node_size, config.num_labels)

        self.classifier = RobertaClassificationHead(config)

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
            projected = self.dropout(self.projection(sequence_output))
            convolved = self.gconv(projected)
            logits = self.classifier(convolved)
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
