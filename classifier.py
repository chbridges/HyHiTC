import argparse
from typing import Optional

import networkx as nx
import torch
from torch import nn
from torch_geometric.nn import GCN
from torch_geometric.utils.convert import from_networkx
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead

from models.base_models import NCModel


class HieRobertaConfig(XLMRobertaConfig):
    model_type = "HieRoberta"


class HieRobertaModel(XLMRobertaPreTrainedModel):
    """Largely based on the official RobertaForSequenceClassification implementation."""

    def __init__(
        self,
        language_model: XLMRobertaModel,
        args: argparse.Namespace,
        config: XLMRobertaConfig,
        hierarchy: Optional[nx.DiGraph] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(config)
        self.hierarchy = hierarchy
        self.hyperbolic = hierarchy is not None and args.gnn in ["HGCN", "HIE"]
        self.node_classification = args.node_classification
        self.num_labels = config.num_labels
        self.use_return_dict = config.use_return_dict

        if args.mcloss:
            self.loss_fct = nn.BCELoss()
            self.R = torch.Tensor(nx.adjacency_matrix(hierarchy).todense().T)
        else:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.R = None

        self.roberta = language_model

        # Insert GNN between LM and classification head
        self.projection = nn.Linear(config.hidden_size, self.num_labels * args.node_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.hyperbolic:
            if args.directed:
                sparse_adj = nx.adjacency_matrix(hierarchy)
            else:
                sparse_adj = nx.adjacency_matrix(hierarchy.to_undirected(as_view=True))
            self.adj = torch.Tensor(sparse_adj.todense())
            self.hgcn = NCModel(args)
        elif self.hierarchy:
            if args.directed:
                self.adj = from_networkx(hierarchy).edge_index
            else:
                self.adj = from_networkx(hierarchy.to_undirected(as_view=True)).edge_index
            self.gcn = GCN(
                in_channels=args.node_dim,
                hidden_channels=args.node_dim,
                num_layers=args.layers,
                out_channels=1 if self.node_classification else None,
                dropout=config.hidden_dropout_prob,
            )
        else:
            self.classifier = XLMRobertaClassificationHead(config)

        if self.hierarchy and not self.node_classification:
            self.out_proj = nn.Linear(self.num_labels * args.node_dim, self.num_labels)

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
        return_dict = return_dict if return_dict is not None else self.use_return_dict

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
            node_features = projected.view(projected.shape[0], self.num_labels, -1)  # -1 == node_dim

            if self.hyperbolic:
                convolved_hyp = self.hgcn.encode(node_features, self.adj)
                convolved = self.hgcn.decoder.decode(convolved_hyp, self.adj)
            else:
                convolved = self.gcn(node_features, self.adj)

            if self.node_classification:
                logits = convolved.view(convolved.shape[0], -1)  # HGCN decoder returns [batch_size, n_classes, 1]
            else:
                convolved = torch.relu(convolved)
                convolved = convolved.view(convolved.shape[0], -1)
                logits = self.out_proj(convolved)
        else:
            logits = self.classifier(sequence_output)

        if self.R is not None:  # MCM
            probas = logits.sigmoid()
            if self.training:
                probas = self.get_mcloss(probas, labels, self.R)
            else:
                probas = self.get_constr_out(probas, self.R)

        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            if self.R is not None:
                with torch.autocast("cuda", enabled=False):
                    loss = self.loss_fct(probas, labels.double())
            else:
                loss = self.loss_fct(logits, labels.double())

            if self.hyperbolic and self.hgcn.args.hyp_ireg != "0":  # if HIE
                loss_hir = self.hgcn.hir_loss(convolved_hyp)
                loss + self.hgcn.args.ireg_lambda * (max(loss_hir, -10) + 10)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_constr_out(self, x: torch.Tensor, R: torch.Tensor):
        """MCM implementation by Giunchiglia and Lukasiewicz: https://github.com/EGiunchiglia/C-HMCNN"""
        c_out = x.double()
        c_out = c_out.unsqueeze(1)
        c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
        R_batch = R.expand(len(x), R.shape[1], R.shape[1]).to(x.device)
        final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
        return final_out

    def get_mcloss(self, output: torch.Tensor, labels: torch.Tensor, R: torch.Tensor):
        """MCLoss implementation by Giunchiglia and Lukasiewicz: https://github.com/EGiunchiglia/C-HMCNN"""
        constr_output = self.get_constr_out(output, R)
        train_output = labels * output.double()
        train_output = self.get_constr_out(train_output, R)
        return (1 - labels) * constr_output.double() + labels * train_output

    def freeze_lm(self, ratio: float):
        if not ratio:
            return
        n_layers = int(ratio * len(self.roberta.encoder.layer))
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for module in self.roberta.encoder.layer[:n_layers]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = True
        for module in self.roberta.encoder.layer:
            for param in module.parameters():
                param.requires_grad = True
