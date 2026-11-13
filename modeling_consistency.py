"""Training-only clean/purified consistency extension of the official PURE model."""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from modeling_bert import PureBertForSequenceClassification


class ConsistencyPureBert(PureBertForSequenceClassification):
    def __init__(self, config, consistency_lambda=None, auxiliary_weight=None, **kwargs):
        super().__init__(config, **kwargs)
        self.consistency_lambda = (getattr(config, "consistency_lambda", 0.0)
                                   if consistency_lambda is None else consistency_lambda)
        self.auxiliary_weight = (getattr(config, "auxiliary_weight", 1.0)
                                 if auxiliary_weight is None else auxiliary_weight)
        config.consistency_lambda = self.consistency_lambda
        config.auxiliary_weight = self.auxiliary_weight
        config.pilot_variant = "consistency_pilot_v1"
        self.loss_components = {}

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, return_dict=None, **kwargs):
        # Deployment uses precisely the original purified path.
        if not self.training or labels is None or (self.auxiliary_weight == 0 and self.consistency_lambda == 0):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids, labels=labels,
                                   return_dict=return_dict, **kwargs)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, return_dict=True, **kwargs)
        h = outputs.last_hidden_state
        zp = self.mean_pooling(self.pure(h, attention_mask)).squeeze(1)
        logits = self.classifier(self.dropout(zp))
        # Compute the purified supervised path first to preserve its RNG ordering.
        z0 = self.mean_pooling(h).squeeze(1)
        clean_logits = self.classifier(self.dropout(z0))
        ce = self._calculate_loss(logits, labels)
        auxiliary = self._calculate_loss(clean_logits, labels)
        teacher = self.classifier(z0).softmax(-1).detach()
        kl = F.kl_div(self.classifier(zp).log_softmax(-1), teacher, reduction="batchmean")
        loss = ce + self.auxiliary_weight * auxiliary + self.consistency_lambda * kl
        self.loss_components = dict(purified_ce=float(ce.detach()),
                                    auxiliary_ce=float(auxiliary.detach()), kl=float(kl.detach()))
        if return_dict is False:
            return (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)
