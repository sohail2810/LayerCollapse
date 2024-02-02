import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2ForSequenceClassification, GPT2_INPUTS_DOCSTRING
import copy

class GPT2BlockGP(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.bypass = nn.Parameter(torch.zeros(1), requires_grad=True)

        
        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        if isinstance(self.mlp, nn.Identity):
            hidden_states = 2 * residual
        else:
        
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection

            # hidden_states = residual + feed_forward_hidden_states
            hidden_states = 2 * ((1 - self.bypass.sigmoid()) * residual + self.bypass.sigmoid() * feed_forward_hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    
    def get_loss(self):
        return self.bypass.sigmoid().mean()
    
    def purge_layer(self):
        self.mlp = nn.Identity()
        self.ln_2 = nn.Identity()

    def load_from_basic(self, module):
        self.ln_1 = copy.deepcopy(module.ln_1)
        self.attn = copy.deepcopy(module.attn)
        self.ln_2 = copy.deepcopy(module.ln_2)
        if hasattr(module, "crossattention"):
            self.crossattention = copy.deepcopy(module.crossattention)
            self.ln_cross_attn = copy.deepcopy(module.ln_cross_attn)
        self.mlp = copy.deepcopy(module.mlp)


# class GPT2ForSequenceClassificationGP(GPT2ForSequenceClassification):
#     def __init__(self, config, reg_GP=0):
#         super().__init__(config)
#         self.reg_GP = reg_GP

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint="microsoft/DialogRPT-updown",
#         output_type=SequenceClassifierOutputWithPast,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size, sequence_length = input_ids.shape[:2]
#         else:
#             batch_size, sequence_length = inputs_embeds.shape[:2]

#         assert (
#             self.config.pad_token_id is not None or batch_size == 1
#         ), "Cannot handle batch sizes > 1 if no padding token is defined."
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
#                     logits.device
#                 )
#             else:
#                 sequence_lengths = -1
#                 logger.warning(
#                     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#                 )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         loss = loss + self.reg_GP * self.get_GP_loss() + 500
#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )

#     def get_GP_loss(self):
#         loss = 0
#         for name, modeule in self.named_modules():
#             if isinstance(modeule, GPT2BlockGP):
#                 loss += modeule.get_loss()
#         return loss