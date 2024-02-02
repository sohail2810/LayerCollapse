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

class GPT2MLPLC(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act_LC = nn.PReLU(num_parameters=1, init=0.01)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act_LC(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def collapse(self):
        # W1 = self.fc1.weight.data
        # B1 = self.fc1.bias.data
        # W2 = self.fc2.weight.data
        # B2 = self.fc2.bias.data

        # new_W = W2 @ W1
        # new_B = W2 @ B1 + B2

        # self.fc1 = nn.Linear(self.fc1.in_features, self.fc2.out_features)
        # self.fc1.weight.data = new_W
        # self.fc1.bias.data = new_B
        # self.fc2 = nn.Identity()
        # self.act = nn.Identity()
        # self.drop1 = nn.Identity()

        if isinstance(self.act_LC, nn.Identity):
            return

        W1 = self.c_fc.weight.data
        B1 = self.c_fc.bias.data
        W2 = self.c_proj.weight.data
        B2 = self.c_proj.bias.data

        new_W = W1 @ W2
        new_B = B1 @ W2 + B2

        self.c_fc = Conv1D(new_W.shape[1], new_W.shape[0])
        self.c_fc.weight.data = new_W
        self.c_fc.bias.data = new_B
        self.c_proj = nn.Identity()
        self.act_LC = nn.Identity()

    def load_from_basic(self, module):
        self.c_fc = copy.deepcopy(module.c_fc)
        self.c_proj = copy.deepcopy(module.c_proj)

    def get_loss(self):
        if isinstance(self.act_LC, nn.Identity):
            return 0
        return (self.act_LC.weight - 1)**2
