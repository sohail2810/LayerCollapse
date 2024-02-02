from transformers import Trainer
from transformers import TrainingArguments
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from collapsible_bert import BertLayerLC
import torch
from accelerate import Accelerator

def get_GP_loss(model):
    loss = 0
    for name, module in model.named_modules():
        if isinstance(module, BertLayerLC):# or isinstance(module, DistilBertLayerGP) or isinstance(module, GPT2BlockGP):
            loss += module.get_loss()
    return loss

class TrainerGP(Trainer):
    def __init__(self, *args, **kwargs):
        self.gp = kwargs.pop('gp')
        self.use_sgd = kwargs.pop('use_sgd')
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # if self.use_gp:
            # loss = loss + get_GP_loss(model)* self.gp
        if self.use_sgd:
            loss_gp = get_GP_loss(model)
            loss = loss + loss_gp * self.gp
        else:
            with torch.no_grad():
                loss_gp = get_GP_loss(model) * 100 // 1 * 100
            loss = loss + loss_gp 
        return (loss, outputs) if return_outputs else loss
