import copy
import torch
import torch.nn as nn
import numpy as np
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertForSequenceClassification
from transformers import pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from evaluate import EvaluationSuite, evaluator
from trainer_wrapper import TrainerGP
from collapsible_bert import BertLayerLC
from transformers.models.bert.modeling_bert import BertLayer
from torchprofile import profile_macs

# from GatedBert import BertForSequenceClassificationGP

# from GatedDistilBert import  DistilBertLayerGP
# from transformers.models.distilbert.modeling_distilbert import TransformerBlock

# from GatedGPT2 import GPT2BlockGP#, GPT2ForSequenceClassificationGP
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2MLP
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from optimizer import CustomAdamW

from collapsible_gpt2 import GPT2MLPLC
# from GatedPhi import PhiDecoderLayerGP
# from transformers.models.phi.modeling_phi import PhiDecoderLayer

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


TASKS = ["sst2", "cola", "mrpc", "mnli", "stsb", "qqp", "qnli", "rte", "wnli"]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_glue_task_dataset(task, tokenizer, train_batch_size=64, eval_batch_size=256, label2id=None, matched=True):
  
    sentence1_key, sentence2_key = task_to_keys[task]
    is_regression = task == "stsb"
    padding = "max_length"
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if  "label" in examples and label2id is not None:
            result["labels"] = [(label2id[l] if l != -1 else -1) for l in examples["label"]]
        elif "label" in examples:
            result["labels"] = examples["label"]
        return result
    
    dataset = load_dataset("glue", task, cache_dir="/scr/datasets/huggingface/nlp")
    temp_set = dataset.map(preprocess_function, batched=True)
    temp_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = torch.utils.data.DataLoader(temp_set["train"], batch_size=train_batch_size, shuffle=True)
    if task == "mnli" and matched:
        validation_dataloader = torch.utils.data.DataLoader(temp_set["validation_matched"], batch_size=eval_batch_size, shuffle=False)
    elif task == "mnli" and matched:
        validation_dataloader = torch.utils.data.DataLoader(temp_set["validation_mismatched"], batch_size=eval_batch_size, shuffle=False)
    else:
        validation_dataloader = torch.utils.data.DataLoader(temp_set["validation"], batch_size=eval_batch_size, shuffle=False)
    return train_dataloader, validation_dataloader, temp_set

def get_classification_bert_model(pre_trained_model_name = None):
    if pre_trained_model_name in ["bert-base-uncased", "bert-large-uncased"]:
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        model = BertForSequenceClassification.from_pretrained(pre_trained_model_name)
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pre_trained_model_name, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pre_trained_model_name)
    return model, tokenizer
    
def get_classification_distilbert_model(pre_trained_model_name = None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pre_trained_model_name) # distilbert-base-uncased-finetuned-sst-2-english
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pre_trained_model_name)
    return model, tokenizer

def get_classification_gpt2_model(pre_trained_model_name = None, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name, padding = True, truncation = True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(pre_trained_model_name, **kwargs)
    return model, tokenizer

def basic_evaluate(model, dataset, metric=clf_metrics, device="cuda:0"):
    model.eval()
    for batch in tqdm(dataset):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            metric.add_batch(references=batch["labels"], predictions=logits.argmax(dim=-1))
    return metric.compute()

def standard_evaluate(model, dataset_validation, task):
    metric = evaluate.load("glue", task)
    is_regression = task == "stsb"
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    trainer = Trainer(
        model=model,                  
        eval_dataset=dataset_validation,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate()


def train_with_trainer(model, train_dataloader, validation_dataloader, epochs=3, learning_rate=5e-3, train_batch_size=64, eval_batch_size=256, gp_weight=0, task = "sst2", lr_decay=0.9, weight_decay=0.01, use_sgd=False):
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)``
    else:
        optimizer = CustomAdamW(model, lr=learning_rate, gp_weight=gp_weight, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_decay)
    metric = evaluate.load("glue", task)
    is_regression = task == "stsb"
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        overwrite_output_dir=True,
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        learning_rate=learning_rate,
        max_grad_norm=10.0,
    )
    trainer = TrainerGP(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataloader,         
        eval_dataset=validation_dataloader,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        gp = gp_weight,
        use_sgd = use_sgd,
    )
    trainer.train()
    
# def train(model, train_dataloader, validation_dataloader, device, epochs=3, learning_rate=5e-5, train_batch_size=64, eval_batch_size=256):
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#     for epoch in range(epochs):
#         model.train()
#         for batch in tqdm(train_dataloader):
#             optimizer.zero_grad()
#             batch = {k: v.to(device) for k, v in batch.items()}
#             loss = model(**batch).loss# + get_GP_loss(model) * 1.0
#             # loss = get_GP_loss(model) * 500.0
#             loss.backward()
#             optimizer.step()
#             print(loss.item())
#         print("Epoch {}:".format(epoch))
#         print("Training set:")
#         print(basic_evaluate(model, train_dataloader))
#         print("Validation set:")
#         print(basic_evaluate(model, validation_dataloader))

def change_module(model, name, module):
    name_list = name.split(".")
    if len(name_list) == 1:
        model._modules[name_list[0]] = module
    else:
        change_module(model._modules[name_list[0]], ".".join(name_list[1:]), module)

def get_LC_model_bert(model, num_GP_layers):
    config = model.config
    copy_model = copy.deepcopy(model)
    if num_GP_layers <= 0:
        return copy_model
    for name, module in list(copy_model.named_modules())[::-1]:
        if isinstance(module, BertLayer):
            print("Setting up gated layer {}".format(name))
            new_module = BertLayerLC(config=config)
            new_module.load_from_basic(module=module)
            change_module(copy_model, name, new_module)
            num_GP_layers -= 1
            if num_GP_layers == 0:
                return copy_model
    return copy_model

# def get_GP_model_distilbert(model, num_GP_layers):
#     config = model.config
#     copy_model = copy.deepcopy(model)
#     if num_GP_layers <= 0:
#         return copy_model
#     for name, module in list(copy_model.named_modules())[::-1]:
#         if isinstance(module, TransformerBlock):
#             print("Setting up gated layer {}".format(name))
#             new_module = DistilBertLayerGP(config=config)
#             new_module.load_from_basic(module=module)
#             change_module(copy_model, name, new_module)
#             num_GP_layers -= 1
#             if num_GP_layers == 0:
#                 return copy_model
#     return copy_model

def get_LC_model_gpt2(model, num_GP_layers, only_list = []):
    config = model.config
    copy_model = copy.deepcopy(model)
    if num_GP_layers <= 0:
        return copy_model
    if len(only_list) == 0:
        for name, module in list(copy_model.named_modules())[::-1]:
            if isinstance(module, GPT2MLP):
                print("Setting up gated layer {}".format(name))
                intermediate_size = module.c_fc.nf
                new_module = GPT2MLPLC(intermediate_size=intermediate_size, config=config)
                new_module.load_from_basic(module=module)
                change_module(copy_model, name, new_module)
                num_GP_layers -= 1
                if num_GP_layers == 0:
                    return copy_model
        return copy_model
    else:
        for name, module in list(copy_model.named_modules())[::-1]:
            if isinstance(module, GPT2MLP) and name.split(".")[-2] in only_list:
                print("Setting up gated layer {}".format(name))
                intermediate_size = module.c_fc.nf
                new_module = GPT2MLPLC(intermediate_size=intermediate_size, config=config)
                new_module.load_from_basic(module=module)
                change_module(copy_model, name, new_module)
        return copy_model

# def get_GP_module_phi(model, num_GP_layers):
#     config = model.config
#     copy_model = copy.deepcopy(model)
#     if num_GP_layers <= 0:
#         return copy_model
#     for name, module in list(copy_model.named_modules())[::-1]:
#         if isinstance(module, PhiDecoderLayer):
#             print("Setting up gated layer {}".format(name))
#             new_module = PhiDecoderLayerGP(config=config)
#             new_module.load_from_basic(module=module)
#             change_module(copy_model, name, new_module)
#             num_GP_layers -= 1
#             if num_GP_layers == 0:
#                 return copy_model
#     return copy_model

def get_GP_loss(model):
    loss = 0
    for name, module in model.named_modules():
        if isinstance(module, BertLayerLC) or isinstance(module, GPT2MLPLC): #or isinstance(module, DistilBertLayerGP) or isinstance(module, GPT2BlockGP):
            loss += module.get_loss()
    return loss

def get_layer_gates_loss(model):
    for name, module in model.named_modules():
        if isinstance(module, BertLayerLC) or isinstance(module, GPT2MLPLC):# or isinstance(module, DistilBertLayerGP) or isinstance(module, GPT2BlockGP):
            print("Layer {}:".format(name), "loss: ", module.get_loss())

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

def collapse_model(model, num_layers = 100):
    for name, module in list(model.named_modules())[::-1]:
        if (isinstance(module, BertLayerLC)) or isinstance(module, GPT2MLPLC):# or isinstance(module, DistilBertLayerGP) or isinstance(module, GPT2BlockGP)) and num_layers > 0:
            print("Collapsing layer {}".format(name))
            module.collapse()
            num_layers -= 1

def collapse_bypass_only(model, num_layers = 100, single_layer_id=-1):
    if single_layer_id == -1:
        for name, module in list(model.named_modules())[::-1]:
            if (isinstance(module, GPT2MLPLC)) and num_layers > 0:
                print("Collapsing layer {}".format(name))
                module.act_LC.weight.data = torch.ones_like(module.act_LC.weight.data)
                num_layers -= 1
    else:
        for name, module in list(model.named_modules())[::-1]:
            if (isinstance(module, GPT2MLPLC)) and str(single_layer_id) == name.split(".")[-2]:
                print("Collapsing layer {}".format(name))
                module.act_LC.weight.data = torch.ones_like(module.act_LC.weight.data)