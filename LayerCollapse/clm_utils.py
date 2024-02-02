import copy
import torch
import torch.nn as nn
import numpy as np
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from transformers import pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from evaluate import EvaluationSuite, evaluator
from trainer_wrapper import TrainerGP
from torchprofile import profile_macs

# from GatedBert import BertForSequenceClassificationGP

# from GatedDistilBert import  DistilBertLayerGP
from transformers.models.distilbert.modeling_distilbert import TransformerBlock

# from GatedGPT2 import GPT2BlockGP#, GPT2ForSequenceClassificationGP
from collapsible_gpt2 import GPT2MLPLC
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2ForSequenceClassification
from transformers import GPT2Tokenizer
from transformers.trainer_utils import EvalPrediction

from optimizer import CustomAdamW_GPT2

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer
import math
from transformers import DataCollatorForLanguageModeling
from scheduler import LiveTuneGammaLR
import LiveTune as lt

def get_wikitext_dataset(tokenizer, block_size=128, fraction_of_train=0.1):
    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir="/scr/datasets/huggingface/nlp")
    wikitext_len = wikitext["train"].__len__()
    wikitext["train"] = wikitext["train"].shuffle().select(range(int(fraction_of_train * wikitext_len)))
    def preprocess_function(examples):
        return tokenizer(" ".join(examples["text"]))
    tokenized_datasets = wikitext.map(preprocess_function, batched=True, remove_columns=wikitext["train"].column_names)

    def group_texts(examples):
        # Concatenate all texts.
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    # remove the last datapoint
    lm_datasets["train"] = lm_datasets["train"].select(range(lm_datasets["train"].__len__() - 1))
    lm_datasets["test"] = lm_datasets["test"].select(range(lm_datasets["test"].__len__() - 1))
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=block_size)

    encodings_for_eval_pt = tokenizer(" ".join(wikitext["test"]["text"]), return_tensors="pt")

    return lm_datasets, data_collator, encodings_for_eval_pt

def eval_perplexity_with_trainer(model, data_collator, eval_dataset):
    ''' this is a hacky way to get perplexity using the Trainer class. and always returns worse perplexity than eval_perplexity()'''
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    eval_results = trainer.evaluate()
    return math.exp(eval_results['eval_loss'])

def eval_perplexity(model, encodings, stride, device, max_length=512):
    seq_len = encodings.input_ids.size(1)

    nlls = []
    model.eval().to(device)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def train_lm(model, lm_datasets, data_collator, learning_rate=2e-5, weight_decay=0.01, gp_weight=0,  epochs=1, batch_size=8, use_sgd=False, lr_decay = 0.9, gradient_accumulation_steps=1):
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = CustomAdamW_GPT2(model, lr=learning_rate, gp_weight=gp_weight, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=lr_decay)
        
    live_gamma = lt.liveVar(1.0, "gamma")
    scheduler = LiveTuneGammaLR(optimizer, live_gamma=live_gamma, verbose=False)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        num_train_epochs=epochs if epochs >= 1 else 1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # learning_rate=2e-5,
        # weight_decay=0.01,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=50,
        max_grad_norm=10.0,
        dataloader_pin_memory=True,
    )

    trainer = TrainerGP(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        gp = gp_weight,
        use_sgd=use_sgd,
    )

    trainer.train()