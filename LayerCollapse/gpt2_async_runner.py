import torch
from utils import *
from clm_utils import *

from tqdm.auto import tqdm
import argparse

assert torch.cuda.is_available(), \
"CUDA support is not available."

import pickle
import os

import LiveTune as lt

from utils import TASKS


MODELS = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr-decay", type=float, default=0.99)
parser.add_argument("--gp", type=float, default=10.0, required=True)
parser.add_argument("--epochs", type=float, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--model", type=str, choices=MODELS)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--num_layers", "-nl", type=int)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--use-sgd", action="store_true")
parser.add_argument("--use-LC", action="store_true")
parser.add_argument("--do-middle", action="store_true")
args = parser.parse_args()

# device = torch.device("cuda:" + args.device)

directory = f'/scr/models/LC/models_archive/GPT2/{args.model}/'
if not os.path.exists(directory):
    os.makedirs(directory)
# filename = directory + str("LC" if args.use_LC else "vanilla") + args.name + ".pth"

model_base, tokenizer = get_classification_gpt2_model(pre_trained_model_name=args.model, embd_pdrop=0.0)
# model_base.embd_pdrop = 0.0
# model_base.attn_pdrop = 0.0
# model_base.resid_pdrop = 0.0 

if args.use_LC:
    if args.do_middle:
        model_GP = get_LC_model_gpt2(model_base, num_GP_layers=1, only_list=["10"])
    else:
        model_GP = get_LC_model_gpt2(model_base, num_GP_layers=args.num_layers)
else:
    model_GP = model_base

lm_datasets, data_collator, encodings_for_eval_pt = get_wikitext_dataset(tokenizer, block_size=args.block_size, fraction_of_train=1 if args.epochs >= 1 else args.epochs)

# if os.path.exists(filename):
#     model_GP.load_state_dict(torch.load(filename))

filename = directory + str("LC" if args.use_LC else "vanilla") + args.name + ".pth"
filebacklog = directory + str("LC" if args.use_LC else "vanilla") + args.name +  "_back.pth"


if os.path.exists(filename):
    model_GP.load_state_dict(torch.load(filename))
    print(filebacklog)
    print(filename)
    torch.save(model_GP.state_dict(), filebacklog)

get_layer_gates_loss(model_GP)


train_lm(model_GP, lm_datasets=lm_datasets, epochs=args.epochs, data_collator=data_collator,
                    batch_size=args.batch_size, gp_weight=args.gp, learning_rate=args.lr, lr_decay=args.lr_decay, weight_decay=args.wd, use_sgd=args.use_sgd, gradient_accumulation_steps=8 if args.model == "gpt2-large" else 1)
torch.save(model_GP.state_dict(), filename)
print(eval_perplexity_with_trainer(model_GP, data_collator, lm_datasets["test"]))
get_layer_gates_loss(model_GP)