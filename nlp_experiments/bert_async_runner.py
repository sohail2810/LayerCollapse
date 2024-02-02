import torch
from utils import *

from tqdm.auto import tqdm
import argparse

assert torch.cuda.is_available(), \
"CUDA support is not available."

import pickle
import os

import LiveTune as lt

# bert_choices = ["base", "large", "base-ft-sst2", "large-ft-sst2", "base-ft-stsb"]

huggingface_models = {"base": "bert-base-uncased",
                        "large": "bert-large-uncased",
                      "base-ft-sst2": "yoshitomo-matsubara/bert-base-uncased-sst2",
                        "large-ft-sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
                      "base-ft-stsb": "gchhablani/bert-base-cased-finetuned-stsb",
                        "large-ft-stsb": "yoshitomo-matsubara/bert-large-uncased-stsb",
                      "base-ft-mrpc": "textattack/bert-base-uncased-MRPC",
                        "large-ft-mrpc": "yoshitomo-matsubara/bert-large-uncased-mrpc",
                      "base-ft-cola": "yoshitomo-matsubara/bert-base-uncased-cola",
                        "large-ft-cola": "yoshitomo-matsubara/bert-large-uncased-cola",
                      "base-ft-qnli": "gchhablani/bert-base-cased-finetuned-qnli",
                        "large-ft-qnli": "yoshitomo-matsubara/bert-large-uncased-qnli",
                      "base-ft-mnli": "yoshitomo-matsubara/bert-base-uncased-mnli",
                        "large-ft-mnli": "yoshitomo-matsubara/bert-large-uncased-mnli",
                      "base-ft-rte": "anirudh21/bert-base-uncased-finetuned-rte",
                        "large-ft-rte": "yoshitomo-matsubara/bert-large-uncased-rte",
                      "base-ft-qqp": "A-bhimany-u08/bert-base-cased-qqp",
                        "large-ft-qqp": "yoshitomo-matsubara/bert-large-uncased-qqp",
                      "base-ft-wnli": "gchhablani/bert-base-cased-finetuned-wnli",
                        "large-ft-wnli": "yoshitomo-matsubara/bert-large-uncased-wnli",
                      }

parser = argparse.ArgumentParser()
parser.add_argument("--device", "-d", type=str, default='0')
parser.add_argument("--task", "-t", type=str, default='sst2', choices=TASKS)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr-decay", type=float, default=0.9)
parser.add_argument("--gp", type=float, default=10.0)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--model", type=str, choices=huggingface_models.keys())
parser.add_argument("--name", type=str, default="")
parser.add_argument("--num_layers", "-nl", type=int)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--use-sgd", action="store_true")
parser.add_argument("--fraction", type=float, default=1.0)

args = parser.parse_args()

device = torch.device("cuda:" + args.device)

directory = f'/scr/models/LC/models_archive/Bert/{args.model}/'
if not os.path.exists(directory):
    os.makedirs(directory)
filename = directory + str(args.num_layers - 1) + ("" if args.name=="" else "_") + args.name + "_" + args.task + ".pth"

model_base, tokenizer = get_classification_bert_model(pre_trained_model_name=huggingface_models[args.model])
model_GP = get_LC_model_bert(model_base, num_GP_layers=args.num_layers - 1).to(device)

train_dataloader, validation_dataloader, dataset = get_glue_task_dataset(args.task, tokenizer)
if os.path.exists(filename):
	try:
		model_GP.load_state_dict(torch.load(filename, map_location=device))
	except RuntimeError:
		model_base.load_state_dict(torch.load(filename, map_location=device))
		model_GP = get_LC_model_bert(model_base, num_GP_layers=args.num_layers - 1).to(device)
	print(filename)

filename = directory + str(args.num_layers) + ("" if args.name=="" else "_") + args.name + "_" +  args.task + ".pth"
filebacklog = directory + str(args.num_layers) + ("" if args.name=="" else "_") + args.name + "_" +  args.task + "_back.pth"

model_GP = get_LC_model_bert(model_GP, num_GP_layers=1)

if os.path.exists(filename):
	try:
		model_GP.load_state_dict(torch.load(filename, map_location=device))
	except RuntimeError:
		model_base.load_state_dict(torch.load(filename, map_location=device))
		model_GP = get_LC_model_bert(model_base, num_GP_layers=args.num_layers).to(device)
	
	print(filebacklog)
	print(filename)

	torch.save(model_GP.state_dict(), filebacklog)

get_layer_gates_loss(model_GP)

fraction_of_train = dataset["train"].__len__() * args.fraction
# random select
dataset["train"] = dataset["train"].shuffle().select(range(int(fraction_of_train)))

train_with_trainer(model_GP, dataset["train"], dataset["validation" if args.task!="mnli" else "validation_matched"], epochs=args.epochs,
                    eval_batch_size=args.batch_size, train_batch_size=args.batch_size, gp_weight=args.gp, learning_rate=args.lr, task=args.task, lr_decay=args.lr_decay, weight_decay=args.wd, use_sgd=args.use_sgd)
torch.save(model_GP.state_dict(), filename)
print("eval: ", standard_evaluate(model_GP, dataset["validation" if args.task!="mnli" else "validation_matched"], args.task))
get_layer_gates_loss(model_GP)