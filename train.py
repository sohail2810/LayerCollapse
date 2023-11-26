import argparse
import pickle
import random
import numpy as np

from torch.optim import *

from utils import *

assert torch.cuda.is_available(), "CUDA support is not available."

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument("--wd", default=0.0, type=float, help="weight decay")
parser.add_argument('--lc1', default=0.2, type=float, help='lc1')
parser.add_argument('--lc2', default=0.2, type=float, help='lc2')
parser.add_argument('--lc3', default=0.1, type=float, help='lc3')
parser.add_argument('--lc4', default=0.1, type=float, help='lc4')
parser.add_argument('--lc5', default=0.01, type=float, help='lc5')
parser.add_argument('--lc6', default=0.01, type=float, help='lc6')
parser.add_argument('--lc7', default=0.01, type=float, help='lc7')
parser.add_argument('--lc8', default=0.01, type=float, help='lc8')
parser.add_argument('--lc9', default=0.01, type=float, help='lc9')
parser.add_argument('--lc10', default=0.01, type=float, help='lc10')
parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device')
parser.add_argument('--model', default="VGG16", type=str, help='model')
parser.add_argument('--dataset', default="cifar10", type=str, help='dataset')
parser.add_argument('--save_dir', default="./data/", type=str, help='save directory')
parser.add_argument('--save_name', default="test", type=str, help='save name')
parser.add_argument('--reg', default="none", type=str, help='regularization')
parser.add_argument('--reg_strength', default=0.0, type=float, help='regularization strength')
parser.add_argument('--load', default="", type=str, help='load model')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--use_scheduler', action="store_true", help='use scheduler')
parser.add_argument('--fraction', default=0.5, type=float, help='fraction')
parser.add_argument('--patch_size', default=4, type=int, help='patch size')
parser.add_argument('--image_size', default=32, type=int, help='image size')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# Set device
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
# Set dataset
dataloader = get_dataloader(args.dataset, args.batch_size)
# Set model
model = get_model(args.model, args.num_classes, patch_size=args.patch_size, image_size=args.image_size).to(device)
if args.load != "":
    model.load_state_dict(torch.load(args.load))

# Set optimizer
# make lr scheduler for cifar 100 and adamw optimizer
lr = lt.liveVar(args.lr, "lr")
wd = lt.liveVar(args.wd, "wd")
opt = SGD(model.parameters(), lr=lr(), weight_decay=wd(), momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(opt, milestones=[75, 150, 250, 350], gamma=0.3)

# Register live variables
lc1 = lt.liveVar(args.lc1, "lc1")
fraction = lt.liveVar(args.fraction, "fraction")
old_lts = {"lr": lr(), "wd": wd(), "lc1": lc1()}

# Set loss function
criterion = nn.CrossEntropyLoss()

# Set save trigger
save_trigger = lt.liveTrigger("save")

train_losses = []
train_accs = []

val_accs = []
train_top5_accs = []
val_top5_accs = []

# Train
for epoch in tqdm(range(args.epochs)):
    # Train
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_top5_acc = 0.0
    for i, (inputs, labels) in enumerate(dataloader["train"]):
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)

        # Regularized loss
        if args.reg == "LC" and (
                args.model == "VGG16" or args.model == "VGG11" or args.model == "VGG13" or args.model == "VGG19"):
            loss = criterion(outputs, labels) + model.get_linear_loss(fraction=fraction()) * lc1()
        elif args.reg == "LC" and args.model == "mixer":
            loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction()) * lc1()
        elif args.reg == "LC" and args.model == "timm_vit":
            loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction()) * lc1()
        elif args.reg == "LC" and args.model == "vit_pretrained":
            loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction()) * lc1()
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()
        top5 = torch.topk(outputs, 5, dim=1)[1]
        train_top5_acc += (top5 == labels.unsqueeze(1)).sum().item()

    get_model_collapsible_slopes(model, fraction=fraction())
    model.get_slopes(fraction=fraction())
    scheduler.step()

    train_loss /= len(dataloader["train"])
    train_acc /= len(dataloader["train"].dataset)
    train_top5_acc /= len(dataloader["train"].dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_top5_accs.append(train_top5_acc)

    # Validation
    # val_losses.append(eval(model, dataloader["val"], criterion, device))
    val_top5_accs.append(eval_top5(model, dataloader["val"], criterion, device))
    val_accs.append(evaluate(model, dataloader["val"], device=device))

    # Save
    if save_trigger() or epoch % 10 == 1:
        torch.save(model.state_dict(), args.save_dir + args.save_name + ".pth")
        with open(args.save_dir + args.save_name + ".pkl", "wb") as f:
            pickle.dump({
                "train_losses": train_losses,
                "train_accs": train_accs,
                # "val_losses": val_losses,
                "val_accs": val_accs,
                "train_top5": train_top5_accs,
                "val_top5": val_top5_accs,
            }, f)

    # Print
    print(
        "Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Train top5: {:.4f}, Val Acc: {:.4f}, Val top5: {:.4f}".format(
            epoch, train_losses[-1], train_accs[-1], train_top5_accs[-1], val_accs[-1], val_top5_accs[-1]))

    # Scheduler
    if args.use_scheduler and epoch + 1 % 2 == 0:
        with lr.lock:
            lr.var_value = lr.var_value * 0.5 ** (epoch // 20)
    # LiveTune
    if old_lts["lr"] != lr() or old_lts["wd"] != wd():
        old_lts["lr"] = lr()
        old_lts["wd"] = wd()
        opt = SGD(model.parameters(), lr=lr(), momentum=0.9, weight_decay=wd())
        # scheduler = lr_scheduler.MultiStepLR(opt, milestones=[50, 100, 150, 200], gamma=0.5)

torch.save(model.state_dict(), args.save_dir + args.save_name + ".pth")
with open(args.save_dir + args.save_name + ".pkl", "wb") as f:
    pickle.dump({
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_top5": train_top5_accs,
        "val_top5": val_top5_accs,
    }, f)
