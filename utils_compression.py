import copy
import time
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchvision.datasets import *
from torchvision.transforms import *
import torchvision.models as models
import timm

from torchprofile import profile_macs
import numpy as np

# # import Subset function of torchvision
# from torchvision.datasets.utils import 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp

from collapsible_MLP import CollapsibleMlp

import LiveTune as lt

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = 'center'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def eval(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss/len(dataloader)

def eval_top5(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
):
    model.eval()
    running_loss = 0.0
    top5_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.topk(5, 1, True, True)
            top5_acc += (predicted == labels.view(-1, 1)).sum().item()
            running_loss += loss.item()
    return top5_acc/len(dataloader.dataset)

def train(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        learning_rate: float,
        device: torch.device,
        epochs: int,
):
    train_losses = []
    val_losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr = lt.liveVar(learning_rate, "lr")
    old_lr = learning_rate
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader["train"]):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss/len(dataloader["train"]))
        val_losses.append(eval(model, dataloader["val"], criterion, device))
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_losses[-1]} | Val loss: {val_losses[-1]}")
        
        if old_lr != lr():
            optimizer = torch.optim.SGD(model.parameters(), lr=lr())
            old_lr = lr()
            
    return train_losses, val_losses


@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader, 
  verbose=True,
  device=None,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, 
                                disable=not verbose):
        # Move the data from CPU to GPU
        if device is None:
            inputs = inputs.cuda()
            targets = targets.cuda()
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()



def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


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

def get_model(model_name, num_classes=10, drop_rate=0.1, image_size=32, patch_size=4, num_layers=6, hidden_dim=512, num_heads=8, mlp_dim=2048, dropout=0.1):
    if model_name == "VGG16":
        return VGG16(num_classes=num_classes)
    elif model_name == "ViT":
        return models.VisionTransformer(image_size=image_size, patch_size=4, num_layers=6, num_classes=num_classes, hidden_dim=512, num_heads=8, mlp_dim=2048, dropout=0.1)
    elif model_name == "mixer":
        return timm.models.mlp_mixer.MlpMixer(num_classes=num_classes, patch_size=patch_size, img_size=image_size, drop_rate=0.1, mlp_layer=CollapsibleMlp)
    elif model_name == "timm_vit":
        return timm.models.vision_transformer.VisionTransformer(num_classes=num_classes, patch_size=patch_size, img_size=image_size, drop_rate=0.1, mlp_layer=CollapsibleMlp, depth=num_layers)
    return models.__dict__[model_name]()

def get_model_linear_loss(model, fraction=1.0):
    linear_loss = 0
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            linear_loss += module.linear_loss()
    return linear_loss

def get_model_collapsible_slopes(model, fraction=1.0):
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            if isinstance(module.act, nn.Identity):
                print(name, 1)
            else:
                print(name, module.act.weight.item())

def collapse_model(model, fraction=1.0, threshold=0.05, device=None):
    num_mlp_layers = len(list(model.named_modules()))
    for name, module in list(model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, CollapsibleMlp):
            print("Collapsing layer {}".format(name))
            module.collapse(threshold=threshold)
            

def change_module(model, name, module):
    name_list = name.split(".")
    if len(name_list) == 1:
        model._modules[name_list[0]] = module
    else:
        change_module(model._modules[name_list[0]], ".".join(name_list[1:]), module)


def get_collapsible_model(model, fraction=1.0, device=None):
    num_mlp_layers = len(list(model.named_modules()))
    copy_model = copy.deepcopy(model).to(device)
    for name, module in list(copy_model.named_modules())[::-1][:int(num_mlp_layers * fraction)]:
        if isinstance(module, timm.layers.Mlp):
            print("Collapsing layer {}".format(name))
            collapsibleMLP = CollapsibleMlp(module.fc1.in_features, module.fc2.in_features, module.fc2.out_features, act_layer=module.act, batch_norm=False, bias=module.fc2.bias, drop=module.drop2.p, use_conv=False)
            collapsibleMLP.load_from_Mlp(module)
            if device is not None:
                collapsibleMLP.to(device)
            change_module(copy_model, name, collapsibleMLP)
    return copy_model.to(device)


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def evaluate_model(model, dataloader, count_nonzero_only=False, device=None):
    model_test_accuracy = evaluate(model, dataloader['val'], device=device)
    # model_train_accuracy = evaluate(model, dataloader['train'], device=device)
    top5 = eval_top5(model, dataloader['val'], nn.CrossEntropyLoss(), device)
    model.eval()
    model_size = get_model_size(model, count_nonzero_only=count_nonzero_only)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model_macs = get_model_macs(model, dummy_input)
    print(f"model has test accuracy={model_test_accuracy:.2f}%")
    print(f"model has top5 accuracy={top5:.2f}%")
    # print(f"model has train accuracy={model_train_accuracy:.2f}%")
    print(f"model has size={model_size/MiB:.2f} MiB")
    print(f"model has macs={model_macs/1e9:.2f} Gmacs")
    average_time = 0
    with torch.no_grad():
        for i in range(1000):
            start = time.time()
            output = model(dummy_input)
            end = time.time()
            average_time += (end - start)

    print(f"average inference time is {average_time/1000:.4f} seconds")
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"model has {get_num_parameters(model, count_nonzero_only)/1e6:.2f} M parameters")
    model.train()


def get_dataloader(dataset_name="cifar100", batch_size=512, num_workers=10, model_name="VGG16"):
    
    if dataset_name == "imagenet":
        image_size = 224
        transforms = {
            "train": Compose([
                Resize(256),
                RandomCrop(image_size, padding=4),
                RandomHorizontalFlip(),
                RandomRotation(15),
                ToTensor(),
                # Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if model_name != "mixer" else Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]),
            "val": Compose([
                Resize(256),
                CenterCrop(image_size),
                ToTensor(),
                # Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if model_name != "mixer" else Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]),
        }
    elif dataset_name == "cifar100":
        image_size = 32
        transforms = {
            "train": Compose([
                RandomCrop(image_size, padding=4),
                RandomHorizontalFlip(),
                RandomRotation(15),
                ToTensor(),
                # Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]),
            "val": Compose([
                ToTensor(),
                # Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]),
        }
    elif dataset_name == "cifar10":
        image_size = 32
        transforms = {
            "train": Compose([
                RandomCrop(image_size, padding=4),
                RandomHorizontalFlip(),
                RandomRotation(15),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ]),
            "val": Compose([
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ]),
        }

    dataset = {}
    for split in ["train", "val"]:
       # get cifar100 dataset
        if dataset_name == "cifar100":
            dataset[split] = CIFAR100(
                root="data/cifar100",
                train=(split == "train"),
                download=True,
                transform=transforms[split],
            )
        elif dataset_name == "cifar10":
            dataset[split] = CIFAR10(
                root="data/cifar10",
                train=(split == "train"),
                download=True,
                transform=transforms[split],
            )
        elif dataset_name == "imagenet":
            dataset[split] = ImageNet(
                root="/mnt/ram_dataset/mnt/ram_dataset/imagenet/",
                split=split,
                transform=transforms[split],
            )
    dataloader = {}
    for split in ['train', 'val']:
        dataloader[split] = DataLoader(
            torch.utils.data.Subset(dataset[split], list(range(0, len(dataset[split]), 1 if split == 'train' else 1))),
            # dataset[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.3))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.4))
        
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.4))
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1))
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            # nn.Linear(7*7*512, 4096),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(num_parameters=1, init=0.1))
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(num_parameters=1, init=0.1),
            nn.Dropout(0.5))
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        # self.fc_small = nn.Sequential(
        #     nn.Dropout(0.5),
        #     # nn.Linear(7*7*512, num_classes))
        #     nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc_small(out)
        return out
    

def get_lc_loss(model, lc1, lc2):
    if isinstance(model, VGG16):
        slope1 = model._modules["fc"][2].weight
        slope2 = model._modules["fc1"][2].weight
        print(slope1)
        print(slope2)
        return lc1 * (slope1 - 1) ** 2 + lc2 * (slope2 - 1) ** 2
    else :
        raise NotImplementedError
    
def finetune(model, dataloader, lc, fraction, lr, epochs=1, device=None, verbose=None, teacher=None):
    criterion = nn.CrossEntropyLoss()
    teacher_criterion = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
    # Train
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_top5_acc = 0.0
        for i, (inputs, labels) in tqdm(enumerate(dataloader["train"])):
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(inputs)

            # Regularized loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

                teacher_loss = teacher_criterion(torch.log_softmax(outputs, dim=1), torch.softmax(teacher_outputs, dim=1))

                loss = .5 * criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction) * lc + .5 * teacher_loss
            else:
                loss = criterion(outputs, labels) + get_model_linear_loss(model, fraction=fraction) * lc

            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            top5 = torch.topk(outputs, 5, dim=1)[1]
            train_top5_acc += (top5 == labels.unsqueeze(1)).sum().item()
            if verbose():
                get_model_collapsible_slopes(model, fraction=fraction)
                print("val acc: ", evaluate(model, dataloader=dataloader['val'], device=device))

        train_loss /= len(dataloader["train"])
        train_acc /= len(dataloader["train"].dataset)
        train_top5_acc /= len(dataloader["train"].dataset)
        
        val_acc = evaluate(model, dataloader=dataloader['val'], device=device)
        

        # Print
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Train top5: {:.4f}, Val Acc: {:.4f}".format(epoch, train_loss, train_acc, train_top5_acc, val_acc))
        if lc != 0:
            get_model_collapsible_slopes(model, fraction=fraction)


    
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = CollapsibleMlp(25088, 4096, 1000, batch_norm=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()


def get_collapsible_model_vgg_torchvision(model, device, all=True):
    new_model = copy.deepcopy(model)
    if all:
        new_model._modules['classifier'][1] = nn.PReLU(num_parameters=1, init=0).to(device)
    new_model._modules['classifier'][4] = nn.PReLU(num_parameters=1, init=0).to(device)
    return new_model

def get_linear_loss_vgg_torchvision(models):
    loss = 0
    for name, module in models.named_modules():
        if isinstance(module, nn.PReLU):
            loss += (module.weight - 1) ** 2
    return loss

def get_slopes_vgg_torchvision(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.PReLU):
            print(name, module.weight.item())

def finetune_vgg_torchvision(model, dataloader, lc, lr, epochs=1, device=None):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
    # Train
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_top5_acc = 0.0
        for i, (inputs, labels) in tqdm(enumerate(dataloader["train"])):
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(inputs)

            # Regularized loss

            loss = criterion(outputs, labels) + get_linear_loss_vgg_torchvision(model) * lc

            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            top5 = torch.topk(outputs, 5, dim=1)[1]
            train_top5_acc += (top5 == labels.unsqueeze(1)).sum().item()
        get_slopes_vgg_torchvision(model)

        train_loss /= len(dataloader["train"])
        train_acc /= len(dataloader["train"].dataset)
        train_top5_acc /= len(dataloader["train"].dataset)
        
        val_acc = evaluate(model, dataloader=dataloader['val'], device=device)
        # Print
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Train top5: {:.4f}, Val Acc: {:.4f}".format(epoch, train_loss, train_acc, train_top5_acc, val_acc))


def finetune_vgg_torchvision_with_teacher(model, teacher, dataloader, lc, lr, epochs=1, device=None):
    criterion = nn.CrossEntropyLoss()
    teacher_criterion = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
    # Train
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_top5_acc = 0.0
        for i, (inputs, labels) in tqdm(enumerate(dataloader["train"])):
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(inputs)

            # Regularized loss
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            teacher_loss = teacher_criterion(torch.log_softmax(outputs, dim=1), torch.softmax(teacher_outputs, dim=1))

            loss = .5 * criterion(outputs, labels) + .5 * teacher_loss + get_linear_loss_vgg_torchvision(model) * lc

            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            top5 = torch.topk(outputs, 5, dim=1)[1]
            train_top5_acc += (top5 == labels.unsqueeze(1)).sum().item()
        get_slopes_vgg_torchvision(model)

        train_loss /= len(dataloader["train"])
        train_acc /= len(dataloader["train"].dataset)
        train_top5_acc /= len(dataloader["train"].dataset)
        
        val_acc = evaluate(model, dataloader=dataloader['val'], device=device)
        # Print
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Train top5: {:.4f}, Val Acc: {:.4f}".format(epoch, train_loss, train_acc, train_top5_acc, val_acc))
