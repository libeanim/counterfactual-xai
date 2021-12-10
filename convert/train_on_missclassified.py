import os
from datetime import datetime
import numpy as np
import torch
from torch.nn.modules.linear import Linear
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torch import nn
import torch.optim as optim


def get_accuracy(model, loader, cuda=True):
    correct = 0
    total = 0
    with torch.inference_mode():
        for data in loader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


assert torch.cuda.is_available()
print('CUDA is availbale', torch.cuda.is_available())


# Generate id to reference files/plots/images
dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')

# General output directory
RESULT_DIR = f'/mnt/qb/work/bethge/ahochlehnert48/results/resnet50new/subset/{dt_id}'
os.mkdir(RESULT_DIR)
print('Result dir:', RESULT_DIR)

config = dict(
    id = dt_id,
    random_seed = 42,
    model = "resnet50new",
    learning_rate = 0.01,
    # learning_rate = 0.5,
    momentum = 0.9,
    epochs = 5,  # 18
    log_steps = 100,
    # k = 5,
    weight_decay = 1e-4,
    batch_size = 64,
    base_model = None,
    # base_model = "/home/bethge/ahochlehnert48/results/resnet50new/2021-09-21T10:29:43.000481_final.pth",

    # Scheduler step size
    # step_size = 100,
    scheduler = 'MultiStepLR',
    milestones = [1, 3, 4, 5], #[10, 20, 30, 40, 50, 60]#[15, 30, 60, 90, 110, 140]
    gamma = 0.1,
    init_readout = None,
    train_subset = '/mnt/qb/work/bethge/ahochlehnert48/results/resnet50_wrong_train_indices.npy'
)

train_indices: np.ndarray = np.load(config["train_subset"])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = ImageFolder(
    '/mnt/qb/datasets/ImageNet2012/train',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)

# train_dataset = Subset(train_dataset, train_indices.astype(int).tolist())

# Change labels of trainable subset to range between 1000-1999
for i in train_indices.astype(int).tolist():
    train_dataset.imgs[i] = (train_dataset.imgs[i][0], train_dataset.imgs[i][1] + 1000)


# val_dataset = ImageFolder(
#     '/mnt/qb/datasets/ImageNet2012/val',
#     transforms.Compose([
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])
# )
    
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], num_workers=1)
# val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], num_workers=1)

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # type: ignore

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        last_layer = resnet50(pretrained=True).fc
        last_layer.weight.requires_grad = False
        last_layer.bias.requires_grad = False
        self.original_fc = last_layer
        
        self.new_fc = nn.Linear(2048, 1000)
    
    def forward(self, x):
        x = self.backbone(x)
        x_old = self.original_fc(x)
        x_new = self.new_fc(x)
        
        return torch.cat((x_old, x_new), 1)
        

model = TestModel()

import wandb
# wandb.init()
wandb.init(project='resnet-50-mpl', entity='libeanim', config=config)

wandb.watch(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                      weight_decay=config["weight_decay"])

assert config['scheduler'] is None or config['scheduler'] == 'MultiStepLR'

scheduler = None
if config['scheduler'] == "MultiStepLR":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config["milestones"], gamma=config["gamma"])
model.cuda()

for epoch in range(config["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0
    running_accuracy = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        preditions = torch.argmax(outputs, dim=1)
        running_accuracy += (torch.sum(preditions == labels) / outputs.shape[0]).cpu().detach().numpy()
        if i % config["log_steps"] == config["log_steps"] - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / config["log_steps"]))
            # test_accuracy = get_accuracy(model, val_loader)
            wandb.log({
                "loss": running_loss / config["log_steps"],
                "accuracy": running_accuracy / config["log_steps"],
                "lr": optimizer.param_groups[0]['lr']
            })
            running_loss = 0.0
            running_accuracy = 0
        
    # model.eval()
    # wandb.log({ "test_accuracy": get_accuracy(model, val_loader) })
    # model.train()

    # learning rate decay
    if scheduler is not None:
        scheduler.step()
    
    torch.save(model.state_dict(), f"{RESULT_DIR}/{ config['id'] }_{epoch}.pth")
# model.cpu()
torch.save(model.state_dict(), f"{RESULT_DIR}/{ config['id'] }_final.pth")
print('Finished Training')